"""Build a self-contained Plotly comparison from completed, matched TUM runs."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from amber_slam.benchmark import trajectory_diagnostics
from amber_slam.evaluation import evaluate, read_tum
from amber_slam.experiment_tracking import resource_history


def report(root: Path, output: Path) -> None:
    paths = sorted(root.rglob("results.json"))
    if not paths:
        raise ValueError("No completed runs found")
    results = [json.loads(p.read_text()) for p in paths]
    for key in ("sequence", "input_manifest_sha256", "groundtruth_sha256"):
        if len({r[key] for r in results}) != 1:
            raise ValueError(f"Comparison requires matching {key}")
    if any(r["config"] != results[0]["config"] for r in results):
        raise ValueError("Comparison requires matching SLAM configuration")
    environments = [json.loads((p.parent.parent / "environment.json").read_text()) for p in paths]
    for key in ("checkpoint_hashes", "resolution", "torch_threads", "sources"):
        if any(e.get(key) != environments[0].get(key) for e in environments):
            raise ValueError(f"Comparison requires matching {key}")
    trajectory = go.Figure()
    errors = go.Figure()
    resources = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("CPU utilization (100% = one core)", "Resident memory"),
    )
    latency = go.Figure()
    comparison = make_subplots(
        rows=1,
        cols=4,
        subplot_titles=("Replay seconds", "CPU core-seconds", "Peak RSS MiB", "Corrected ATE m"),
    )
    rows = []
    colors = ["#2563eb", "#d97706", "#059669", "#9333ea"]
    reference_times, reference = read_tum(paths[0].parent / "reference.tum")
    trajectory.add_trace(
        go.Scatter3d(
            x=reference[:, 0, 3],
            y=reference[:, 1, 3],
            z=reference[:, 2, 3],
            mode="lines",
            name="Ground truth",
            line={"color": "#334155", "width": 4},
            text=[f"Timestamp {t:.6f}" for t in reference_times],
        )
    )
    for i, (path, result) in enumerate(zip(paths, results, strict=True)):
        directory = path.parent
        name = directory.parent.name
        color = colors[i % len(colors)]
        profile = json.loads((directory / "profile/summary.json").read_text())
        for label, filename, dash in [
            ("corrected", "trajectory.tum", "solid"),
            ("online", "trajectory_online.tum", "dash"),
        ]:
            if not (directory / filename).is_file():
                continue
            measured = evaluate(directory / "reference.tum", directory / filename)
            expected = result["metrics"][label]["ate_rmse_m"]
            if not np.isclose(measured["ate_rmse_m"], expected, rtol=1e-7, atol=1e-9):
                raise ValueError(f"Saved metrics disagree with trajectory: {path}")
            times, _, aligned, error = trajectory_diagnostics(
                directory / "reference.tum", directory / filename
            )
            title = f"{name} · {label}"
            elapsed = times - reference_times[0]
            trajectory.add_trace(
                go.Scatter3d(
                    x=aligned[:, 0, 3],
                    y=aligned[:, 1, 3],
                    z=aligned[:, 2, 3],
                    mode="lines",
                    name=title,
                    line={"color": color, "dash": dash},
                    customdata=np.column_stack([elapsed, error]),
                    hovertemplate="%{customdata[0]:.2f}s<br>Error %{customdata[1]:.3f} m<extra>%{fullData.name}</extra>",
                )
            )
            errors.add_trace(
                go.Scatter(x=elapsed, y=error, name=title, line={"color": color, "dash": dash})
            )
        history = list(resource_history(directory / "profile/samples.csv"))
        for row, key in [(1, "resources/cpu_percent"), (2, "resources/rss_mib")]:
            valid = [r for r in history if key in r]
            resources.add_trace(
                go.Scatter(
                    x=[r["replay/seconds"] for r in valid],
                    y=[r[key] for r in valid],
                    name=name,
                    legendgroup=name,
                    showlegend=row == 1,
                    line={"color": color},
                ),
                row=row,
                col=1,
            )
        with (directory / "latency.csv").open() as stream:
            samples = list(csv.DictReader(stream))
        latency.add_trace(
            go.Scatter(
                x=[int(float(r["frame"])) for r in samples],
                y=[float(r["push_latency_ms"]) for r in samples],
                name=name,
                line={"color": color},
            )
        )
        values = [
            result["statistics"]["replay_elapsed_s"],
            profile["cpu_total_s"],
            profile["rss_peak_bytes_sampled"] / 2**20,
            result["metrics"]["corrected"]["ate_rmse_m"],
        ]
        for col, value in enumerate(values, start=1):
            comparison.add_trace(
                go.Bar(x=[name], y=[value], name=name, marker_color=color, showlegend=False),
                row=1,
                col=col,
            )
        rows.append(
            "<tr><td>"
            + html.escape(name)
            + "</td>"
            + "".join(f"<td>{v:,.3f}</td>" for v in values)
            + f"<td>{result['statistics'].get('tracking_scale_retries', 0)}</td></tr>"
        )
    trajectory.update_layout(
        scene={
            "aspectmode": "data",
            "xaxis_title": "x (m)",
            "yaxis_title": "y (m)",
            "zaxis_title": "z (m)",
        }
    )
    errors.update_layout(xaxis_title="Sequence elapsed seconds", yaxis_title="Position error (m)")
    resources.update_xaxes(title_text="Replay elapsed seconds", row=2, col=1)
    resources.update_yaxes(title_text="CPU %", row=1, col=1)
    resources.update_yaxes(title_text="RSS MiB", row=2, col=1)
    latency.update_layout(xaxis_title="Sampled frame", yaxis_title="Push latency (ms)")
    figures = [
        ("Runtime, compute and accuracy", comparison),
        ("Trajectory: rotate and zoom", trajectory),
        ("Trajectory error", errors),
        ("CPU and memory", resources),
        ("Tracking latency", latency),
    ]
    sections = []
    for i, (title, figure) in enumerate(figures):
        figure.update_layout(
            template="plotly_white",
            height=620,
            title=title,
            margin={"l": 65, "r": 25, "t": 80, "b": 65},
        )
        sections.append(
            figure.to_html(
                full_html=False,
                include_plotlyjs=True if i == 0 else False,
                config={"responsive": True, "displaylogo": False},
            )
        )
    body = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>SLAM experiment comparison</title>
<style>body{font:16px system-ui;margin:24px auto;max-width:1300px;padding:0 16px;color:#17243b}
table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}
.note{background:#eff6ff;padding:16px;border-radius:8px}</style></head><body>
<h1>SLAM · SEQUENCE_LABEL</h1><p class="note">Recorded RGB replays. Individual trials are shown;
this is not a variance estimate or a real-time claim.
Each trajectory has its own full-trajectory Sim(3) alignment, which removes global scale error.
Replay excludes model download/loading. CPU and RAM cover the replay process; memory peaks are sampled.
Hover for values, drag to zoom, rotate the 3D plot, and click legends to toggle series.</p>
<table><tr><th>Run</th><th>Replay s</th><th>CPU core-s</th><th>Peak RSS MiB</th><th>Corrected ATE m</th><th>Scale retries</th></tr>"""
    body = body.replace("SEQUENCE_LABEL", html.escape(results[0]["sequence"]))
    body += "".join(rows) + "</table>" + "".join(sections)
    body += "<p>Metrics recomputed from saved trajectories. W&B destination: jlh15/slam.</p></body></html>"
    output.write_text(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new report output path")
    report(args.reports, args.output)
