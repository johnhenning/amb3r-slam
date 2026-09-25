# Data contract and dataset preparation

A manifest is JSON with `version: 1` and a nonempty `sequences` list. Each scene
has a unique `id`, explicit `split` (`train`, `val`, `test`) and ordered `frames`.
Paths are relative to the manifest directory or absolute. Example:

```json
{
  "version": 1,
  "sequences": [{
    "id": "office-01",
    "split": "test",
    "frames": [{
      "timestamp": 0.0,
      "rgb": "office/rgb/000.png",
      "depth": "office/depth/000.png",
      "depth_scale": 0.001,
      "intrinsics": [[500,0,320],[0,500,240],[0,0,1]],
      "c2w": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    }]
  }]
}
```

Replay only requires timestamp and RGB for monocular input. Training additionally
requires depth, intrinsics and camera-to-world ground truth. Depth must be
registered to RGB at identical dimensions; zero/nonfinite values are invalid.
Integer depth images are multiplied by `depth_scale`. NPY arrays are also accepted
and never loaded with pickle. Intrinsics must reflect image resizing/cropping.

Stereo records use `right_rgb` and positive `baseline_m`; both views must already
be rectified. The converter uses SGBM and does not estimate calibration. LiDAR
records use `lidar` pointing to an Nx3 NPY array in **camera coordinates, meters**.
Sensor extrinsics, time synchronization and deskew must be applied upstream.
Sparse LiDAR must provide enough valid projected depth for scale fitting.

## Existing converters

```bash
uv run --no-sync scripts/prepare_dataset.py tum /path/to/tum data/tum.json \
  --id tum-desk --intrinsics 517.3 516.5 318.6 255.3
uv run --no-sync scripts/prepare_dataset.py kitti /path/to/sequences/00 data/kitti00.json \
  --id kitti-00 --poses /path/to/poses/00.txt
```

The TUM intrinsics above are an example only: supply the values for the actual
camera/sequence. TUM default depth unit is 1/5000 m. KITTI conversion uses
rectified color cameras 2/3 and converts cam0 ground-truth poses to cam2.
Converters retain absolute paths to the original dataset; moving the dataset
requires regenerating the manifest. Calibration and all timestamps should be
reviewed before a benchmark campaign.

Synthetic generation creates analytic plane views with train/val/test scene
identities. It tests plumbing; it has insufficient diversity for useful SLAM
training or generalization claims. Real splits must be scene-disjoint across
all data sources, with duplicate scenes removed, not merely renamed IDs.
