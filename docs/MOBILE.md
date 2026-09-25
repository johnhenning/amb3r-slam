# C++ and mobile deployment

The first target is Python on a desktop GPU using DA3. The C++ code currently
runs an exported network, **not the full SLAM graph**. No iOS/Android app or
device timing is claimed.

## Network boundary

Export emits ONNX, JSON shapes/preprocessing, a checksum, and float32 golden
input/output files. Input is fixed `[1,V,3,H,W]`; outputs are camera-to-world
poses, depth, confidence and intrinsics. Arrays are row-major float32.
Context selection and mapping state stay outside the network.

Export parity is validated for the experimental transformer. Exporting DA3
checkpoints is not yet verified: large-model operators, reference selection,
external-data files and delegates need separate work. The tiny smoke model is
not useful real-world localization.

## C++ build

Supply an ONNX Runtime release with matching `include/` and `lib/`:

```bash
cmake -S cpp -B build/native -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build build/native
build/native/amber_infer runs/smoke_training/model.onnx \
  runs/smoke_training/model.onnx.input.f32 runs/smoke_training/native
```

Compare native `.poses.f32`, `.depth.f32`, `.confidence.f32` and `.intrinsics.f32`
with export golden vectors using the JSON shapes. The adapter uses CPU ORT;
CoreML/NNAPI/QNN acceleration is not configured. Verify graph partitioning and
accuracy before enabling any device provider.

## Port sequence

1. Freeze preprocessing and compare PyTorch, Python ORT and native ORT on clips.
2. Port geometry/graph residuals with golden-vector tests; assess Eigen and a
   sparse optimizer while preserving scale, gauge and robust-loss conventions.
3. Define a C ABI around create/push/poll/reset/destroy. Keep exceptions inside
   the ABI, serialize backend jobs and bound sensor queues.
4. iOS: AVFoundation, image orientation/intrinsics, YUV conversion,
   Objective-C++ bridge, then CoreML/CPU provider validation.
5. Android: CameraX/Camera2, calibrated transforms, JNI/NDK, provider validation.
6. Measure sustained capture-to-pose p50/p95, memory, energy, thermal throttling,
   frame drops, drift and tracking loss on named devices.
7. Distill/quantize against an accuracy budget. The giant backend may require
   a server or a smaller replacement, which changes the accuracy target.

Rolling shutter, stabilization, camera switching, IMU fusion, ARKit/ARCore frame
conversions and relocalization UX remain app-level work. Sensor calibration,
synchronization and rectification are caller responsibilities.
