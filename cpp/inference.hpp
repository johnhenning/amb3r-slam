#pragma once
#include <onnxruntime_cxx_api.h>
#include <cstdint>
#include <string>
#include <vector>

namespace amber {
// All tensors are row-major float32. Poses are camera-to-world matrices.
struct Tensor {
  std::vector<int64_t> shape;
  std::vector<float> values;
};
struct Prediction {
  Tensor poses, depth, confidence, intrinsics;
};

// Own one instance per worker. Input preprocessing follows the .onnx.json
// contract: RGB -> resize -> divide by 255 -> ImageNet normalize -> BVCHW.
// This adapter runs the network only; it does not claim to port the SLAM graph.
class GeometryInference {
 public:
  explicit GeometryInference(const std::string& model_path, int threads = 2);
  const std::vector<int64_t>& input_shape() const { return input_shape_; }
  Prediction run(const std::vector<float>& normalized_images);
 private:
  Ort::Env environment_;
  Ort::SessionOptions options_;
  Ort::Session session_{nullptr};
  std::vector<int64_t> input_shape_;
};
}  // namespace amber
