#include "inference.hpp"
#include <cmath>
#include <stdexcept>

namespace amber {
namespace {
size_t elements(const std::vector<int64_t>& shape) {
  size_t count = 1;
  for (auto dimension : shape) {
    if (dimension <= 0) throw std::invalid_argument("Expected fixed positive tensor dimensions");
    count *= static_cast<size_t>(dimension);
  }
  return count;
}
Tensor copy_tensor(const Ort::Value& value) {
  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    throw std::runtime_error("Model output must be float32");
  const float* data = value.GetTensorData<float>();
  return {info.GetShape(), std::vector<float>(data, data + info.GetElementCount())};
}
}  // namespace

GeometryInference::GeometryInference(const std::string& model_path, int threads)
    : environment_(ORT_LOGGING_LEVEL_WARNING, "amber") {
  if (threads <= 0) throw std::invalid_argument("Thread count must be positive");
  options_.SetIntraOpNumThreads(threads);
  options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_ = Ort::Session(environment_, model_path.c_str(), options_);
  if (session_.GetInputCount() != 1 || session_.GetOutputCount() != 4)
    throw std::invalid_argument("Expected Amber single-input/four-output model contract");
  input_shape_ = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  if (input_shape_.size() != 5 || input_shape_[0] != 1 || input_shape_[2] != 3)
    throw std::invalid_argument("Expected input shape [1, views, 3, height, width]");
  elements(input_shape_);
}
Prediction GeometryInference::run(const std::vector<float>& images) {
  if (images.size() != elements(input_shape_))
    throw std::invalid_argument("Input size differs from model contract");
  for (float value : images)
    if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite input pixel");
  auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto tensor = Ort::Value::CreateTensor<float>(memory, const_cast<float*>(images.data()),
                                               images.size(), input_shape_.data(), input_shape_.size());
  const char* inputs[] = {"images"};
  const char* outputs[] = {"poses", "depth", "confidence", "intrinsics"};
  auto values = session_.Run(Ort::RunOptions{nullptr}, inputs, &tensor, 1, outputs, 4);
  return {copy_tensor(values[0]), copy_tensor(values[1]), copy_tensor(values[2]), copy_tensor(values[3])};
}
}  // namespace amber
