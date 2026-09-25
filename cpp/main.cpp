#include "inference.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
  try {
    if (argc != 4) {
      std::cerr << "usage: amber_infer model.onnx input.f32 output_prefix\n";
      return 2;
    }
    amber::GeometryInference model(argv[1]);
    size_t count = 1;
    for (auto size : model.input_shape()) count *= static_cast<size_t>(size);
    std::vector<float> input(count);
    std::ifstream file(argv[2], std::ios::binary);
    if (!file.read(reinterpret_cast<char*>(input.data()), count * sizeof(float)) || file.peek() != EOF)
      throw std::runtime_error("Input file must contain exactly one normalized float32 tensor");
    auto prediction = model.run(input);
    auto save = [&](const char* name, const amber::Tensor& tensor) {
      std::ofstream output(std::string(argv[3]) + "." + name + ".f32", std::ios::binary);
      output.write(reinterpret_cast<const char*>(tensor.values.data()), tensor.values.size() * sizeof(float));
      if (!output) throw std::runtime_error("Could not write output tensor");
    };
    save("poses", prediction.poses); save("depth", prediction.depth);
    save("confidence", prediction.confidence); save("intrinsics", prediction.intrinsics);
    std::cout << "Inference completed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
