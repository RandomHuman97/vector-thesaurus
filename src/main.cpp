#include <iostream>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <stdexcept>
class EmbeddingEncoder {
private:
    Ort::Env env;
    Ort::Session session;
    Ort::MemoryInfo memory_info;

public:
    explicit EmbeddingEncoder(const std::string& model_path)
        : env(ORT_LOGGING_LEVEL_WARNING, "ThesaurusEncoder"),
          session(env, model_path.c_str(), Ort::SessionOptions{}),
          memory_info(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)) {
        std::cout << "ONNX model loaded successfully into C++ memory!" << std::endl;
    }

    std::vector<float> encode(std::vector<int64_t> input_ids) {
        if (input_ids.empty()) {
            throw std::invalid_argument("input_ids must not be empty");
        }

        std::vector<int64_t> attention_mask(input_ids.size(), 1);
        std::vector<int64_t> token_type_ids(input_ids.size(), 0);
        const std::vector shape = {1, static_cast<int64_t>(input_ids.size())};

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, input_ids.data(), input_ids.size(), shape.data(), shape.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, attention_mask.data(), attention_mask.size(), shape.data(), shape.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, token_type_ids.data(), token_type_ids.size(), shape.data(), shape.size()));

        const char* input_names[] = {"input_ids", "attention_mask", "token_type_ids"};
        const char* output_names[] = {"last_hidden_state"};

        // run inference
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names, input_tensors.data(), input_tensors.size(),
            output_names, 1
        );

        auto* float_arr = output_tensors[0].GetTensorMutableData<float>();

        std::vector<float> embedding(float_arr, float_arr + 384);
        return embedding;
    }
};

int main() {

    auto embeddingEncoder = EmbeddingEncoder("../model.onnx");
    embeddingEncoder.encode({101, 3231, 102});
    for (int i = 1; i <= 5; i++) {
        std::cout << "i = " << i << std::endl;
    }

    return 0;
}
