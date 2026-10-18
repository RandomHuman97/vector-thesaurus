#include <iostream>
#include <onnxruntime/onnxruntime_cxx_api.h>
class EmbeddingEncoder {
private:
    Ort::Env env;
    Ort::Session session;
    Ort::MemoryInfo memory_info;

public:
    EmbeddingEncoder(const std::string& model_path)
        : env(ORT_LOGGING_LEVEL_WARNING, "ThesaurusEncoder"),
          session(env, model_path.c_str(), Ort::SessionOptions{}),
          memory_info(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)) {
        std::cout << "ONNX model loaded successfully into C++ memory!" << std::endl;
    }

    std::vector<float> encode(const std::string& query_str) {

        std::vector<int64_t> input_ids = { /* tokenized IDs */ };
        std::vector<int64_t> attention_mask = { /* mask IDs */ };
        std::vector<int64_t> shape = {1, (int64_t)input_ids.size()};

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, input_ids.data(), input_ids.size(), shape.data(), shape.size()));

        const char* input_names[] = {"input_ids", "attention_mask"};
        const char* output_names[] = {"sentence_embedding"};

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

    const auto lang = "C++";
    std::cout << "Hello and welcome to " << lang << "!\n";

    for (int i = 1; i <= 5; i++) {
        std::cout << "i = " << i << std::endl;
    }

    return 0;
}