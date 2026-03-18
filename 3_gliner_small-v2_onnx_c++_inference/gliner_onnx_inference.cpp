#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

struct WordInfo {
    int64_t start = 0;
    int64_t end = 0;
};

struct Entity {
    int64_t start_word = 0;
    int64_t end_word = 0;
    int64_t start_char = 0;
    int64_t end_char = 0;
    std::string label;
    float score = 0.0F;
};

struct Args {
    fs::path model_path;
    fs::path assets_dir;
    float threshold = 0.5F;
};

std::string ReadTextFile(const fs::path& file_path) {
    std::ifstream ifs(file_path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open file: " + file_path.string());
    }
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    return content;
}

std::unordered_map<std::string, int64_t> ReadMeta(const fs::path& meta_path) {
    std::ifstream ifs(meta_path);
    if (!ifs) {
        throw std::runtime_error("Failed to open metadata file: " + meta_path.string());
    }

    std::unordered_map<std::string, int64_t> meta;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const size_t pos = line.find('=');
        if (pos == std::string::npos) {
            continue;
        }
        meta[line.substr(0, pos)] = std::stoll(line.substr(pos + 1));
    }

    return meta;
}

int64_t RequiredMeta(const std::unordered_map<std::string, int64_t>& meta, const std::string& key) {
    const auto it = meta.find(key);
    if (it == meta.end()) {
        throw std::runtime_error("Missing metadata key: " + key);
    }
    return it->second;
}

size_t NumElements(const std::vector<int64_t>& shape) {
    size_t count = 1;
    for (const int64_t dim : shape) {
        if (dim <= 0) {
            throw std::runtime_error("Invalid tensor dimension: " + std::to_string(dim));
        }
        count *= static_cast<size_t>(dim);
    }
    return count;
}

template <typename T>
std::vector<T> ReadBinary(const fs::path& file_path, size_t expected_count) {
    std::ifstream ifs(file_path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open binary file: " + file_path.string());
    }

    std::vector<T> buffer(expected_count);
    ifs.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(expected_count * sizeof(T)));
    if (!ifs) {
        throw std::runtime_error("Failed to read expected bytes from: " + file_path.string());
    }

    return buffer;
}

std::string ShapeToString(const std::vector<int64_t>& shape) {
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        out += std::to_string(shape[i]);
        if (i + 1 < shape.size()) {
            out += ", ";
        }
    }
    out += "]";
    return out;
}

std::vector<std::string> ParseJsonStringArray(const std::string& json_text) {
    std::vector<std::string> values;
    bool in_string = false;
    bool escaping = false;
    std::string current;

    for (const char ch : json_text) {
        if (!in_string) {
            if (ch == '"') {
                in_string = true;
                current.clear();
            }
            continue;
        }

        if (escaping) {
            current.push_back(ch);
            escaping = false;
            continue;
        }

        if (ch == '\\') {
            escaping = true;
            continue;
        }

        if (ch == '"') {
            values.push_back(current);
            in_string = false;
            continue;
        }

        current.push_back(ch);
    }

    return values;
}

std::vector<WordInfo> BuildWordOffsets(const std::string& text) {
    std::vector<WordInfo> words;
    const int64_t n = static_cast<int64_t>(text.size());
    int64_t i = 0;

    while (i < n) {
        while (i < n && std::isspace(static_cast<unsigned char>(text[static_cast<size_t>(i)]))) {
            ++i;
        }
        if (i >= n) {
            break;
        }

        const int64_t start = i;
        while (i < n && !std::isspace(static_cast<unsigned char>(text[static_cast<size_t>(i)]))) {
            ++i;
        }
        words.push_back(WordInfo{start, i});
    }

    return words;
}

float Sigmoid(const float x) {
    if (x >= 0.0F) {
        const float z = std::exp(-x);
        return 1.0F / (1.0F + z);
    }
    const float z = std::exp(x);
    return z / (1.0F + z);
}

bool HasOverlap(const Entity& a, const Entity& b) {
    return !(a.end_word < b.start_word || b.end_word < a.start_word);
}

std::vector<Entity> GreedyNonOverlapping(std::vector<Entity> entities) {
    std::sort(entities.begin(), entities.end(), [](const Entity& x, const Entity& y) {
        if (x.score == y.score) {
            if (x.start_word == y.start_word) {
                return x.end_word < y.end_word;
            }
            return x.start_word < y.start_word;
        }
        return x.score > y.score;
    });

    std::vector<Entity> kept;
    kept.reserve(entities.size());
    for (const auto& candidate : entities) {
        bool overlaps = false;
        for (const auto& existing : kept) {
            if (HasOverlap(candidate, existing)) {
                overlaps = true;
                break;
            }
        }
        if (!overlaps) {
            kept.push_back(candidate);
        }
    }

    std::sort(kept.begin(), kept.end(), [](const Entity& x, const Entity& y) {
        if (x.start_char == y.start_char) {
            return x.end_char < y.end_char;
        }
        return x.start_char < y.start_char;
    });

    return kept;
}

std::string JsonEscape(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 8);

    for (const unsigned char ch : text) {
        switch (ch) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out.push_back(static_cast<char>(ch));
                break;
        }
    }

    return out;
}

Args ParseArgs(int argc, char* argv[]) {
    if (argc < 3) {
        throw std::runtime_error(
            "Usage: gliner_onnx_inference <model.onnx> <assets_dir> [--threshold 0.5]"
        );
    }

    Args args;
    args.model_path = fs::path(argv[1]);
    args.assets_dir = fs::path(argv[2]);

    for (int i = 3; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--threshold") {
            if (i + 1 >= argc) {
                throw std::runtime_error("--threshold requires a value");
            }
            args.threshold = std::stof(argv[++i]);
        } else {
            throw std::runtime_error("Unknown option: " + key);
        }
    }

    return args;
}

int main(int argc, char* argv[]) {
    try {
        const Args args = ParseArgs(argc, argv);

        const auto meta = ReadMeta(args.assets_dir / "input_meta.txt");

        const int64_t batch_size = RequiredMeta(meta, "batch_size");
        const int64_t sequence_length = RequiredMeta(meta, "sequence_length");
        const int64_t num_spans = RequiredMeta(meta, "num_spans");
        const int64_t text_length_dim = RequiredMeta(meta, "text_length_dim");
        const int64_t span_idx_last_dim = RequiredMeta(meta, "span_idx_last_dim");

        std::vector<int64_t> shape_bs_seq{batch_size, sequence_length};
        std::vector<int64_t> shape_text_lengths{batch_size, text_length_dim};
        std::vector<int64_t> shape_span_idx{batch_size, num_spans, span_idx_last_dim};
        std::vector<int64_t> shape_span_mask{batch_size, num_spans};

        auto input_ids = ReadBinary<int64_t>(args.assets_dir / "input_ids.bin", NumElements(shape_bs_seq));
        auto attention_mask = ReadBinary<int64_t>(args.assets_dir / "attention_mask.bin", NumElements(shape_bs_seq));
        auto words_mask = ReadBinary<int64_t>(args.assets_dir / "words_mask.bin", NumElements(shape_bs_seq));
        auto text_lengths = ReadBinary<int64_t>(args.assets_dir / "text_lengths.bin", NumElements(shape_text_lengths));
        auto span_idx = ReadBinary<int64_t>(args.assets_dir / "span_idx.bin", NumElements(shape_span_idx));
        auto span_mask_u8 = ReadBinary<uint8_t>(args.assets_dir / "span_mask.bin", NumElements(shape_span_mask));

        const size_t span_mask_count = span_mask_u8.size();
        std::unique_ptr<bool[]> span_mask_bool(new bool[span_mask_count]);
        for (size_t i = 0; i < span_mask_count; ++i) {
            span_mask_bool[i] = span_mask_u8[i] != 0;
        }

        const std::string sample_text = ReadTextFile(args.assets_dir / "sample_text.txt");
        const std::vector<WordInfo> words = BuildWordOffsets(sample_text);
        const std::vector<std::string> labels = ParseJsonStringArray(ReadTextFile(args.assets_dir / "labels.json"));

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "gliner_cpp_onnx");
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        Ort::Session session(env, args.model_path.c_str(), session_options);
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::array<const char*, 6> input_names = {
            "input_ids",
            "attention_mask",
            "words_mask",
            "text_lengths",
            "span_idx",
            "span_mask",
        };

        std::array<Ort::Value, 6> input_tensors = {
            Ort::Value::CreateTensor<int64_t>(
                mem_info, input_ids.data(), input_ids.size(), shape_bs_seq.data(), shape_bs_seq.size()),
            Ort::Value::CreateTensor<int64_t>(
                mem_info, attention_mask.data(), attention_mask.size(), shape_bs_seq.data(), shape_bs_seq.size()),
            Ort::Value::CreateTensor<int64_t>(
                mem_info, words_mask.data(), words_mask.size(), shape_bs_seq.data(), shape_bs_seq.size()),
            Ort::Value::CreateTensor<int64_t>(
                mem_info, text_lengths.data(), text_lengths.size(), shape_text_lengths.data(), shape_text_lengths.size()),
            Ort::Value::CreateTensor<int64_t>(
                mem_info, span_idx.data(), span_idx.size(), shape_span_idx.data(), shape_span_idx.size()),
            Ort::Value::CreateTensor<bool>(
                mem_info, span_mask_bool.get(), span_mask_count, shape_span_mask.data(), shape_span_mask.size()),
        };

        const char* output_names[] = {"logits"};
        auto outputs = session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names,
            1
        );

        if (outputs.empty()) {
            throw std::runtime_error("ONNX session returned no outputs.");
        }

        const Ort::Value& logits = outputs.front();
        const auto info = logits.GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> out_shape = info.GetShape();
        if (out_shape.size() != 4) {
            throw std::runtime_error("Unexpected logits rank: " + std::to_string(out_shape.size()));
        }
        if (out_shape[0] != 1) {
            throw std::runtime_error("Only batch size 1 is supported by this executable.");
        }

        const int64_t L = out_shape[1];
        const int64_t K = out_shape[2];
        const int64_t C = out_shape[3];
        if (L <= 0 || K <= 0 || C <= 0) {
            throw std::runtime_error(
                "Output logits has an empty dimension. Ensure input assets were generated with GLiNER preprocessing."
            );
        }

        const size_t out_count = info.GetElementCount();
        const float* out_data = logits.GetTensorData<float>();

        std::vector<Entity> candidates;
        candidates.reserve(static_cast<size_t>(L * K));

        const int64_t usable_words = std::min<int64_t>(static_cast<int64_t>(words.size()), L);
        for (int64_t s = 0; s < usable_words; ++s) {
            for (int64_t k = 0; k < K; ++k) {
                const int64_t flat_span = s * K + k;
                if (flat_span < 0 || flat_span >= static_cast<int64_t>(span_mask_count)) {
                    continue;
                }
                if (!span_mask_bool[static_cast<size_t>(flat_span)]) {
                    continue;
                }

                const int64_t e = s + k;
                if (e >= usable_words) {
                    continue;
                }

                int64_t best_class = 0;
                float best_score = -1.0F;
                for (int64_t c = 0; c < C; ++c) {
                    const size_t index = static_cast<size_t>(((s * K) + k) * C + c);
                    if (index >= out_count) {
                        throw std::runtime_error("Computed logits index out of bounds.");
                    }
                    const float score = Sigmoid(out_data[index]);
                    if (score > best_score) {
                        best_score = score;
                        best_class = c;
                    }
                }

                if (best_score < args.threshold) {
                    continue;
                }

                const std::string label =
                    (best_class >= 0 && best_class < static_cast<int64_t>(labels.size()))
                        ? labels[static_cast<size_t>(best_class)]
                        : ("class_" + std::to_string(best_class));

                const int64_t start_char = words[static_cast<size_t>(s)].start;
                const int64_t end_char = words[static_cast<size_t>(e)].end;
                if (start_char < 0 || end_char <= start_char || end_char > static_cast<int64_t>(sample_text.size())) {
                    continue;
                }

                candidates.push_back(Entity{
                    s,
                    e,
                    start_char,
                    end_char,
                    label,
                    best_score,
                });
            }
        }

        const std::vector<Entity> entities = GreedyNonOverlapping(std::move(candidates));

        std::cout << "Inference succeeded.\n";
        std::cout << "Input shape (input_ids): " << ShapeToString(shape_bs_seq) << "\n";
        std::cout << "Output shape (logits): " << ShapeToString(out_shape) << "\n";
        std::cout << "Detected entities: " << entities.size() << "\n";

        std::cout << "{\n";
        std::cout << "  \"text\": \"" << JsonEscape(sample_text) << "\",\n";
        std::cout << "  \"entities\": [\n";
        for (size_t i = 0; i < entities.size(); ++i) {
            const auto& ent = entities[i];
            const std::string ent_text = sample_text.substr(
                static_cast<size_t>(ent.start_char),
                static_cast<size_t>(ent.end_char - ent.start_char)
            );

            std::cout << "    {\"start\": " << ent.start_char << ", \"end\": " << ent.end_char
                      << ", \"text\": \"" << JsonEscape(ent_text) << "\", \"label\": \""
                      << JsonEscape(ent.label) << "\", \"score\": "
                      << std::fixed << std::setprecision(6) << ent.score << "}";
            if (i + 1 < entities.size()) {
                std::cout << ",";
            }
            std::cout << "\n";
        }
        std::cout << "  ]\n";
        std::cout << "}\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
