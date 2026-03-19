#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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

namespace {

const std::vector<std::string> kLabels = {
    "Person",
    "Location",
    "Organization",
    "Date",
    "Time",
    "Animal",
    "Quantity",
    "Event",
    "LocationCountry",
    "LocationCity",
    "Shop",
    "CultureSite",
    "Building",
    "Duraion",
    "TimeDuration",
    "Sports",
    "Food",
    "Currency",
    "Law",
    "QuantityAge",
    "QunatityTemperature",
    "QuantityPrice",
    "EventSports",
    "EventFestival",
    "TermMedical",
    "TermSports",
};

const std::string kMetaspace = "\xE2\x96\x81";

struct Rune {
    uint32_t cp = 0;
    size_t byte_start = 0;
    size_t byte_end = 0;
};

struct WordSpan {
    std::string token;
    int64_t char_start = 0;
    int64_t char_end = 0;
    size_t byte_start = 0;
    size_t byte_end = 0;
};

struct InputRecord {
    int line = 0;
    std::string text;
};

struct Entity {
    int64_t start_word = 0;
    int64_t end_word = 0;
    int64_t start_char = 0;
    int64_t end_char = 0;
    size_t start_byte = 0;
    size_t end_byte = 0;
    std::string label;
    float score = 0.0F;
};

struct RuntimeConfig {
    int32_t max_len = 384;
    int32_t max_width = 12;
    int32_t cls_id = 1;
    int32_t sep_id = 2;
    int32_t pad_id = 0;
    int32_t unk_id = 3;
    int32_t class_token_index = 128002;
    std::string ent_token = "<<ENT>>";
    std::string sep_token = "<<SEP>>";
};

struct PreparedInput {
    std::string text;
    std::vector<WordSpan> words;

    std::vector<int64_t> input_ids;
    std::vector<int64_t> attention_mask;
    std::vector<int64_t> words_mask;
    std::vector<int64_t> text_lengths;  // [1]
    std::vector<int64_t> span_idx;      // flattened [num_spans, 2]
    std::vector<uint8_t> span_mask_u8;  // [num_spans]

    int64_t seq_len = 0;
    int64_t num_words = 0;
    int64_t num_spans = 0;
};

struct AppArgs {
    fs::path model_path = fs::path("..") / "2_gliner_small-v2_onnx_inference" / "onnx_model" / "model_opset13.onnx";
    fs::path tokenizer_path = fs::path("..") / "2_gliner_small-v2_onnx_inference" / "onnx_model" / "tokenizer.json";
    fs::path config_path = fs::path("..") / "2_gliner_small-v2_onnx_inference" / "onnx_model" / "gliner_config.json";
    fs::path input_jsonl = fs::path("..") / "sample.jsonl";
    fs::path output_jsonl = fs::path("onnx_cpp_output.jsonl");
    float threshold = 0.5F;
};

struct VocabPiece {
    std::string piece;
    float score = 0.0F;
    int32_t id = 0;
};

size_t SkipWs(const std::string& s, size_t pos) {
    while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos])) != 0) {
        ++pos;
    }
    return pos;
}

void AppendUtf8(uint32_t cp, std::string* out) {
    if (cp <= 0x7F) {
        out->push_back(static_cast<char>(cp));
        return;
    }
    if (cp <= 0x7FF) {
        out->push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        return;
    }
    if (cp <= 0xFFFF) {
        out->push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        return;
    }
    if (cp <= 0x10FFFF) {
        out->push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out->push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        return;
    }
    throw std::runtime_error("Invalid Unicode code point");
}

uint32_t ParseHex4(const std::string& s, size_t pos) {
    if (pos + 4 > s.size()) {
        throw std::runtime_error("Invalid \\u escape");
    }
    uint32_t value = 0;
    for (size_t i = 0; i < 4; ++i) {
        const char ch = s[pos + i];
        value <<= 4;
        if (ch >= '0' && ch <= '9') {
            value |= static_cast<uint32_t>(ch - '0');
        } else if (ch >= 'a' && ch <= 'f') {
            value |= static_cast<uint32_t>(10 + ch - 'a');
        } else if (ch >= 'A' && ch <= 'F') {
            value |= static_cast<uint32_t>(10 + ch - 'A');
        } else {
            throw std::runtime_error("Invalid hex in \\u escape");
        }
    }
    return value;
}

std::string ParseJsonString(const std::string& s, size_t* pos) {
    if (*pos >= s.size() || s[*pos] != '"') {
        throw std::runtime_error("Expected JSON string");
    }
    ++(*pos);

    std::string out;
    while (*pos < s.size()) {
        const char ch = s[*pos];
        ++(*pos);

        if (ch == '"') {
            return out;
        }

        if (ch != '\\') {
            out.push_back(ch);
            continue;
        }

        if (*pos >= s.size()) {
            throw std::runtime_error("Invalid JSON escape");
        }

        const char esc = s[*pos];
        ++(*pos);
        switch (esc) {
            case '"':
                out.push_back('"');
                break;
            case '\\':
                out.push_back('\\');
                break;
            case '/':
                out.push_back('/');
                break;
            case 'b':
                out.push_back('\b');
                break;
            case 'f':
                out.push_back('\f');
                break;
            case 'n':
                out.push_back('\n');
                break;
            case 'r':
                out.push_back('\r');
                break;
            case 't':
                out.push_back('\t');
                break;
            case 'u': {
                uint32_t cp = ParseHex4(s, *pos);
                *pos += 4;

                if (cp >= 0xD800 && cp <= 0xDBFF) {
                    if (*pos + 6 <= s.size() && s[*pos] == '\\' && s[*pos + 1] == 'u') {
                        const uint32_t low = ParseHex4(s, *pos + 2);
                        if (low >= 0xDC00 && low <= 0xDFFF) {
                            cp = 0x10000 + (((cp - 0xD800) << 10) | (low - 0xDC00));
                            *pos += 6;
                        }
                    }
                }

                AppendUtf8(cp, &out);
                break;
            }
            default:
                throw std::runtime_error("Unsupported JSON escape sequence");
        }
    }

    throw std::runtime_error("Unterminated JSON string");
}

double ParseJsonNumber(const std::string& s, size_t* pos) {
    *pos = SkipWs(s, *pos);
    const size_t start = *pos;

    if (*pos < s.size() && (s[*pos] == '-' || s[*pos] == '+')) {
        ++(*pos);
    }
    while (*pos < s.size() && std::isdigit(static_cast<unsigned char>(s[*pos])) != 0) {
        ++(*pos);
    }
    if (*pos < s.size() && s[*pos] == '.') {
        ++(*pos);
        while (*pos < s.size() && std::isdigit(static_cast<unsigned char>(s[*pos])) != 0) {
            ++(*pos);
        }
    }
    if (*pos < s.size() && (s[*pos] == 'e' || s[*pos] == 'E')) {
        ++(*pos);
        if (*pos < s.size() && (s[*pos] == '-' || s[*pos] == '+')) {
            ++(*pos);
        }
        while (*pos < s.size() && std::isdigit(static_cast<unsigned char>(s[*pos])) != 0) {
            ++(*pos);
        }
    }

    if (start == *pos) {
        throw std::runtime_error("Expected JSON number");
    }
    return std::stod(s.substr(start, *pos - start));
}

size_t FindMatching(const std::string& s, size_t start, char open_ch, char close_ch) {
    if (start >= s.size() || s[start] != open_ch) {
        throw std::runtime_error("FindMatching called with invalid start");
    }

    int depth = 0;
    bool in_string = false;
    bool escape = false;
    for (size_t i = start; i < s.size(); ++i) {
        const char ch = s[i];

        if (in_string) {
            if (escape) {
                escape = false;
                continue;
            }
            if (ch == '\\') {
                escape = true;
                continue;
            }
            if (ch == '"') {
                in_string = false;
            }
            continue;
        }

        if (ch == '"') {
            in_string = true;
            continue;
        }
        if (ch == open_ch) {
            ++depth;
            continue;
        }
        if (ch == close_ch) {
            --depth;
            if (depth == 0) {
                return i;
            }
        }
    }

    throw std::runtime_error("Matching bracket not found");
}

std::string ReadTextFile(const fs::path& file_path) {
    std::ifstream ifs(file_path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open file: " + file_path.string());
    }
    return std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
}

std::optional<int64_t> ExtractIntByKey(const std::string& json, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const size_t key_pos = json.find(needle);
    if (key_pos == std::string::npos) {
        return std::nullopt;
    }
    const size_t colon = json.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        return std::nullopt;
    }
    size_t pos = SkipWs(json, colon + 1);
    bool neg = false;
    if (pos < json.size() && json[pos] == '-') {
        neg = true;
        ++pos;
    }
    size_t start = pos;
    while (pos < json.size() && std::isdigit(static_cast<unsigned char>(json[pos])) != 0) {
        ++pos;
    }
    if (start == pos) {
        return std::nullopt;
    }
    int64_t value = std::stoll(json.substr(start, pos - start));
    return neg ? -value : value;
}

std::optional<std::string> ExtractStringByKey(const std::string& json, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const size_t key_pos = json.find(needle);
    if (key_pos == std::string::npos) {
        return std::nullopt;
    }
    const size_t colon = json.find(':', key_pos + needle.size());
    if (colon == std::string::npos) {
        return std::nullopt;
    }
    size_t pos = SkipWs(json, colon + 1);
    if (pos >= json.size() || json[pos] != '"') {
        return std::nullopt;
    }
    return ParseJsonString(json, &pos);
}

class UnigramTokenizer {
public:
    void LoadFromTokenizerJson(const fs::path& tokenizer_json_path) {
        const std::string json = ReadTextFile(tokenizer_json_path);
        ParseAddedTokens(json);
        ParseUnigramModel(json);
    }

    int32_t GetSpecialTokenId(const std::string& token, int32_t fallback) const {
        const auto it = special_token_to_id_.find(token);
        if (it == special_token_to_id_.end()) {
            return fallback;
        }
        return it->second;
    }

    int32_t GetUnkId() const {
        return unk_id_;
    }

    std::vector<int32_t> EncodeWord(const std::string& word) const {
        const auto special_it = special_token_to_id_.find(word);
        if (special_it != special_token_to_id_.end()) {
            return {special_it->second};
        }

        const std::string normalized = NormalizeWord(word);
        if (normalized.empty()) {
            return {unk_id_};
        }

        const std::string source = kMetaspace + normalized;
        std::vector<int32_t> pieces = UnigramEncode(source);
        if (pieces.empty()) {
            pieces.push_back(unk_id_);
        }
        return pieces;
    }

private:
    void ParseAddedTokens(const std::string& json) {
        const size_t key_pos = json.find("\"added_tokens\"");
        if (key_pos == std::string::npos) {
            throw std::runtime_error("tokenizer.json missing added_tokens");
        }
        const size_t arr_start = json.find('[', key_pos);
        if (arr_start == std::string::npos) {
            throw std::runtime_error("tokenizer.json added_tokens parse failed");
        }
        const size_t arr_end = FindMatching(json, arr_start, '[', ']');

        size_t pos = arr_start + 1;
        while (pos < arr_end) {
            pos = SkipWs(json, pos);
            if (pos >= arr_end) {
                break;
            }
            if (json[pos] == ',') {
                ++pos;
                continue;
            }
            if (json[pos] != '{') {
                ++pos;
                continue;
            }

            const size_t obj_end = FindMatching(json, pos, '{', '}');
            const std::string obj = json.substr(pos, obj_end - pos + 1);

            const auto id = ExtractIntByKey(obj, "id");
            const auto content = ExtractStringByKey(obj, "content");
            if (id.has_value() && content.has_value()) {
                special_token_to_id_[*content] = static_cast<int32_t>(*id);
            }

            pos = obj_end + 1;
        }
    }

    void ParseUnigramModel(const std::string& json) {
        const size_t model_key = json.find("\"model\"");
        if (model_key == std::string::npos) {
            throw std::runtime_error("tokenizer.json missing model section");
        }
        const size_t model_obj_start = json.find('{', model_key);
        if (model_obj_start == std::string::npos) {
            throw std::runtime_error("tokenizer.json model parse failed");
        }
        const size_t model_obj_end = FindMatching(json, model_obj_start, '{', '}');
        const std::string model_json = json.substr(model_obj_start, model_obj_end - model_obj_start + 1);

        const auto parsed_unk = ExtractIntByKey(model_json, "unk_id");
        if (parsed_unk.has_value()) {
            unk_id_ = static_cast<int32_t>(*parsed_unk);
        }

        const size_t vocab_key = model_json.find("\"vocab\"");
        if (vocab_key == std::string::npos) {
            throw std::runtime_error("tokenizer.json model missing vocab");
        }
        const size_t arr_start = model_json.find('[', vocab_key);
        if (arr_start == std::string::npos) {
            throw std::runtime_error("tokenizer.json vocab parse failed");
        }
        const size_t arr_end = FindMatching(model_json, arr_start, '[', ']');

        vocab_.clear();
        int32_t next_id = 0;
        size_t pos = arr_start + 1;
        while (pos < arr_end) {
            pos = SkipWs(model_json, pos);
            if (pos >= arr_end) {
                break;
            }
            if (model_json[pos] == ',') {
                ++pos;
                continue;
            }
            if (model_json[pos] == ']') {
                break;
            }
            if (model_json[pos] != '[') {
                ++pos;
                continue;
            }

            ++pos;
            pos = SkipWs(model_json, pos);
            std::string piece = ParseJsonString(model_json, &pos);

            pos = SkipWs(model_json, pos);
            if (pos >= model_json.size() || model_json[pos] != ',') {
                throw std::runtime_error("Invalid vocab entry (missing comma)");
            }
            ++pos;

            pos = SkipWs(model_json, pos);
            const float score = static_cast<float>(ParseJsonNumber(model_json, &pos));

            pos = SkipWs(model_json, pos);
            if (pos >= model_json.size() || model_json[pos] != ']') {
                throw std::runtime_error("Invalid vocab entry (missing ] )");
            }
            ++pos;

            vocab_.push_back(VocabPiece{piece, score, next_id++});
        }

        for (auto& bucket : pieces_by_first_byte_) {
            bucket.clear();
        }

        for (const auto& piece : vocab_) {
            if (piece.piece.empty()) {
                continue;
            }
            const unsigned char first = static_cast<unsigned char>(piece.piece[0]);
            pieces_by_first_byte_[first].push_back(piece.id);
        }
    }

    std::string NormalizeWord(const std::string& word) const {
        std::string out;
        out.reserve(word.size());

        bool prev_space = false;
        for (const unsigned char ch : word) {
            char c = static_cast<char>(ch);
            if (c == '\n' || c == '\r' || c == '\t') {
                c = ' ';
            }
            if (c == ' ') {
                if (!prev_space) {
                    out.push_back(' ');
                    prev_space = true;
                }
            } else {
                out.push_back(c);
                prev_space = false;
            }
        }

        while (!out.empty() && out.back() == ' ') {
            out.pop_back();
        }

        return out;
    }

    std::vector<int32_t> UnigramEncode(const std::string& source) const {
        const size_t n = source.size();
        if (n == 0) {
            return {};
        }

        const double neg_inf = -1e30;
        std::vector<double> best(n + 1, neg_inf);
        std::vector<int32_t> prev_pos(n + 1, -1);
        std::vector<int32_t> prev_piece_id(n + 1, -1);
        best[0] = 0.0;

        for (size_t i = 0; i < n; ++i) {
            if (best[i] <= neg_inf / 2) {
                continue;
            }

            const unsigned char first = static_cast<unsigned char>(source[i]);
            const auto& candidate_ids = pieces_by_first_byte_[first];

            for (const int32_t piece_id : candidate_ids) {
                const auto& piece = vocab_[static_cast<size_t>(piece_id)];
                const size_t len = piece.piece.size();
                if (i + len > n) {
                    continue;
                }
                if (source.compare(i, len, piece.piece) != 0) {
                    continue;
                }

                const size_t j = i + len;
                const double cand = best[i] + static_cast<double>(piece.score);
                if (cand > best[j]) {
                    best[j] = cand;
                    prev_pos[j] = static_cast<int32_t>(i);
                    prev_piece_id[j] = piece_id;
                }
            }
        }

        if (best[n] <= neg_inf / 2) {
            return {unk_id_};
        }

        std::vector<int32_t> reversed_ids;
        size_t cursor = n;
        while (cursor > 0) {
            const int32_t pid = prev_piece_id[cursor];
            const int32_t ppos = prev_pos[cursor];
            if (pid < 0 || ppos < 0) {
                return {unk_id_};
            }
            reversed_ids.push_back(pid);
            cursor = static_cast<size_t>(ppos);
        }

        std::reverse(reversed_ids.begin(), reversed_ids.end());
        return reversed_ids;
    }

    std::unordered_map<std::string, int32_t> special_token_to_id_;
    std::vector<VocabPiece> vocab_;
    int32_t unk_id_ = 3;
    std::array<std::vector<int32_t>, 256> pieces_by_first_byte_;
};

bool DecodeOneRune(const std::string& text, size_t pos, Rune* out) {
    if (pos >= text.size()) {
        return false;
    }

    const unsigned char c0 = static_cast<unsigned char>(text[pos]);
    out->byte_start = pos;

    if (c0 < 0x80) {
        out->cp = c0;
        out->byte_end = pos + 1;
        return true;
    }

    if ((c0 & 0xE0) == 0xC0 && pos + 1 < text.size()) {
        const unsigned char c1 = static_cast<unsigned char>(text[pos + 1]);
        if ((c1 & 0xC0) == 0x80) {
            out->cp = static_cast<uint32_t>(((c0 & 0x1F) << 6) | (c1 & 0x3F));
            out->byte_end = pos + 2;
            return true;
        }
    }

    if ((c0 & 0xF0) == 0xE0 && pos + 2 < text.size()) {
        const unsigned char c1 = static_cast<unsigned char>(text[pos + 1]);
        const unsigned char c2 = static_cast<unsigned char>(text[pos + 2]);
        if ((c1 & 0xC0) == 0x80 && (c2 & 0xC0) == 0x80) {
            out->cp = static_cast<uint32_t>(((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F));
            out->byte_end = pos + 3;
            return true;
        }
    }

    if ((c0 & 0xF8) == 0xF0 && pos + 3 < text.size()) {
        const unsigned char c1 = static_cast<unsigned char>(text[pos + 1]);
        const unsigned char c2 = static_cast<unsigned char>(text[pos + 2]);
        const unsigned char c3 = static_cast<unsigned char>(text[pos + 3]);
        if ((c1 & 0xC0) == 0x80 && (c2 & 0xC0) == 0x80 && (c3 & 0xC0) == 0x80) {
            out->cp = static_cast<uint32_t>(
                ((c0 & 0x07) << 18) | ((c1 & 0x3F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F)
            );
            out->byte_end = pos + 4;
            return true;
        }
    }

    // Invalid UTF-8 byte sequence: keep byte as-is.
    out->cp = c0;
    out->byte_end = pos + 1;
    return true;
}

std::vector<Rune> DecodeRunes(const std::string& text) {
    std::vector<Rune> runes;
    size_t pos = 0;
    while (pos < text.size()) {
        Rune r;
        DecodeOneRune(text, pos, &r);
        runes.push_back(r);
        pos = r.byte_end;
    }
    return runes;
}

bool IsWhitespaceCp(uint32_t cp) {
    switch (cp) {
        case 0x0009:
        case 0x000A:
        case 0x000B:
        case 0x000C:
        case 0x000D:
        case 0x0020:
        case 0x0085:
        case 0x00A0:
        case 0x1680:
        case 0x2028:
        case 0x2029:
        case 0x202F:
        case 0x205F:
        case 0x3000:
            return true;
        default:
            break;
    }
    if (cp >= 0x2000 && cp <= 0x200A) {
        return true;
    }
    return false;
}

bool IsLikelyPunctuationCp(uint32_t cp) {
    switch (cp) {
        case '.':
        case ',':
        case '!':
        case '?':
        case ':':
        case ';':
        case '(': 
        case ')':
        case '[':
        case ']':
        case '{':
        case '}':
        case '"':
        case '\'':
        case '/':
        case '\\':
        case '|':
        case '*':
        case '&':
        case '^':
        case '%':
        case '$':
        case '#':
        case '@':
        case '~':
        case '`':
        case '+':
        case '=':
            return true;
        default:
            break;
    }

    // Common CJK punctuation.
    switch (cp) {
        case 0x3001:  // ideographic comma
        case 0x3002:  // ideographic full stop
        case 0xFF0C:  // fullwidth comma
        case 0xFF0E:  // fullwidth full stop
        case 0xFF01:  // fullwidth exclamation mark
        case 0xFF1F:  // fullwidth question mark
        case 0xFF1A:  // fullwidth colon
        case 0xFF1B:  // fullwidth semicolon
        case 0x2014:  // em dash
        case 0x2026:  // horizontal ellipsis
        case 0x3008:
        case 0x3009:
        case 0x300C:
        case 0x300D:
        case 0x300E:
        case 0x300F:
        case 0x3010:
        case 0x3011:
            return true;
        default:
            return false;
    }
}

bool IsWordCp(uint32_t cp) {
    if (cp < 128) {
        return (std::isalnum(static_cast<unsigned char>(cp)) != 0) || cp == '_';
    }
    if (IsWhitespaceCp(cp) || IsLikelyPunctuationCp(cp)) {
        return false;
    }
    return true;
}

std::vector<WordSpan> SplitWordsWithOffsets(const std::string& text) {
    const std::vector<Rune> runes = DecodeRunes(text);
    std::vector<WordSpan> out;

    size_t i = 0;
    while (i < runes.size()) {
        if (IsWhitespaceCp(runes[i].cp)) {
            ++i;
            continue;
        }

        const size_t start = i;
        if (IsWordCp(runes[i].cp)) {
            ++i;
            while (i < runes.size()) {
                if (IsWordCp(runes[i].cp)) {
                    ++i;
                    continue;
                }
                if ((runes[i].cp == '-' || runes[i].cp == '_') && i + 1 < runes.size() && IsWordCp(runes[i + 1].cp)) {
                    ++i;
                    continue;
                }
                break;
            }
        } else {
            ++i;
        }

        const size_t end = i;
        if (end <= start) {
            continue;
        }

        const size_t byte_start = runes[start].byte_start;
        const size_t byte_end = runes[end - 1].byte_end;
        out.push_back(WordSpan{
            text.substr(byte_start, byte_end - byte_start),
            static_cast<int64_t>(start),
            static_cast<int64_t>(end),
            byte_start,
            byte_end,
        });
    }

    return out;
}

float Sigmoid(float x) {
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

    for (const auto& cand : entities) {
        bool overlap = false;
        for (const auto& k : kept) {
            if (HasOverlap(cand, k)) {
                overlap = true;
                break;
            }
        }
        if (!overlap) {
            kept.push_back(cand);
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

std::optional<std::string> ExtractTextFieldFromJsonLine(const std::string& line) {
    const size_t key_pos = line.find("\"text\"");
    if (key_pos == std::string::npos) {
        return std::nullopt;
    }
    const size_t colon = line.find(':', key_pos + 6);
    if (colon == std::string::npos) {
        return std::nullopt;
    }

    size_t pos = SkipWs(line, colon + 1);
    if (pos >= line.size() || line[pos] != '"') {
        return std::nullopt;
    }
    return ParseJsonString(line, &pos);
}

std::vector<InputRecord> ReadJsonl(const fs::path& input_path) {
    std::ifstream ifs(input_path);
    if (!ifs) {
        throw std::runtime_error("Failed to open input JSONL: " + input_path.string());
    }

    std::vector<InputRecord> records;
    std::string line;
    int line_no = 0;
    while (std::getline(ifs, line)) {
        ++line_no;
        if (line.empty()) {
            continue;
        }
        auto text = ExtractTextFieldFromJsonLine(line);
        if (!text.has_value()) {
            continue;
        }
        records.push_back(InputRecord{line_no, *text});
    }
    return records;
}

RuntimeConfig LoadRuntimeConfig(const fs::path& config_path) {
    RuntimeConfig cfg;
    const std::string config_text = ReadTextFile(config_path);

    if (const auto v = ExtractIntByKey(config_text, "max_len"); v.has_value()) {
        cfg.max_len = static_cast<int32_t>(*v);
    }
    if (const auto v = ExtractIntByKey(config_text, "max_width"); v.has_value()) {
        cfg.max_width = static_cast<int32_t>(*v);
    }
    if (const auto v = ExtractIntByKey(config_text, "class_token_index"); v.has_value()) {
        cfg.class_token_index = static_cast<int32_t>(*v);
    }
    if (const auto v = ExtractStringByKey(config_text, "ent_token"); v.has_value()) {
        cfg.ent_token = *v;
    }
    if (const auto v = ExtractStringByKey(config_text, "sep_token"); v.has_value()) {
        cfg.sep_token = *v;
    }

    return cfg;
}

void ValidateRuntimeConfig(const RuntimeConfig& cfg) {
    if (cfg.max_len <= 0) {
        throw std::runtime_error("Invalid max_len in gliner_config.json (must be > 0)");
    }
    if (cfg.max_width <= 0) {
        throw std::runtime_error("Invalid max_width in gliner_config.json (must be > 0)");
    }
    if (cfg.class_token_index < 0) {
        throw std::runtime_error("Invalid class_token_index in gliner_config.json (must be >= 0)");
    }
}

PreparedInput BuildInputForText(
    const std::string& text,
    const RuntimeConfig& cfg,
    const UnigramTokenizer& tokenizer
) {
    PreparedInput input;
    input.text = text;
    input.words = SplitWordsWithOffsets(text);

    std::vector<std::string> prompt_words;
    prompt_words.reserve(kLabels.size() * 2 + 1);
    for (const auto& label : kLabels) {
        prompt_words.push_back(cfg.ent_token);
        prompt_words.push_back(label);
    }
    prompt_words.push_back(cfg.sep_token);

    std::vector<std::string> model_words = prompt_words;
    model_words.reserve(prompt_words.size() + input.words.size());
    for (const auto& w : input.words) {
        model_words.push_back(w.token);
    }

    std::vector<int64_t> body_ids;
    std::vector<int64_t> body_words_mask;
    body_ids.reserve(512);
    body_words_mask.reserve(512);

    for (size_t wid = 0; wid < model_words.size(); ++wid) {
        const bool is_prompt_word = wid < prompt_words.size();
        const int64_t text_word_1based = is_prompt_word ? 0 : static_cast<int64_t>(wid - prompt_words.size() + 1);

        const std::vector<int32_t> token_ids = tokenizer.EncodeWord(model_words[wid]);
        bool first_piece = true;
        for (const int32_t tid : token_ids) {
            body_ids.push_back(static_cast<int64_t>(tid));
            if (!is_prompt_word && first_piece) {
                body_words_mask.push_back(text_word_1based);
            } else {
                body_words_mask.push_back(0);
            }
            first_piece = false;
        }
    }

    input.input_ids.clear();
    input.words_mask.clear();
    input.input_ids.reserve(body_ids.size() + 2);
    input.words_mask.reserve(body_words_mask.size() + 2);

    input.input_ids.push_back(cfg.cls_id);
    input.words_mask.push_back(0);
    input.input_ids.insert(input.input_ids.end(), body_ids.begin(), body_ids.end());
    input.words_mask.insert(input.words_mask.end(), body_words_mask.begin(), body_words_mask.end());
    input.input_ids.push_back(cfg.sep_id);
    input.words_mask.push_back(0);

    if (cfg.max_len > 0 && input.input_ids.size() > static_cast<size_t>(cfg.max_len)) {
        input.input_ids.resize(static_cast<size_t>(cfg.max_len));
        input.words_mask.resize(static_cast<size_t>(cfg.max_len));
        input.input_ids.back() = cfg.sep_id;
        input.words_mask.back() = 0;
    }

    input.seq_len = static_cast<int64_t>(input.input_ids.size());

    input.attention_mask.assign(input.seq_len, 1);

    input.num_words = 0;
    for (const int64_t w : input.words_mask) {
        input.num_words = std::max(input.num_words, w);
    }
    if (input.num_words > static_cast<int64_t>(input.words.size())) {
        input.num_words = static_cast<int64_t>(input.words.size());
    }
    input.text_lengths = {input.num_words};

    input.num_spans = input.num_words * static_cast<int64_t>(cfg.max_width);
    input.span_idx.clear();
    input.span_mask_u8.clear();
    input.span_idx.reserve(static_cast<size_t>(input.num_spans * 2));
    input.span_mask_u8.reserve(static_cast<size_t>(input.num_spans));

    for (int64_t s = 0; s < input.num_words; ++s) {
        for (int64_t k = 0; k < cfg.max_width; ++k) {
            const int64_t e = s + k;
            input.span_idx.push_back(s);
            input.span_idx.push_back(e);
            input.span_mask_u8.push_back(static_cast<uint8_t>(e < input.num_words ? 1 : 0));
        }
    }

    return input;
}

std::vector<Entity> DecodeEntities(
    const float* logits,
    int64_t out_l,
    int64_t out_k,
    int64_t out_c,
    const PreparedInput& input,
    const RuntimeConfig& cfg,
    float threshold
) {
    const int64_t valid_l = std::min(out_l, input.num_words);
    const int64_t valid_k = std::min<int64_t>(out_k, cfg.max_width);
    const int64_t valid_c = std::min<int64_t>(out_c, static_cast<int64_t>(kLabels.size()));

    std::vector<Entity> candidates;
    candidates.reserve(static_cast<size_t>(valid_l * valid_k));

    for (int64_t s = 0; s < valid_l; ++s) {
        for (int64_t k = 0; k < valid_k; ++k) {
            const int64_t e = s + k;
            if (e >= input.num_words) {
                continue;
            }
            const int64_t flat_span = s * cfg.max_width + k;
            if (flat_span < 0 || flat_span >= static_cast<int64_t>(input.span_mask_u8.size())) {
                continue;
            }
            if (input.span_mask_u8[static_cast<size_t>(flat_span)] == 0U) {
                continue;
            }

            for (int64_t c = 0; c < valid_c; ++c) {
                const size_t idx = static_cast<size_t>(((s * out_k) + k) * out_c + c);
                const float prob = Sigmoid(logits[idx]);
                if (prob < threshold) {
                    continue;
                }

                const auto& start_word = input.words[static_cast<size_t>(s)];
                const auto& end_word = input.words[static_cast<size_t>(e)];

                candidates.push_back(Entity{
                    s,
                    e,
                    start_word.char_start,
                    end_word.char_end,
                    start_word.byte_start,
                    end_word.byte_end,
                    kLabels[static_cast<size_t>(c)],
                    prob,
                });
            }
        }
    }

    return GreedyNonOverlapping(std::move(candidates));
}

void WriteResultJsonLine(
    std::ofstream* ofs,
    int sample_index,
    int line_no,
    const PreparedInput& input,
    const std::vector<Entity>& entities
) {
    *ofs << "{";
    *ofs << "\"sample_index\":" << sample_index << ",";
    *ofs << "\"line\":" << line_no << ",";
    *ofs << "\"text\":\"" << JsonEscape(input.text) << "\",";
    *ofs << "\"entities\":[";

    for (size_t i = 0; i < entities.size(); ++i) {
        const auto& e = entities[i];
        const std::string text = input.text.substr(e.start_byte, e.end_byte - e.start_byte);

        *ofs << "{";
        *ofs << "\"start\":" << e.start_char << ",";
        *ofs << "\"end\":" << e.end_char << ",";
        *ofs << "\"text\":\"" << JsonEscape(text) << "\",";
        *ofs << "\"label\":\"" << JsonEscape(e.label) << "\",";
        *ofs << "\"score\":" << std::fixed << std::setprecision(6) << e.score;
        *ofs << "}";

        if (i + 1 < entities.size()) {
            *ofs << ",";
        }
    }

    *ofs << "]}" << '\n';
}

AppArgs ParseArgs(int argc, char* argv[]) {
    AppArgs args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];

        auto require_value = [&](const char* opt_name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string(opt_name) + " requires a value");
            }
            ++i;
            return argv[i];
        };

        if (key == "--model") {
            args.model_path = require_value("--model");
        } else if (key == "--tokenizer") {
            args.tokenizer_path = require_value("--tokenizer");
        } else if (key == "--config") {
            args.config_path = require_value("--config");
        } else if (key == "--input") {
            args.input_jsonl = require_value("--input");
        } else if (key == "--output") {
            args.output_jsonl = require_value("--output");
        } else if (key == "--threshold") {
            args.threshold = std::stof(require_value("--threshold"));
        } else if (key == "--help" || key == "-h") {
            std::cout << "Usage: gliner_onnx_inference [options]\n"
                      << "  --model <path>      ONNX model path\n"
                      << "  --tokenizer <path>  tokenizer.json path\n"
                      << "  --config <path>     gliner_config.json path\n"
                      << "  --input <path>      input JSONL path\n"
                      << "  --output <path>     output JSONL path\n"
                      << "  --threshold <float> score threshold (default: 0.5)\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown option: " + key);
        }
    }
    return args;
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const AppArgs args = ParseArgs(argc, argv);

        if (args.threshold < 0.0F || args.threshold > 1.0F) {
            throw std::runtime_error("--threshold must be in [0.0, 1.0]");
        }

        if (!fs::exists(args.model_path)) {
            throw std::runtime_error("ONNX model not found: " + args.model_path.string());
        }
        if (!fs::exists(args.tokenizer_path)) {
            throw std::runtime_error("tokenizer.json not found: " + args.tokenizer_path.string());
        }
        if (!fs::exists(args.config_path)) {
            throw std::runtime_error("gliner_config.json not found: " + args.config_path.string());
        }
        if (!fs::exists(args.input_jsonl)) {
            throw std::runtime_error("input JSONL not found: " + args.input_jsonl.string());
        }

        RuntimeConfig cfg = LoadRuntimeConfig(args.config_path);
        ValidateRuntimeConfig(cfg);

        UnigramTokenizer tokenizer;
        tokenizer.LoadFromTokenizerJson(args.tokenizer_path);

        cfg.unk_id = tokenizer.GetUnkId();
        cfg.pad_id = tokenizer.GetSpecialTokenId("[PAD]", cfg.pad_id);
        cfg.cls_id = tokenizer.GetSpecialTokenId("[CLS]", cfg.cls_id);
        cfg.sep_id = tokenizer.GetSpecialTokenId("[SEP]", cfg.sep_id);

        if (cfg.pad_id < 0 || cfg.cls_id < 0 || cfg.sep_id < 0 || cfg.unk_id < 0) {
            throw std::runtime_error("Tokenizer special token ids must be non-negative");
        }

        const int32_t ent_id = tokenizer.GetSpecialTokenId(cfg.ent_token, -1);
        if (ent_id >= 0 && ent_id != cfg.class_token_index) {
            std::cerr << "Warning: class_token_index(" << cfg.class_token_index
                      << ") != tokenizer id of ent_token(" << ent_id << ")\n";
        }

        std::vector<InputRecord> records = ReadJsonl(args.input_jsonl);
        if (records.empty()) {
            throw std::runtime_error("No valid text rows in input JSONL");
        }

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "gliner_cpp_onnx_prod");
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        Ort::Session session(env, args.model_path.c_str(), session_options);
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        const std::array<const char*, 6> input_names = {
            "input_ids",
            "attention_mask",
            "words_mask",
            "text_lengths",
            "span_idx",
            "span_mask",
        };
        const char* output_names[] = {"logits"};

        const fs::path out_dir = args.output_jsonl.parent_path();
        if (!out_dir.empty()) {
            fs::create_directories(out_dir);
        }
        std::ofstream ofs(args.output_jsonl, std::ios::binary);
        if (!ofs) {
            throw std::runtime_error("Cannot open output JSONL for write: " + args.output_jsonl.string());
        }

        int sample_index = 0;
        for (const auto& rec : records) {
            ++sample_index;

            PreparedInput input = BuildInputForText(rec.text, cfg, tokenizer);
            std::vector<Entity> entities;

            if (input.num_words > 0) {
                const std::vector<int64_t> shape_seq{1, input.seq_len};
                const std::vector<int64_t> shape_text_lengths{1, 1};
                const std::vector<int64_t> shape_span_idx{1, input.num_spans, 2};
                const std::vector<int64_t> shape_span_mask{1, input.num_spans};

                std::unique_ptr<bool[]> span_mask_bool(new bool[static_cast<size_t>(input.num_spans)]);
                for (size_t i = 0; i < static_cast<size_t>(input.num_spans); ++i) {
                    span_mask_bool[i] = input.span_mask_u8[i] != 0U;
                }

                std::array<Ort::Value, 6> input_tensors = {
                    Ort::Value::CreateTensor<int64_t>(
                        mem_info,
                        input.input_ids.data(),
                        input.input_ids.size(),
                        shape_seq.data(),
                        shape_seq.size()),
                    Ort::Value::CreateTensor<int64_t>(
                        mem_info,
                        input.attention_mask.data(),
                        input.attention_mask.size(),
                        shape_seq.data(),
                        shape_seq.size()),
                    Ort::Value::CreateTensor<int64_t>(
                        mem_info,
                        input.words_mask.data(),
                        input.words_mask.size(),
                        shape_seq.data(),
                        shape_seq.size()),
                    Ort::Value::CreateTensor<int64_t>(
                        mem_info,
                        input.text_lengths.data(),
                        input.text_lengths.size(),
                        shape_text_lengths.data(),
                        shape_text_lengths.size()),
                    Ort::Value::CreateTensor<int64_t>(
                        mem_info,
                        input.span_idx.data(),
                        input.span_idx.size(),
                        shape_span_idx.data(),
                        shape_span_idx.size()),
                    Ort::Value::CreateTensor<bool>(
                        mem_info,
                        span_mask_bool.get(),
                        static_cast<size_t>(input.num_spans),
                        shape_span_mask.data(),
                        shape_span_mask.size()),
                };

                auto outputs = session.Run(
                    Ort::RunOptions{nullptr},
                    input_names.data(),
                    input_tensors.data(),
                    input_tensors.size(),
                    output_names,
                    1
                );

                if (outputs.empty()) {
                    throw std::runtime_error("ONNX session returned no outputs");
                }

                const Ort::Value& logits = outputs.front();
                const auto info = logits.GetTensorTypeAndShapeInfo();
                const std::vector<int64_t> out_shape = info.GetShape();
                if (out_shape.size() != 4) {
                    throw std::runtime_error("Unexpected logits rank: " + std::to_string(out_shape.size()));
                }
                if (out_shape[0] != 1) {
                    std::cerr << "Warning: batch size is " << out_shape[0]
                              << " (decoder expects batch=1 per run)\n";
                }
                if (out_shape[1] <= 0 || out_shape[2] <= 0 || out_shape[3] <= 0) {
                    throw std::runtime_error("Invalid logits shape (non-positive dimension)");
                }

                const float* out_data = logits.GetTensorData<float>();
                entities = DecodeEntities(out_data, out_shape[1], out_shape[2], out_shape[3], input, cfg, args.threshold);
            }

            WriteResultJsonLine(&ofs, sample_index, rec.line, input, entities);
        }

        std::cout << "Processed samples: " << sample_index << "\n";
        std::cout << "Saved output to: " << args.output_jsonl << "\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
