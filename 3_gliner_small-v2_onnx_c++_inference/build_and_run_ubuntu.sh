#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

ONNXRUNTIME_DIR="${ONNXRUNTIME_DIR:-}"
BUILD_DIR="build"
MODEL_PATH="../2_gliner_small-v2_onnx_inference/onnx_model/model_opset13.onnx"
TOKENIZER_PATH="../2_gliner_small-v2_onnx_inference/onnx_model/tokenizer.json"
CONFIG_PATH="../2_gliner_small-v2_onnx_inference/onnx_model/gliner_config.json"
INPUT_PATH="../sample.jsonl"
OUTPUT_PATH="onnx_cpp_output.jsonl"
THRESHOLD="0.5"
SKIP_RUN=0
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

usage() {
  cat <<'EOF'
Usage: ./build_and_run_ubuntu.sh [options]

Options:
  --onnxruntime-dir <path>   ONNX Runtime SDK root directory
  --build-dir <path>         CMake build directory (default: build)
  --model <path>             ONNX model path
  --tokenizer <path>         tokenizer.json path
  --config <path>            gliner_config.json path
  --input <path>             input JSONL path
  --output <path>            output JSONL path
  --threshold <float>        score threshold in [0, 1] (default: 0.5)
  --jobs <int>               parallel build jobs (default: nproc)
  --skip-run                 only configure/build, skip inference run
  -h, --help                 show this help
EOF
}

fail() {
  echo "Error: $*" >&2
  exit 1
}

require_file() {
  local file_path="$1"
  [[ -f "$file_path" ]] || fail "File not found: $file_path"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --onnxruntime-dir)
      ONNXRUNTIME_DIR="$2"
      shift 2
      ;;
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    --tokenizer)
      TOKENIZER_PATH="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --input)
      INPUT_PATH="$2"
      shift 2
      ;;
    --output)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    --threshold)
      THRESHOLD="$2"
      shift 2
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    --skip-run)
      SKIP_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "Unknown option: $1"
      ;;
  esac
done

if ! command -v cmake >/dev/null 2>&1; then
  fail "cmake is required. Install it first (e.g., sudo apt install cmake build-essential)."
fi

if ! command -v c++ >/dev/null 2>&1; then
  fail "C++ compiler is required. Install build-essential (g++)."
fi

if ! awk -v t="$THRESHOLD" 'BEGIN { if ((t + 0) != t || t < 0 || t > 1) exit 1; }'; then
  fail "--threshold must be a number in [0, 1]"
fi

cd "$SCRIPT_DIR"

if [[ -z "$ONNXRUNTIME_DIR" ]]; then
  mapfile -t candidates < <(find "./third_party" -maxdepth 1 -mindepth 1 -type d -name "onnxruntime-linux-x64-*" | sort -r)
  if [[ ${#candidates[@]} -gt 0 ]]; then
    ONNXRUNTIME_DIR="${candidates[0]}"
  fi
fi

if [[ -z "$ONNXRUNTIME_DIR" ]]; then
  fail "ONNX Runtime SDK not found. Pass --onnxruntime-dir or set ONNXRUNTIME_DIR or place SDK under ./third_party/onnxruntime-linux-x64-*"
fi

if [[ ! -d "$ONNXRUNTIME_DIR" ]]; then
  fail "Invalid ONNX Runtime directory: $ONNXRUNTIME_DIR"
fi

require_file "$ONNXRUNTIME_DIR/include/onnxruntime_cxx_api.h"
require_file "$ONNXRUNTIME_DIR/lib/libonnxruntime.so"
require_file "$MODEL_PATH"
require_file "$TOKENIZER_PATH"
require_file "$CONFIG_PATH"
require_file "$INPUT_PATH"

cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_DIR="$ONNXRUNTIME_DIR"
cmake --build "$BUILD_DIR" --config Release -j "$JOBS"

if [[ "$SKIP_RUN" -eq 0 ]]; then
  EXE="$BUILD_DIR/gliner_onnx_inference"
  [[ -x "$EXE" ]] || fail "Executable not found: $EXE"

  LD_LIBRARY_PATH="$ONNXRUNTIME_DIR/lib:${LD_LIBRARY_PATH:-}" \
    "$EXE" \
      --model "$MODEL_PATH" \
      --tokenizer "$TOKENIZER_PATH" \
      --config "$CONFIG_PATH" \
      --input "$INPUT_PATH" \
      --output "$OUTPUT_PATH" \
      --threshold "$THRESHOLD"

  if [[ -f "$OUTPUT_PATH" ]]; then
    line_count="$(wc -l < "$OUTPUT_PATH" | tr -d ' ')"
    echo "Output file: $OUTPUT_PATH"
    echo "Output lines: $line_count"
  fi
fi

echo "Done"