# Ubuntu ONNX C++ Inference Guide

This folder now supports Ubuntu build/run for GLiNER ONNX C++ inference.

## 1) Install build tools

```bash
sudo apt update
sudo apt install -y build-essential cmake wget tar
```

## 2) Download ONNX Runtime Linux SDK

```bash
cd 3_gliner_small-v2_onnx_c++_inference
mkdir -p third_party
cd third_party
wget https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-linux-x64-1.24.4.tgz
tar -xzf onnxruntime-linux-x64-1.24.4.tgz
cd ..
```

Expected SDK path:
- `third_party/onnxruntime-linux-x64-1.24.4`

## 3) Build and run

```bash
chmod +x ./build_and_run_ubuntu.sh
./build_and_run_ubuntu.sh
```

The script does:
- CMake configure/build
- Runs inference on `../sample.jsonl`
- Writes `onnx_cpp_output.jsonl`

## 4) Optional arguments

```bash
./build_and_run_ubuntu.sh \
  --onnxruntime-dir ./third_party/onnxruntime-linux-x64-1.24.4 \
  --threshold 0.5 \
  --output onnx_cpp_output.jsonl
```

## 5) Manual run (without helper script)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_DIR=./third_party/onnxruntime-linux-x64-1.24.4
cmake --build build -j
LD_LIBRARY_PATH=./third_party/onnxruntime-linux-x64-1.24.4/lib:${LD_LIBRARY_PATH} \
  ./build/gliner_onnx_inference \
  --model ../2_gliner_small-v2_onnx_inference/onnx_model/model_opset13.onnx \
  --tokenizer ../2_gliner_small-v2_onnx_inference/onnx_model/tokenizer.json \
  --config ../2_gliner_small-v2_onnx_inference/onnx_model/gliner_config.json \
  --input ../sample.jsonl \
  --output onnx_cpp_output.jsonl \
  --threshold 0.5
```
