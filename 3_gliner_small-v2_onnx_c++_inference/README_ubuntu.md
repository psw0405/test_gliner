# Ubuntu ONNX C++ Inference Guide

This folder now supports Ubuntu build/run for GLiNER ONNX C++ inference.

Supported baseline:
- Ubuntu `20.04` / `22.04` / `24.04` (x86_64)
- ONNX Runtime Linux x64 SDK (tested with `1.24.4`)

## 1) Quick setup (recommended)

```bash
cd 3_gliner_small-v2_onnx_c++_inference
chmod +x ./setup_ubuntu_env.sh ./build_and_run_ubuntu.sh
./setup_ubuntu_env.sh --download-ort
./build_and_run_ubuntu.sh
```

`setup_ubuntu_env.sh` does:
- Installs apt dependencies
- Optionally downloads ONNX Runtime SDK to `third_party/`
- Prints `ONNXRUNTIME_DIR` value to export if needed

## 2) Manual dependency install

```bash
sudo apt update
sudo apt install -y build-essential cmake wget tar ca-certificates pkg-config git
```

## 3) Download ONNX Runtime Linux SDK

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

## 4) Build and run

```bash
chmod +x ./build_and_run_ubuntu.sh
./build_and_run_ubuntu.sh
```

The script does:
- CMake configure/build
- Runs inference on `../sample.jsonl`
- Writes `onnx_cpp_output.jsonl`

## 5) Optional arguments

```bash
./build_and_run_ubuntu.sh \
  --onnxruntime-dir ./third_party/onnxruntime-linux-x64-1.24.4 \
  --threshold 0.5 \
  --output onnx_cpp_output.jsonl
```

`setup_ubuntu_env.sh` optional arguments:

```bash
./setup_ubuntu_env.sh --download-ort --ort-version 1.24.4
./setup_ubuntu_env.sh --skip-apt
```

## 6) Manual run (without helper script)

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

## 7) Server notes

- If `build_and_run_ubuntu.sh` cannot find ONNX Runtime SDK, set it explicitly:

```bash
export ONNXRUNTIME_DIR="$PWD/third_party/onnxruntime-linux-x64-1.24.4"
./build_and_run_ubuntu.sh
```

- If you run on a restricted server without sudo, run setup with `--skip-apt` and install dependencies through your base image or admin flow.
