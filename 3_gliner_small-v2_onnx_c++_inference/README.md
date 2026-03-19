# GLiNER ONNX C++ Inference

`urchade/gliner_small-v2`를 ONNX Runtime C++로 실행해 NER(entity 추출)하는 예제입니다.

## 지원 환경
- Windows (MSVC + CMake)
- Ubuntu 20.04 / 22.04 / 24.04 (g++ + CMake)

## 입력/출력 형식
입력은 JSONL이며, 한 줄당 아래 형식을 사용합니다.

```json
{"text":"삼성전자는 서울에서 신제품을 공개했다."}
```

출력도 JSONL이며, 각 줄에 원문과 추출 엔티티가 저장됩니다.

```json
{"sample_index":1,"line":1,"text":"...","entities":[{"start":0,"end":4,"text":"삼성전자","label":"Organization","score":0.91}]}
```

## 빠른 시작 (Ubuntu)

```bash
cd 3_gliner_small-v2_onnx_c++_inference
chmod +x ./setup_ubuntu_env.sh ./build_and_run_ubuntu.sh
./setup_ubuntu_env.sh --download-ort
./build_and_run_ubuntu.sh
```

## 빠른 시작 (Windows)

```powershell
cd 3_gliner_small-v2_onnx_c++_inference
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\build_and_run.ps1
```

## 아무 문장 테스트하기
임의 문장을 바로 테스트하려면 `one.jsonl`을 만들어 실행하면 됩니다.

### Ubuntu

```bash
cd 3_gliner_small-v2_onnx_c++_inference
printf '{"text":"내일 서울 코엑스에서 AI 컨퍼런스가 열린다."}\n' > one.jsonl
./build_and_run_ubuntu.sh --input one.jsonl --output one_out.jsonl --threshold 0.5
cat one_out.jsonl
```

### Windows (PowerShell)

```powershell
cd 3_gliner_small-v2_onnx_c++_inference
'{"text":"내일 서울 코엑스에서 AI 컨퍼런스가 열린다."}' | Set-Content one.jsonl -Encoding UTF8
.\build_and_run.ps1 -InputPath one.jsonl -OutputPath one_out.jsonl -Threshold 0.5
Get-Content one_out.jsonl
```

## 스크립트 옵션

### `build_and_run_ubuntu.sh`
- `--onnxruntime-dir <path>`: ONNX Runtime SDK 경로
- `--build-dir <path>`: 빌드 디렉토리
- `--input <path>`: 입력 JSONL
- `--output <path>`: 출력 JSONL
- `--threshold <float>`: confidence threshold (0~1)
- `--skip-run`: 빌드만 수행

### `build_and_run.ps1`
- `-OnnxRuntimeDir <path>`
- `-BuildDir <path>`
- `-InputPath <path>`
- `-OutputPath <path>`
- `-Threshold <double>`
- `-SkipRun`

## 참고
- 현재 엔티티 라벨셋은 `gliner_onnx_inference.cpp` 내부 `kLabels`(26개)로 고정되어 있습니다.
- Ubuntu에서 ONNX Runtime `.so` 로딩 문제 시:

```bash
export ONNXRUNTIME_DIR="$PWD/third_party/onnxruntime-linux-x64-1.24.4"
./build_and_run_ubuntu.sh
```

---
추가 Ubuntu 상세 환경설정은 `README_ubuntu.md`를 참고하세요.
