#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY_DIR="$SCRIPT_DIR/third_party"
ORT_VERSION="1.24.4"
DOWNLOAD_ORT=0
SKIP_APT=0

usage() {
  cat <<'EOF'
Usage: ./setup_ubuntu_env.sh [options]

Options:
  --ort-version <ver>       ONNX Runtime version to download (default: 1.24.4)
  --third-party-dir <path>  Directory to store ONNX Runtime SDK (default: ./third_party)
  --download-ort            Download and extract ONNX Runtime Linux x64 SDK
  --skip-apt                Skip apt dependency installation
  -h, --help                Show this help

Examples:
  ./setup_ubuntu_env.sh
  ./setup_ubuntu_env.sh --download-ort
  ./setup_ubuntu_env.sh --download-ort --ort-version 1.24.4
EOF
}

fail() {
  echo "Error: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ort-version)
      ORT_VERSION="$2"
      shift 2
      ;;
    --third-party-dir)
      THIRD_PARTY_DIR="$2"
      shift 2
      ;;
    --download-ort)
      DOWNLOAD_ORT=1
      shift
      ;;
    --skip-apt)
      SKIP_APT=1
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

if [[ ! -f /etc/os-release ]]; then
  fail "Cannot detect Linux distribution (/etc/os-release missing)."
fi

# shellcheck source=/etc/os-release
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" ]]; then
  fail "This setup script currently supports Ubuntu only (detected: ${ID:-unknown})."
fi

UBUNTU_VER="${VERSION_ID:-unknown}"
echo "Detected Ubuntu ${UBUNTU_VER}"

if [[ "$SKIP_APT" -eq 0 ]]; then
  if ! command -v sudo >/dev/null 2>&1; then
    fail "sudo is required for apt installation. Use --skip-apt to skip this step."
  fi

  sudo apt update
  sudo apt install -y \
    build-essential \
    cmake \
    wget \
    tar \
    ca-certificates \
    pkg-config \
    git
fi

if command -v g++ >/dev/null 2>&1; then
  GXX_VER="$(g++ -dumpfullversion -dumpversion 2>/dev/null || true)"
  echo "Detected g++ version: ${GXX_VER:-unknown}"
fi

if [[ "$DOWNLOAD_ORT" -eq 1 ]]; then
  mkdir -p "$THIRD_PARTY_DIR"
  ARCHIVE="onnxruntime-linux-x64-${ORT_VERSION}.tgz"
  URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ARCHIVE}"

  echo "Downloading ONNX Runtime ${ORT_VERSION} ..."
  wget -O "$THIRD_PARTY_DIR/$ARCHIVE" "$URL"

  echo "Extracting $ARCHIVE ..."
  tar -xzf "$THIRD_PARTY_DIR/$ARCHIVE" -C "$THIRD_PARTY_DIR"
fi

if [[ -d "$THIRD_PARTY_DIR" ]]; then
  mapfile -t ort_dirs < <(find "$THIRD_PARTY_DIR" -maxdepth 1 -mindepth 1 -type d -name "onnxruntime-linux-x64-*" | sort -r)
  if [[ ${#ort_dirs[@]} -gt 0 ]]; then
    echo ""
    echo "Found ONNX Runtime SDK: ${ort_dirs[0]}"
    echo "Export this in your shell if needed:"
    echo "export ONNXRUNTIME_DIR='${ort_dirs[0]}'"
  fi
fi

echo ""
echo "Setup complete."
echo "Next step:"
echo "  ./build_and_run_ubuntu.sh"
