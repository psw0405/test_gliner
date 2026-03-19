param(
    [string]$OnnxRuntimeDir = "",
    [string]$BuildDir = "build",
    [string]$ModelPath = "..\2_gliner_small-v2_onnx_inference\onnx_model\model_opset13.onnx",
    [string]$TokenizerPath = "..\2_gliner_small-v2_onnx_inference\onnx_model\tokenizer.json",
    [string]$ConfigPath = "..\2_gliner_small-v2_onnx_inference\onnx_model\gliner_config.json",
    [string]$InputPath = "..\sample.jsonl",
    [string]$OutputPath = "onnx_cpp_output.jsonl",
    [double]$Threshold = 0.5,
    [switch]$SkipRun
)

$ErrorActionPreference = "Stop"

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [string[]]$Arguments = @(),
        [string]$StepName = "command"
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE"
    }
}

function Get-CMakeCommand {
    $cmd = Get-Command cmake -ErrorAction SilentlyContinue
    if ($null -ne $cmd) {
        return $cmd.Source
    }

    $default = "C:\Program Files\CMake\bin\cmake.exe"
    if (Test-Path $default) {
        return $default
    }

    throw "CMake not found. Install CMake and ensure it is available in PATH or at '$default'."
}

function Resolve-OnnxRuntimeDir {
    param(
        [string]$InputDir
    )

    if (-not [string]::IsNullOrWhiteSpace($InputDir)) {
        return $InputDir
    }

    if ($env:ONNXRUNTIME_DIR) {
        return $env:ONNXRUNTIME_DIR
    }

    $candidates = Get-ChildItem -Path ".\third_party" -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "onnxruntime-win-x64-*" } |
        Sort-Object Name -Descending

    if ($candidates.Count -gt 0) {
        return $candidates[0].FullName
    }

    throw "ONNX Runtime SDK directory not found. Pass -OnnxRuntimeDir or set ONNXRUNTIME_DIR or place SDK under .\\third_party\\onnxruntime-win-x64-*"
}

function Ensure-MsvcEnvironment {
    $cl = Get-Command cl -ErrorAction SilentlyContinue
    if ($null -ne $cl) {
        return
    }

    $vsDevShell = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1"
    if (-not (Test-Path $vsDevShell)) {
        throw "MSVC toolchain not found. Install Visual Studio Build Tools 2022 with C++ workload."
    }

    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
    & $vsDevShell -Arch amd64 -HostArch amd64 | Out-Null

    $cl = Get-Command cl -ErrorAction SilentlyContinue
    if ($null -eq $cl) {
        throw "Failed to initialize MSVC environment (cl.exe still not found)."
    }
}

if ($Threshold -lt 0.0 -or $Threshold -gt 1.0) {
    throw "Threshold must be in range [0.0, 1.0]."
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $scriptRoot
try {
    $cmake = Get-CMakeCommand
    $resolvedOrtDir = Resolve-OnnxRuntimeDir -InputDir $OnnxRuntimeDir

    $header = Join-Path $resolvedOrtDir "include\onnxruntime_cxx_api.h"
    $lib = Join-Path $resolvedOrtDir "lib\onnxruntime.lib"
    $dll = Join-Path $resolvedOrtDir "lib\onnxruntime.dll"
    if (-not (Test-Path $header)) { throw "Missing header: $header" }
    if (-not (Test-Path $lib)) { throw "Missing import lib: $lib" }
    if (-not (Test-Path $dll)) { throw "Missing runtime dll: $dll" }

    if (-not (Test-Path $ModelPath)) { throw "Model not found: $ModelPath" }
    if (-not (Test-Path $TokenizerPath)) { throw "Tokenizer not found: $TokenizerPath" }
    if (-not (Test-Path $ConfigPath)) { throw "Config not found: $ConfigPath" }
    if (-not (Test-Path $InputPath)) { throw "Input JSONL not found: $InputPath" }

    Ensure-MsvcEnvironment

    Invoke-NativeChecked -FilePath $cmake -Arguments @(
        "-S", ".",
        "-B", $BuildDir,
        "-G", "NMake Makefiles",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DONNXRUNTIME_DIR=$resolvedOrtDir"
    ) -StepName "cmake configure"

    Invoke-NativeChecked -FilePath $cmake -Arguments @(
        "--build", $BuildDir,
        "--config", "Release"
    ) -StepName "cmake build"

    if (-not $SkipRun) {
        $exe = Join-Path $BuildDir "gliner_onnx_inference.exe"
        if (-not (Test-Path $exe)) {
            throw "Executable not found after build: $exe"
        }
        $exe = (Resolve-Path $exe).Path

        Invoke-NativeChecked -FilePath $exe -Arguments @(
            "--model", $ModelPath,
            "--tokenizer", $TokenizerPath,
            "--config", $ConfigPath,
            "--input", $InputPath,
            "--output", $OutputPath,
            "--threshold", "$Threshold"
        ) -StepName "inference run"

        if (Test-Path $OutputPath) {
            $lineCount = (Get-Content $OutputPath | Measure-Object -Line).Lines
            Write-Host "Output file: $OutputPath"
            Write-Host "Output lines: $lineCount"
        }
    }

    Write-Host "Done"
} finally {
    Pop-Location
}
