import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    from gliner import GLiNER
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: gliner. Install with `pip install gliner torch onnx onnxruntime`."
    ) from exc


MODEL_ID = "urchade/gliner_small-v2"
DEFAULT_INPUT_PATH = Path(__file__).resolve().parents[1] / "sample.jsonl"
DEFAULT_ONNX_DIR = Path(__file__).resolve().parent / "onnx_model"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "onnx_test_output.jsonl"

LABELS = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export GLiNER PyTorch model to ONNX (opset 13) and run inference on JSONL.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=MODEL_ID,
        help="Hugging Face model id. Default: urchade/gliner_small-v2",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input JSONL path with {'text': ...} lines. Default: ../sample.jsonl",
    )
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=DEFAULT_ONNX_DIR,
        help="Directory to save ONNX model files.",
    )
    parser.add_argument(
        "--onnx-file",
        type=str,
        default="model_opset13.onnx",
        help="ONNX model filename.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset version. Default: 13",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Entity confidence threshold. Default: 0.5",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Number of JSONL lines to process (0 means all).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSONL path for inference results.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip export step and use existing ONNX model from --onnx-dir/--onnx-file.",
    )
    return parser.parse_args()


def read_jsonl(input_path: Path) -> List[Dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rows: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue

            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc

            if not isinstance(obj, dict):
                continue

            text = obj.get("text")
            if isinstance(text, str) and text.strip():
                rows.append({"line": line_no, "text": text})

    return rows


def export_gliner_to_onnx(model_id: str, onnx_dir: Path, onnx_file: str, opset: int) -> Path:
    onnx_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading PyTorch model: {model_id}")
    model = GLiNER.from_pretrained(model_id, map_location="cpu")

    print(f"Exporting ONNX to: {onnx_dir / onnx_file} (opset={opset})")
    export_info = model.export_to_onnx(
        save_dir=onnx_dir,
        onnx_filename=onnx_file,
        quantize=False,
        opset=opset,
        labels=LABELS,
        text="GLiNER ONNX export warm-up sentence.",
    )
    print(json.dumps(export_info, ensure_ascii=False))

    onnx_path = onnx_dir / onnx_file
    if not onnx_path.exists():
        raise FileNotFoundError(f"Export finished but ONNX file not found: {onnx_path}")

    return onnx_path


def load_onnx_model(onnx_dir: Path, onnx_file: str) -> Any:
    onnx_path = onnx_dir / onnx_file
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model file not found: {onnx_path}")

    print(f"Loading ONNX model from: {onnx_path}")
    return GLiNER.from_pretrained(
        str(onnx_dir),
        load_onnx_model=True,
        onnx_model_file=onnx_file,
        map_location="cpu",
    )


def predict_entities(model: Any, text: str, threshold: float) -> List[Dict[str, Any]]:
    # Keep compatibility across GLiNER versions with different method signatures.
    try:
        entities = model.predict_entities(text, LABELS, threshold=threshold)
    except TypeError:
        entities = model.predict_entities(text, LABELS)

    normalized: List[Dict[str, Any]] = []
    for ent in entities:
        score = ent.get("score")
        if isinstance(score, (int, float)) and score < threshold:
            continue

        normalized.append(
            {
                "text": ent.get("text"),
                "label": ent.get("label"),
                "start": ent.get("start"),
                "end": ent.get("end"),
                "score": ent.get("score"),
            }
        )

    return normalized


def run_inference(model: Any, rows: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for idx, rec in enumerate(rows, start=1):
        entities = predict_entities(model, rec["text"], threshold)
        result = {
            "sample_index": idx,
            "line": rec["line"],
            "text": rec["text"],
            "labels": LABELS,
            "entities": entities,
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    return results


def write_jsonl(output_path: Path, rows: List[Dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    onnx_dir = args.onnx_dir.resolve()
    output_path = args.output.resolve()

    if not args.skip_export:
        export_gliner_to_onnx(args.model_id, onnx_dir, args.onnx_file, args.opset)
    else:
        onnx_path = onnx_dir / args.onnx_file
        if not onnx_path.exists():
            raise FileNotFoundError(f"--skip-export was set but ONNX file does not exist: {onnx_path}")

    model = load_onnx_model(onnx_dir, args.onnx_file)

    rows = read_jsonl(args.input)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    print(f"Loaded {len(rows)} samples from: {args.input}")
    results = run_inference(model, rows, args.threshold)
    write_jsonl(output_path, results)
    print(f"Saved inference results to: {output_path}")


if __name__ == "__main__":
    main()
