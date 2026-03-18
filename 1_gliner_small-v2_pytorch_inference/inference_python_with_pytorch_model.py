import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    from gliner import GLiNER
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: gliner. Install with `pip install gliner torch`."
    ) from exc


MODEL_NAME = "urchade/gliner_small-v2"
DEFAULT_INPUT_PATH = Path(__file__).resolve().parents[1] / "sample.jsonl"

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
        description="Run GLiNER small-v2 inference with predefined 26 labels on JSONL input."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to JSONL file. Default: ../sample.jsonl",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="Hugging Face model id. Default: urchade/gliner_small-v2",
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
        help="Number of lines to process (0 means all lines).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSONL path to save inference results.",
    )
    return parser.parse_args()


def read_jsonl(input_path: Path) -> List[Dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc

            if not isinstance(record, dict):
                continue

            text_value = record.get("text")
            if not isinstance(text_value, str) or not text_value.strip():
                continue

            records.append({"line": line_no, "text": text_value})

    return records


def predict_entities(model: Any, text: str, threshold: float) -> List[Dict[str, Any]]:
    # Keep compatibility across GLiNER versions that may differ in method signature.
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


def main() -> None:
    args = parse_args()

    print(f"Loading model: {args.model}")
    model = GLiNER.from_pretrained(args.model)

    records = read_jsonl(args.input)
    if args.max_samples > 0:
        records = records[: args.max_samples]

    print(f"Loaded {len(records)} samples from: {args.input}")

    results: List[Dict[str, Any]] = []
    for idx, rec in enumerate(records, start=1):
        entities = predict_entities(model, rec["text"], args.threshold)
        result = {
            "sample_index": idx,
            "line": rec["line"],
            "text": rec["text"],
            "labels": LABELS,
            "entities": entities,
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved output to: {args.output}")


if __name__ == "__main__":
    main()
