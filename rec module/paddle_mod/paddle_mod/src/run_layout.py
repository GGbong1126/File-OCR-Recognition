from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from layout.structure_analyzer import LayoutAnalyzer, LayoutAnalyzerConfig

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_config(config_path: Path) -> LayoutAnalyzerConfig:
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    return LayoutAnalyzerConfig(
        use_gpu=bool(raw.get("use_gpu", False)),
        lang=str(raw.get("lang", "ch")),
        table=bool(raw.get("table", True)),
        ocr=bool(raw.get("ocr", True)),
        show_log=bool(raw.get("show_log", False)),
    )


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in SUPPORTED_EXTENSIONS else []
    if input_path.is_dir():
        images = [
            p
            for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        return sorted(images)
    raise FileNotFoundError(f"Input path not found: {input_path}")


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        converted: dict[str, Any] = {}
        for k, v in value.items():
            if k == "img":
                continue
            converted[str(k)] = to_jsonable(v)
        return converted
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def save_blocks(blocks: Sequence[dict[str, Any]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    normalized = [to_jsonable(block) for block in blocks]
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PP-Structure layout analysis for table and paragraph blocks."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw"),
        help="Input image path or folder.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/layout_config.json"),
        help="Layout config file path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Output root folder.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    analyzer = LayoutAnalyzer(config)
    image_paths = collect_images(args.input)

    if not image_paths:
        print(f"No supported images found in: {args.input}")
        return

    table_root = args.output / "table_blocks"
    paragraph_root = args.output / "paragraph_blocks"

    for image_path in image_paths:
        layout_results = analyzer.analyze_image(image_path)
        table_blocks, paragraph_blocks = analyzer.split_blocks(layout_results)

        table_output = table_root / f"{image_path.stem}.json"
        paragraph_output = paragraph_root / f"{image_path.stem}.json"

        save_blocks(table_blocks, table_output)
        save_blocks(paragraph_blocks, paragraph_output)

        print(
            f"Processed {image_path.name}: "
            f"tables={len(table_blocks)}, paragraphs={len(paragraph_blocks)}"
        )


if __name__ == "__main__":
    main()
