from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from layout.paddlepico_detector import PaddlePicoDetector, PaddlePicoDetectorConfig
from layout.table_cell_postprocessor import (
    TableCellPostProcessor,
    TableCellPostProcessorConfig,
)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class RuntimeConfig:
    input_path: Path
    output_root: Path
    batch_size: int
    save_visualization: bool
    save_raw_result: bool
    detector: PaddlePicoDetectorConfig
    table_cell_postprocess: "TableCellRuntimeConfig"


@dataclass
class TableCellRuntimeConfig:
    enabled: bool
    output_dirname: str
    visualization_dirname: str
    detector: TableCellPostProcessorConfig


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
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def to_position_quad(coordinate: Any) -> list[float] | None:
    if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 4:
        return None

    try:
        x_min, y_min, x_max, y_max = [float(v) for v in coordinate]
    except (TypeError, ValueError):
        return None

    # 顺序要求: 左下, 右下, 右上, 左上
    return [x_min, y_max, x_max, y_max, x_max, y_min, x_min, y_min]


def build_standard_boxes(
    boxes: list[dict[str, Any]],
    table_cell_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    table_positions: list[list[list[float]]] = []

    if isinstance(table_cell_payload, dict):
        tables = table_cell_payload.get("tables", [])
        if isinstance(tables, list):
            for table in tables:
                if not isinstance(table, dict):
                    table_positions.append([])
                    continue

                cells = table.get("cells", [])
                if not isinstance(cells, list):
                    table_positions.append([])
                    continue

                ordered_cells = sorted(
                    [cell for cell in cells if isinstance(cell, dict)],
                    key=lambda cell: int(cell.get("cell_index", 0)),
                )

                cell_positions: list[list[float]] = []
                for cell in ordered_cells:
                    quad = to_position_quad(cell.get("coordinate"))
                    if quad is not None:
                        cell_positions.append(quad)

                table_positions.append(cell_positions)

    normalized: list[dict[str, Any]] = []
    table_index = 0

    for box in boxes:
        if not isinstance(box, dict):
            continue

        box_class = str(box.get("label", ""))

        if box_class.lower() == "table":
            # 一个Number只对应一个回归框: 将table展开为多个单元格框。
            if table_index < len(table_positions):
                cell_positions = table_positions[table_index]
            else:
                cell_positions = []
            table_index += 1

            if cell_positions:
                for cell_position in cell_positions:
                    normalized.append(
                        {
                            "Position": cell_position,
                            "Number": str(len(normalized)),
                            "Class": "table",
                        }
                    )
            else:
                quad = to_position_quad(box.get("coordinate"))
                if quad is None:
                    continue
                normalized.append(
                    {
                        "Position": quad,
                        "Number": str(len(normalized)),
                        "Class": "table",
                    }
                )
        else:
            quad = to_position_quad(box.get("coordinate"))
            if quad is None:
                continue
            normalized.append(
                {
                    "Position": quad,
                    "Number": str(len(normalized)),
                    "Class": box_class,
                }
            )

    return normalized


def load_runtime_config(
    config_path: Path,
    input_override: Path | None,
    output_override: Path | None,
) -> RuntimeConfig:
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    pico = raw.get("paddlepico", {})
    if not isinstance(pico, dict):
        raise ValueError("配置文件中的 paddlepico 字段必须是对象。")

    input_path = input_override or Path(str(pico.get("input_path", "data/raw")))
    output_root = output_override or Path(str(pico.get("output_root", "outputs/paddlepico")))
    batch_size = int(pico.get("batch_size", 1))
    save_visualization = bool(pico.get("save_visualization", True))
    save_raw_result = bool(pico.get("save_raw_result", True))

    detector = PaddlePicoDetectorConfig(
        model_name=str(pico.get("model_name", "PP-DocLayoutV3")),
        model_dir=(str(pico["model_dir"]) if pico.get("model_dir") else None),
        device=(str(pico["device"]) if pico.get("device") else None),
        threshold=(float(pico["threshold"]) if pico.get("threshold") is not None else None),
        layout_nms=(bool(pico["layout_nms"]) if pico.get("layout_nms") is not None else None),
        layout_unclip_ratio=(
            float(pico["layout_unclip_ratio"])
            if pico.get("layout_unclip_ratio") is not None
            else None
        ),
        layout_merge_bboxes_mode=(
            str(pico["layout_merge_bboxes_mode"])
            if pico.get("layout_merge_bboxes_mode") is not None
            else None
        ),
    )

    table_cell_raw = pico.get("table_cell_postprocess", {})
    if table_cell_raw is None:
        table_cell_raw = {}
    if not isinstance(table_cell_raw, dict):
        raise ValueError("配置文件中的 paddlepico.table_cell_postprocess 字段必须是对象。")

    table_cell_detector = TableCellPostProcessorConfig(
        enabled=bool(table_cell_raw.get("enabled", False)),
        model_name=str(
            table_cell_raw.get("model_name", "RT-DETR-L_wired_table_cell_det")
        ),
        model_dir=(
            str(table_cell_raw["model_dir"])
            if table_cell_raw.get("model_dir")
            else None
        ),
        device=(
            str(table_cell_raw["device"])
            if table_cell_raw.get("device")
            else detector.device
        ),
        threshold=(
            float(table_cell_raw["threshold"])
            if table_cell_raw.get("threshold") is not None
            else 0.3
        ),
        batch_size=int(table_cell_raw.get("batch_size", 1)),
        save_raw_result=bool(table_cell_raw.get("save_raw_result", False)),
        save_visualization=bool(table_cell_raw.get("save_visualization", True)),
    )

    table_cell_runtime = TableCellRuntimeConfig(
        enabled=table_cell_detector.enabled,
        output_dirname=str(table_cell_raw.get("output_dirname", "table_cell_blocks")),
        visualization_dirname=str(
            table_cell_raw.get("visualization_dirname", "table_cell_visualizations")
        ),
        detector=table_cell_detector,
    )

    return RuntimeConfig(
        input_path=input_path,
        output_root=output_root,
        batch_size=batch_size,
        save_visualization=save_visualization,
        save_raw_result=save_raw_result,
        detector=detector,
        table_cell_postprocess=table_cell_runtime,
    )


def save_json(data: dict[str, Any], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 PaddlePico 文档目标检测独立推理。")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/layout_config.json"),
        help="配置文件路径。",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="输入图片路径或目录，传入后覆盖配置文件。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出目录，传入后覆盖配置文件。",
    )
    args = parser.parse_args()

    runtime = load_runtime_config(args.config, args.input, args.output)
    image_paths = collect_images(runtime.input_path)

    if not image_paths:
        print(f"未找到可处理的图片: {runtime.input_path}")
        return

    detector = PaddlePicoDetector(runtime.detector)
    table_cell_processor: TableCellPostProcessor | None = None
    if runtime.table_cell_postprocess.enabled:
        table_cell_processor = TableCellPostProcessor(runtime.table_cell_postprocess.detector)

    json_dir = runtime.output_root / "json"
    viz_dir = runtime.output_root / "visualizations"
    table_cell_dir = runtime.output_root / runtime.table_cell_postprocess.output_dirname
    table_cell_viz_dir = (
        runtime.output_root / runtime.table_cell_postprocess.visualization_dirname
    )
    summary_path = runtime.output_root / "summary.json"

    summary: list[dict[str, Any]] = []

    for image_path in image_paths:
        results = detector.predict(image_path, batch_size=runtime.batch_size)
        if not results:
            print(f"{image_path.name} 未返回检测结果")
            continue

        for index, result in enumerate(results):
            result_dict = detector.result_to_dict(result)
            boxes = detector.extract_boxes(result_dict)

            stem = image_path.stem if index == 0 else f"{image_path.stem}_{index}"
            output_json = json_dir / f"{stem}.json"

            if runtime.save_visualization:
                viz_dir.mkdir(parents=True, exist_ok=True)
                result.save_to_img(save_path=str(viz_dir))

            table_count = 0
            total_cell_count = 0
            table_cell_json_path: str | None = None
            table_cell_visualization_path: str | None = None
            table_cell_payload: dict[str, Any] | None = None

            if table_cell_processor is not None:
                table_boxes = table_cell_processor.pick_table_boxes(boxes)
                table_cell_payload = table_cell_processor.detect_cells(
                    image_path=image_path,
                    table_boxes=table_boxes,
                )
                table_cell_payload["image_name"] = image_path.name
                table_cell_payload["source_layout_result_json"] = str(output_json)

                output_table_cell_json = table_cell_dir / f"{stem}.json"
                save_json(table_cell_payload, output_table_cell_json)

                if runtime.table_cell_postprocess.detector.save_visualization:
                    output_table_cell_viz = table_cell_viz_dir / f"{stem}.jpg"
                    table_cell_processor.save_visualization(
                        image_path=image_path,
                        table_cell_payload=table_cell_payload,
                        output_file=output_table_cell_viz,
                    )
                    table_cell_visualization_path = str(output_table_cell_viz)

                table_count = int(table_cell_payload.get("table_count", 0))
                total_cell_count = int(table_cell_payload.get("total_cell_count", 0))
                table_cell_json_path = str(output_table_cell_json)

            standard_boxes = build_standard_boxes(
                boxes=boxes,
                table_cell_payload=table_cell_payload,
            )
            payload = {
                "Imagepage": image_path.name,
                "Box": standard_boxes,
            }
            save_json(payload, output_json)

            summary_item: dict[str, Any] = {
                "image_name": image_path.name,
                "result_json": str(output_json),
                "box_count": len(standard_boxes),
            }
            if table_cell_processor is not None:
                summary_item["table_count"] = table_count
                summary_item["cell_count"] = total_cell_count
                summary_item["table_cell_json"] = table_cell_json_path
                summary_item["table_cell_visualization"] = table_cell_visualization_path
            summary.append(summary_item)

            if table_cell_processor is None:
                print(f"完成 {image_path.name}: 检测框 {len(standard_boxes)}")
            else:
                print(
                    f"完成 {image_path.name}: 检测框 {len(standard_boxes)}, "
                    f"表格 {table_count}, 单元格 {total_cell_count}"
                )

    summary_payload: dict[str, Any] = {
        "total_images": len(image_paths),
        "processed_results": len(summary),
        "results": summary,
    }
    if table_cell_processor is not None:
        summary_payload["table_cell_postprocess"] = {
            "enabled": True,
            "model_name": table_cell_processor.model_name,
            "output_dir": str(table_cell_dir),
            "visualization_dir": str(table_cell_viz_dir),
        }

    save_json(summary_payload, summary_path)

    print(f"推理完成，汇总文件: {summary_path}")


if __name__ == "__main__":
    main()
