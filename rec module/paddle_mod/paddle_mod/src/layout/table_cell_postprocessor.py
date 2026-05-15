from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from paddleocr import TableCellsDetection


@dataclass
class TableCellPostProcessorConfig:
    enabled: bool = True
    model_name: str = "RT-DETR-L_wired_table_cell_det"
    model_dir: str | None = None
    device: str | None = None
    threshold: float | None = 0.3
    batch_size: int = 1
    save_raw_result: bool = False
    save_visualization: bool = True


class TableCellPostProcessor:
    """对版面检测得到的 table 区域做单元格检测后处理。"""

    def __init__(self, config: TableCellPostProcessorConfig) -> None:
        self._config = config
        self._model_name = config.model_name
        self._model = self._build_model(config)

    @property
    def model_name(self) -> str:
        return self._model_name

    @staticmethod
    def pick_table_boxes(boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for box in boxes:
            if not isinstance(box, dict):
                continue
            if str(box.get("label", "")).lower() == "table":
                selected.append(box)
        return selected

    def detect_cells(
        self,
        image_path: str | Path,
        table_boxes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size

        table_items: list[dict[str, Any]] = []
        total_cell_count = 0

        for table_index, table_box in enumerate(table_boxes):
            table_label = table_box.get("label")
            table_score = table_box.get("score")
            raw_coordinate = table_box.get("coordinate")

            crop_box = self._to_crop_box(raw_coordinate, image_width, image_height)
            if crop_box is None:
                table_items.append(
                    {
                        "table_index": table_index,
                        "label": table_label,
                        "score": table_score,
                        "table_coordinate": raw_coordinate,
                        "cell_count": 0,
                        "cells": [],
                        "skipped_reason": "invalid_table_coordinate",
                    }
                )
                continue

            left, top, right, bottom = crop_box
            crop = image.crop((left, top, right, bottom))
            crop_np = np.asarray(crop)
            crop_width = right - left
            crop_height = bottom - top

            predictions = list(
                self._model.predict(
                    crop_np,
                    batch_size=self._config.batch_size,
                    threshold=self._config.threshold,
                )
            )

            cells: list[dict[str, Any]] = []
            raw_results: list[dict[str, Any]] = []

            for prediction in predictions:
                result_dict = self.result_to_dict(prediction)
                raw_results.append(result_dict)

                for cell_box in self.extract_boxes(result_dict):
                    rel_coord = self._to_float_coordinate(
                        cell_box.get("coordinate"),
                        crop_width,
                        crop_height,
                    )
                    if rel_coord is None:
                        continue

                    abs_coord = [
                        rel_coord[0] + left,
                        rel_coord[1] + top,
                        rel_coord[2] + left,
                        rel_coord[3] + top,
                    ]

                    cells.append(
                        {
                            "cls_id": cell_box.get("cls_id"),
                            "label": cell_box.get("label"),
                            "score": cell_box.get("score"),
                            "coordinate": [round(v, 3) for v in abs_coord],
                            "coordinate_in_table": [round(v, 3) for v in rel_coord],
                        }
                    )

            cells = self._sort_cells_reading_order(cells)
            for cell_index, cell in enumerate(cells):
                cell["cell_index"] = cell_index

            item: dict[str, Any] = {
                "table_index": table_index,
                "label": table_label,
                "score": table_score,
                "table_coordinate": [left, top, right, bottom],
                "cell_count": len(cells),
                "cells": cells,
            }
            if self._config.save_raw_result:
                item["raw_result"] = raw_results

            total_cell_count += len(cells)
            table_items.append(item)

        return {
            "model_name": self._model_name,
            "table_count": len(table_items),
            "total_cell_count": total_cell_count,
            "tables": table_items,
        }

    def save_visualization(
        self,
        image_path: str | Path,
        table_cell_payload: dict[str, Any],
        output_file: str | Path,
    ) -> None:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        tables = table_cell_payload.get("tables", [])
        if not isinstance(tables, list):
            tables = []

        for table in tables:
            if not isinstance(table, dict):
                continue

            table_coord = self._to_int_draw_coordinate(table.get("table_coordinate"))
            if table_coord is not None:
                draw.rectangle(table_coord, outline=(255, 40, 40), width=3)

            cells = table.get("cells", [])
            if not isinstance(cells, list):
                continue

            for cell in cells:
                if not isinstance(cell, dict):
                    continue
                cell_coord = self._to_int_draw_coordinate(cell.get("coordinate"))
                if cell_coord is not None:
                    draw.rectangle(cell_coord, outline=(60, 210, 80), width=2)

                    label = str(cell.get("cell_index", ""))
                    if label:
                        x1, y1, x2, y2 = cell_coord
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        text_w, text_h = self._measure_text(draw, label, cell_coord)
                        text_x = center_x - text_w // 2
                        text_y = center_y - text_h // 2

                        # Draw a center marker to make the number anchor obvious.
                        marker_radius = max(3, min(7, min((x2 - x1), (y2 - y1)) // 12))
                        draw.ellipse(
                            (
                                center_x - marker_radius,
                                center_y - marker_radius,
                                center_x + marker_radius,
                                center_y + marker_radius,
                            ),
                            fill=(255, 0, 0),
                            outline=(255, 255, 255),
                            width=1,
                        )

                        font = self._pick_font(cell_coord)
                        draw.text(
                            (text_x, text_y),
                            label,
                            fill=(255, 255, 0),
                            font=font,
                            stroke_width=2,
                            stroke_fill=(0, 0, 0),
                        )

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)

    def _build_model(self, config: TableCellPostProcessorConfig) -> TableCellsDetection:
        model_names = self._candidate_model_names(config.model_name)
        last_error: Exception | None = None

        for model_name in model_names:
            init_kwargs: dict[str, Any] = {"model_name": model_name}
            if config.model_dir:
                init_kwargs["model_dir"] = config.model_dir
            if config.device:
                init_kwargs["device"] = config.device

            try:
                model = TableCellsDetection(**init_kwargs)
            except Exception as exc:  # pragma: no cover - 依赖运行环境
                last_error = exc
                continue

            self._model_name = model_name
            return model

        tried = ", ".join(model_names)
        raise RuntimeError(
            f"表格单元格检测模型初始化失败，已尝试模型名: {tried}"
        ) from last_error

    @staticmethod
    def _candidate_model_names(model_name: str) -> list[str]:
        names: list[str] = []

        def append_if_needed(name: str) -> None:
            if name and name not in names:
                names.append(name)

        append_if_needed(model_name)
        if model_name.endswith("_de"):
            append_if_needed(f"{model_name}t")
        append_if_needed("RT-DETR-L_wired_table_cell_det")

        return names

    @staticmethod
    def result_to_dict(result: Any) -> dict[str, Any]:
        payload = getattr(result, "json", None)
        if isinstance(payload, dict):
            return payload
        if callable(payload):
            converted = payload()
            if isinstance(converted, dict):
                return converted

        if isinstance(result, dict):
            return result

        raise TypeError("无法将单元格检测结果转换为字典，请检查 PaddleOCR 版本是否兼容。")

    @staticmethod
    def extract_boxes(result_dict: dict[str, Any]) -> list[dict[str, Any]]:
        root = result_dict.get("res", result_dict)
        boxes = root.get("boxes", [])
        if isinstance(boxes, list):
            return [box for box in boxes if isinstance(box, dict)]
        return []

    @staticmethod
    def _to_float_coordinate(
        value: Any,
        max_width: int,
        max_height: int,
    ) -> list[float] | None:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None

        try:
            x1, y1, x2, y2 = [float(v) for v in value]
        except (TypeError, ValueError):
            return None

        x1 = max(0.0, min(float(max_width), x1))
        y1 = max(0.0, min(float(max_height), y1))
        x2 = max(0.0, min(float(max_width), x2))
        y2 = max(0.0, min(float(max_height), y2))

        if x2 <= x1 or y2 <= y1:
            return None

        return [x1, y1, x2, y2]

    @staticmethod
    def _to_crop_box(
        value: Any,
        image_width: int,
        image_height: int,
    ) -> list[int] | None:
        coord = TableCellPostProcessor._to_float_coordinate(value, image_width, image_height)
        if coord is None:
            return None

        x1, y1, x2, y2 = coord

        left = max(0, min(image_width - 1, int(math.floor(x1))))
        top = max(0, min(image_height - 1, int(math.floor(y1))))
        right = max(1, min(image_width, int(math.ceil(x2))))
        bottom = max(1, min(image_height, int(math.ceil(y2))))

        if right <= left or bottom <= top:
            return None

        return [left, top, right, bottom]

    @staticmethod
    def _to_int_draw_coordinate(value: Any) -> tuple[int, int, int, int] | None:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None

        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in value]
        except (TypeError, ValueError):
            return None

        if x2 <= x1 or y2 <= y1:
            return None

        return (x1, y1, x2, y2)

    @staticmethod
    def _measure_text(
        draw: ImageDraw.ImageDraw,
        text: str,
        cell_coord: tuple[int, int, int, int],
    ) -> tuple[int, int]:
        font = TableCellPostProcessor._pick_font(cell_coord)
        if hasattr(draw, "textbbox"):
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font, stroke_width=2)
            return max(1, right - left), max(1, bottom - top)
        width, height = draw.textsize(text, font=font)
        return max(1, width), max(1, height)

    @staticmethod
    def _pick_font(cell_coord: tuple[int, int, int, int]) -> ImageFont.ImageFont:
        x1, y1, x2, y2 = cell_coord
        cell_w = max(1, x2 - x1)
        cell_h = max(1, y2 - y1)
        preferred_size = max(14, min(34, int(min(cell_w, cell_h) * 0.35)))

        for font_name in ("arial.ttf", "simhei.ttf", "msyh.ttc"):
            try:
                return ImageFont.truetype(font_name, size=preferred_size)
            except OSError:
                continue

        return ImageFont.load_default()

    @staticmethod
    def _sort_cells_reading_order(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not cells:
            return []

        parsed: list[dict[str, Any]] = []
        heights: list[float] = []

        for cell in cells:
            if not isinstance(cell, dict):
                continue

            coord = cell.get("coordinate")
            if not isinstance(coord, (list, tuple)) or len(coord) != 4:
                continue

            try:
                x1, y1, x2, y2 = [float(v) for v in coord]
            except (TypeError, ValueError):
                continue

            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                continue

            parsed.append(
                {
                    "cell": cell,
                    "x_center": (x1 + x2) / 2.0,
                    "y_center": (y1 + y2) / 2.0,
                }
            )
            heights.append(height)

        if not parsed:
            return []

        median_height = float(np.median(np.asarray(heights, dtype=np.float32)))
        row_tolerance = max(8.0, median_height * 0.4)

        parsed.sort(key=lambda item: (item["y_center"], item["x_center"]))

        rows: list[dict[str, Any]] = []
        for item in parsed:
            placed = False
            for row in rows:
                if abs(item["y_center"] - row["y_center"]) <= row_tolerance:
                    row["items"].append(item)
                    row["y_center"] = (
                        row["y_center"] * (len(row["items"]) - 1) + item["y_center"]
                    ) / len(row["items"])
                    placed = True
                    break

            if not placed:
                rows.append({"y_center": item["y_center"], "items": [item]})

        rows.sort(key=lambda row: row["y_center"])

        ordered: list[dict[str, Any]] = []
        for row in rows:
            row_items = sorted(row["items"], key=lambda item: item["x_center"])
            for item in row_items:
                ordered.append(item["cell"])

        return ordered
