from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paddleocr import LayoutDetection


@dataclass
class PaddlePicoDetectorConfig:
    model_name: str = "PP-DocLayoutV3"
    model_dir: str | None = None
    device: str | None = None
    threshold: float | None = 0.5
    layout_nms: bool | None = True
    layout_unclip_ratio: float | None = None
    layout_merge_bboxes_mode: str | None = "large"


class PaddlePicoDetector:
    """用于文档目标检测模型 PaddlePico 的独立推理封装。"""

    def __init__(self, config: PaddlePicoDetectorConfig) -> None:
        init_kwargs: dict[str, Any] = {
            "model_name": config.model_name,
        }
        if config.model_dir:
            init_kwargs["model_dir"] = config.model_dir
        if config.device:
            init_kwargs["device"] = config.device

        self._model = LayoutDetection(**init_kwargs)
        self._threshold = config.threshold
        self._layout_nms = config.layout_nms
        self._layout_unclip_ratio = config.layout_unclip_ratio
        self._layout_merge_bboxes_mode = config.layout_merge_bboxes_mode

    def predict(self, input_path: str | Path, batch_size: int = 1) -> list[Any]:
        return list(
            self._model.predict(
                str(input_path),
                batch_size=batch_size,
                threshold=self._threshold,
                layout_nms=self._layout_nms,
                layout_unclip_ratio=self._layout_unclip_ratio,
                layout_merge_bboxes_mode=self._layout_merge_bboxes_mode,
            )
        )

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

        raise TypeError("无法将推理结果转换为字典，请检查 PaddleOCR 版本是否兼容。")

    @staticmethod
    def extract_boxes(result_dict: dict[str, Any]) -> list[dict[str, Any]]:
        root = result_dict.get("res", result_dict)
        boxes = root.get("boxes", [])
        if isinstance(boxes, list):
            return [box for box in boxes if isinstance(box, dict)]
        return []
