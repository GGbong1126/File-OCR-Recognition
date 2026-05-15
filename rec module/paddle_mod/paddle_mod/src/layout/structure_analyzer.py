from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from paddleocr import PPStructure


@dataclass
class LayoutAnalyzerConfig:
    use_gpu: bool = False
    lang: str = "ch"
    table: bool = True
    ocr: bool = True
    show_log: bool = False


class LayoutAnalyzer:
    """仅保留 PP-Structure 的版式分析能力。"""

    def __init__(self, config: LayoutAnalyzerConfig) -> None:
        self._engine = PPStructure(
            show_log=config.show_log,
            use_gpu=config.use_gpu,
            lang=config.lang,
            table=config.table,
            ocr=config.ocr,
        )

    def analyze_image(self, image_path: str | Path) -> list[dict[str, Any]]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")
        return list(self._engine(str(image_path)))

    @staticmethod
    def split_blocks(
        layout_results: Sequence[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        table_blocks: list[dict[str, Any]] = []
        paragraph_blocks: list[dict[str, Any]] = []

        for block in layout_results:
            block_type = str(block.get("type", "")).lower()
            if block_type == "table":
                table_blocks.append(block)
            if block_type in {"text", "title", "list"}:
                paragraph_blocks.append(block)

        return table_blocks, paragraph_blocks
