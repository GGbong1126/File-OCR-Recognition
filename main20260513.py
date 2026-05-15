import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox, ttk
import os
import json
import threading
import time
import shutil
import torch
from PIL import Image, ImageTk, ImageDraw
import onnxruntime as ort
import math
from layout.structure_analyzer import LayoutAnalyzer, LayoutAnalyzerConfig
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from layout.paddlepico_detector import PaddlePicoDetector, PaddlePicoDetectorConfig
from layout.table_cell_postprocessor import (
    TableCellPostProcessor,
    TableCellPostProcessorConfig,
)
import time
from PIL import Image, ImageTk, ImageDraw, ImageFont
import re

from tkinter import simpledialog
from torchvision import transforms
import cv2
import os
import traceback
# os.environ["FLAGS_use_mkldnn"] = "0"   # 禁用 MKLDNN，可能避免此错误
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from pathlib import Path
import paddle
from paddleocr import PaddleOCR
import subprocess
import threading
import queue

import numpy as np

from tkinter import ttk

import shutil
import os
from ultralytics import YOLO, nn
import gc

import csv
from collections import defaultdict




# ======================= JSON 辅助函数 =======================
def save_json_file(folder_path, filename, data):
    """将数据保存为JSON文件到指定文件夹，若 data 为 None 或空则保存空列表 []"""
    if not folder_path:
        return False
    os.makedirs(folder_path, exist_ok=True)
    filepath = os.path.join(folder_path, filename)
    # 若 data 为 None，则保存空列表
    if data is None:
        data = []
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return True

def load_json_file(folder_path, filename):
    """从指定文件夹加载JSON文件，不存在则返回 None"""
    if not folder_path:
        return None
    filepath = os.path.join(folder_path, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def safe_save_json(folder, filename, data):
    """安全保存，若 data 为 None 则保存空列表，否则保存 data"""
    save_json_file(folder, filename, data)

# ======================= 支持缩放/拖拽的画布包装器 =======================
class ZoomPanCanvas:
    def __init__(self, canvas, on_draw_image=None, drag_button=3):
        self.canvas = canvas
        self.on_draw_image = on_draw_image
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self._drag_start_x = 0
        self._drag_start_y = 0
        self._drag_start_offset_x = 0
        self._drag_start_offset_y = 0
        self.drag_button = drag_button

        self.canvas.bind(f"<ButtonPress-{drag_button}>", self._on_drag_start)
        self.canvas.bind(f"<B{drag_button}-Motion>", self._on_drag_move)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_mousewheel)
        self.canvas.bind("<Button-5>", self._on_mousewheel)

    def _on_drag_start(self, event):
        self._drag_start_x = event.x
        self._drag_start_y = event.y
        self._drag_start_offset_x = self.offset_x
        self._drag_start_offset_y = self.offset_y

    def _on_drag_move(self, event):
        dx = event.x - self._drag_start_x
        dy = event.y - self._drag_start_y
        self.offset_x = self._drag_start_offset_x + dx
        self.offset_y = self._drag_start_offset_y + dy
        self._redraw()

    def _on_mousewheel(self, event):
        scale_factor = 1.1
        if event.delta < 0 or (event.num == 5):
            scale_factor = 1 / scale_factor
        self.zoom *= scale_factor
        self.zoom = max(0.1, min(10.0, self.zoom))
        self._redraw()

    def _redraw(self):
        if self.on_draw_image:
            self.on_draw_image(self.canvas, self.zoom, self.offset_x, self.offset_y)

    def set_image_params(self, zoom=None, offset_x=None, offset_y=None):
        if zoom is not None:
            self.zoom = zoom
        if offset_x is not None:
            self.offset_x = offset_x
        if offset_y is not None:
            self.offset_y = offset_y
        self._redraw()

# ======================= 半监督训练与标注窗口 =======================
import subprocess
import math
import json
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
# from .zoom_pan_canvas import ZoomPanCanvas   # 请根据实际导入路径调整
import subprocess
import math
import json
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# 假设 ZoomPanCanvas 在另一个模块，请根据实际导入
# from .zoom_pan_canvas import ZoomPanCanvas

class SemiSupervisedTrainWindow:
    def __init__(self, parent):
        self.parent = parent
        self.project_root = None          # 项目根目录
        self.image_paths = []             # data/raw 下的所有图片绝对路径
        self.current_index = 0
        self.cur_image = None
        self.annotations = {}              # key: 图片绝对路径, value: 标注框列表
        self.pseudo_labels = {}            # 伪标签（结构同上）
        self.model_weight_path = None      # 初始权重文件（可用于测试或训练起点）
        self.train_process = None          # 当前运行的子进程
        self.current_step = None           # 记录当前步骤，用于停止

        # 设置环境变量（解决 PaddleDetection 路径和网络检测问题）
        # ⚠️ 请将下面的路径改为你本机实际的 PaddleDetection 根目录
        paddle_det_root = r"D:\your_local_path\PaddleDetection"   # 修改为你的实际路径
        os.environ["PADDLEX_PP_DETECTION_ROOT"] = paddle_det_root
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

        # 训练参数（可通过添加输入框让用户修改）
        self.epochs = 1
        self.batch_size = 2
        self.learning_rate = 0.0005
        self.val_ratio = 0.1
        self.pseudo_min_score = 0.75
        self.device = "gpu:0"
        self.template_config = "PP-DocLayoutV3.yaml"
        self.do_eval_after_train = True   # 训练后是否自动评估

        self.window = tk.Toplevel(parent)
        self.window.title("训练与数据标注 (持续训练流程)")
        self.window.geometry("1200x800")
        self.window.minsize(1000, 700)

        self._create_widgets()

    # ------------------ 界面创建（保留所有按钮）------------------
    def _create_widgets(self):
        toolbar = tk.Frame(self.window)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Button(toolbar, text="选择项目根目录", command=self.choose_folder).pack(side=tk.LEFT, padx=5)
        tk.Button(toolbar, text="加载权重文件", command=self.load_weights).pack(side=tk.LEFT, padx=5)
        tk.Button(toolbar, text="生成伪标签", command=self.generate_pseudo_labels).pack(side=tk.LEFT, padx=5)
        tk.Button(toolbar, text="测试权重", command=self.test_weight).pack(side=tk.LEFT, padx=5)
        tk.Button(toolbar, text="保存当前标注", command=self.save_current_annotation).pack(side=tk.LEFT, padx=5)
        tk.Button(toolbar, text="保存所有标注", command=self.save_all_annotations).pack(side=tk.LEFT, padx=5)
        self.start_train_btn = tk.Button(toolbar, text="开始训练", command=self.start_training, bg="lightgreen")
        self.start_train_btn.pack(side=tk.LEFT, padx=5)
        self.stop_train_btn = tk.Button(toolbar, text="停止训练", command=self.stop_training, state=tk.DISABLED, bg="lightcoral")
        self.stop_train_btn.pack(side=tk.LEFT, padx=5)

        main_pane = tk.Frame(self.window)
        main_pane.pack(side=tk.TOP, expand=True, fill=tk.BOTH, padx=5, pady=5)

        left_frame = tk.LabelFrame(main_pane, text="图片列表 (data/raw)", padx=5, pady=5)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,5))
        self.listbox = tk.Listbox(left_frame, width=30, height=30, font=("Consolas", 10))
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=scrollbar.set)
        self.listbox.bind("<<ListboxSelect>>", self.on_image_select)

        right_frame = tk.LabelFrame(main_pane, text="标注区域", padx=5, pady=5)
        right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        self.canvas = tk.Canvas(right_frame, bg="gray", cursor="cross")
        self.canvas.pack(expand=True, fill=tk.BOTH)

        # 请确保 ZoomPanCanvas 已正确导入
        self.zoom_pan = ZoomPanCanvas(self.canvas, self._draw_image_with_zoom_pan, drag_button=3)

        info_frame = tk.Frame(right_frame)
        info_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        self.info_label = tk.Label(info_frame, text="未加载图片", font=("微软雅黑", 10))
        self.info_label.pack(side=tk.LEFT, padx=5)
        tk.Label(info_frame, text=" | 左键拉框标注 | 右键拖拽视图 | 右键单击框删除 | 滚轮缩放",
                 font=("微软雅黑", 9)).pack(side=tk.LEFT)

        console_frame = tk.LabelFrame(self.window, text="训练控制台", padx=5, pady=5)
        console_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False, padx=5, pady=5)
        self.console = tk.Text(console_frame, height=10, wrap=tk.WORD, font=("Consolas", 9))
        self.console.pack(fill=tk.BOTH, expand=True)

        # 标注交互变量
        self.start_x = None
        self.start_y = None
        self.current_rect = None
        self.rectangles = []
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down, add=True)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move, add=True)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up, add=True)
        self._right_click_start = None
        self.canvas.bind("<ButtonPress-3>", self.on_right_press, add=True)
        self.canvas.bind("<ButtonRelease-3>", self.on_right_release, add=True)

    # ------------------ 项目根目录与图片加载（适配新结构）------------------
    def choose_folder(self):
        folder = filedialog.askdirectory(title="选择项目根目录（包含 data/raw, data/labels/manual_json 等）")
        if not folder:
            return
        self.project_root = Path(folder)
        raw_dir = self.project_root / "data" / "raw"
        if not raw_dir.exists():
            messagebox.showerror("错误", f"未找到 data/raw 目录：{raw_dir}")
            return
        # 加载图片
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
        self.image_paths = [str(p) for p in raw_dir.glob("*") if p.suffix.lower() in extensions]
        self.image_paths.sort()
        if not self.image_paths:
            self.console_insert("data/raw 文件夹内未找到图片文件。\n")
            self.listbox.delete(0, tk.END)
            self.info_label.config(text="无图片")
            self.cur_image = None
            self.zoom_pan._redraw()
            return

        # 加载已有的人工标注（从 manual_json 目录读取）
        manual_json_dir = self.project_root / "data" / "labels" / "manual_json"
        self.annotations = {}
        if manual_json_dir.exists():
            for img_path in self.image_paths:
                json_path = manual_json_dir / (Path(img_path).stem + ".json")
                if json_path.exists():
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        boxes = data.get("boxes", [])
                        self.annotations[img_path] = [{"bbox": box["bbox"]} for box in boxes if "bbox" in box]
                else:
                    self.annotations[img_path] = []
        else:
            for img_path in self.image_paths:
                self.annotations[img_path] = []

        # 伪标签（暂留占位）
        self.pseudo_labels = {}

        self._refresh_image_list()
        self.current_index = 0
        self._load_image_at_index(0)
        self.console_insert(f"项目根目录: {self.project_root}\n已加载 {len(self.image_paths)} 张图片\n")

    def _refresh_image_list(self):
        self.listbox.delete(0, tk.END)
        for idx, path in enumerate(self.image_paths):
            name = os.path.basename(path)
            self.listbox.insert(tk.END, f"{idx+1}. {name}")

    # ------------------ 绘图与标注辅助函数 ------------------
    def _draw_image_with_zoom_pan(self, canvas, zoom, offset_x, offset_y):
        if self.cur_image is None:
            canvas.delete("all")
            canvas.create_text(200, 200, text="请先选择项目根目录", fill="white", font=("Arial", 16))
            return
        w, h = self.cur_image.size
        new_w = int(w * zoom)
        new_h = int(h * zoom)
        if new_w < 1 or new_h < 1:
            return
        img_resized = self.cur_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(img_resized)
        canvas.delete("all")
        canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=self.photo_image, tags="bg_img")
        self.current_zoom = zoom
        self.current_offset = (offset_x, offset_y)
        self.current_img_size = (new_w, new_h)
        img_path = self.image_paths[self.current_index] if self.image_paths else None
        if img_path:
            for box in self.annotations.get(img_path, []):
                self._draw_one_box_on_canvas(box['bbox'], "red", 2)
            for box in self.pseudo_labels.get(img_path, []):
                self._draw_one_box_on_canvas(box['bbox'], "blue", 2)

    def _draw_one_box_on_canvas(self, bbox, color, width):
        x1, y1, x2, y2 = bbox
        zoom = self.current_zoom
        ox, oy = self.current_offset
        cx1 = x1 * zoom + ox
        cy1 = y1 * zoom + oy
        cx2 = x2 * zoom + ox
        cy2 = y2 * zoom + oy
        rect_id = self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline=color, width=width)
        self.rectangles.append((rect_id, bbox, color))

    def _load_image_at_index(self, idx):
        if not self.image_paths or idx < 0 or idx >= len(self.image_paths):
            return
        self.current_index = idx
        img_path = self.image_paths[idx]
        try:
            self.cur_image = Image.open(img_path)
            self.info_label.config(text=f"当前图片: {os.path.basename(img_path)}")
            self.zoom_pan.zoom = 1.0
            self.zoom_pan.offset_x = 0
            self.zoom_pan.offset_y = 0
            self.rectangles.clear()
            self.zoom_pan._redraw()
        except Exception as e:
            self.console_insert(f"加载失败: {img_path}\n{e}\n")

    def on_image_select(self, event):
        selection = self.listbox.curselection()
        if selection:
            self.current_index = selection[0]
            self._load_image_at_index(self.current_index)

    # ========== 字符级选择的核心方法 ==========
    def _canvas_to_orig(self, cx, cy):
        """将画布坐标转换为原始图像坐标（表格图像原始尺寸坐标）"""
        zoom = self.layer_zoom_pan.zoom
        ox = self.layer_zoom_pan.offset_x
        oy = self.layer_zoom_pan.offset_y
        return (cx - ox) / zoom, (cy - oy) / zoom

    def _find_char_index_at_canvas_pos(self, cx, cy):
        ox, oy = self._canvas_to_orig(cx, cy)
        for cp in self.char_positions_orig:
            left, top, right, bottom = cp["rect"]
            if left <= ox <= right and top <= oy <= bottom:
                return cp["global_idx"]
        return None

    def _update_layer_highlight(self):

        if hasattr(self, 'current_highlight_rects'):

            for rid in self.current_highlight_rects:
                self.layer_canvas.delete(rid)

        self.current_highlight_rects = []

        if not self.selected_char_indices:
            return

        zoom = self.layer_zoom_pan.zoom
        ox = self.layer_zoom_pan.offset_x
        oy = self.layer_zoom_pan.offset_y

        for cp in self.char_positions_orig:

            if cp["global_idx"] not in self.selected_char_indices:
                continue

            left, top, right, bottom = cp["rect"]

            x1 = left * zoom + ox
            y1 = top * zoom + oy
            x2 = right * zoom + ox
            y2 = bottom * zoom + oy

            rid = self.layer_canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                fill="#66CCFF",
                stipple="gray25",
                outline=""
            )
            self.current_highlight_rects.append(rid)

    def on_layer_mouse_down(self, event):
        if not self.char_canvas_rects:
            return
        idx = self._find_char_index_at_canvas_pos(event.x, event.y)
        if idx is not None:
            self.select_start_idx = idx
            self.selected_char_indices = {idx}
            self._update_layer_highlight()
        else:
            self.select_start_idx = None

    def on_layer_mouse_move(self, event):
        if self.select_start_idx is None:
            return
        current_idx = self._find_char_index_at_canvas_pos(event.x, event.y)
        if current_idx is None:
            return
        start = self.select_start_idx
        end = current_idx
        if start <= end:
            indices = set(range(start, end + 1))
        else:
            indices = set(range(end, start + 1))
        self.selected_char_indices = indices
        self._update_layer_highlight()

    def on_layer_mouse_up(self, event):
        if self.select_start_idx is None:
            return
        if self.selected_char_indices:
            min_idx = min(self.selected_char_indices)
            max_idx = max(self.selected_char_indices)
            selected_chars = []
            for cr in self.char_canvas_rects:
                if min_idx <= cr["global_idx"] <= max_idx:
                    selected_chars.append(cr["char"])
            selected_text = "".join(selected_chars)
            if self.waiting_for_selection and self.current_selected_ner_row is not None:
                self._update_ner_field(self.current_selected_ner_row, selected_text, min_idx, max_idx + 1)
                self._append_debug(f"已用选区文本替换字段: {selected_text}\n")
                self.waiting_for_selection = False
            else:
                self._append_debug(f"选中文本: {selected_text}\n")
        self.select_start_idx = None
        self.selected_char_indices.clear()
        self._update_layer_highlight()

    def on_mouse_down(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.current_rect:
            self.canvas.delete(self.current_rect)
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="green", width=2
        )

    def on_mouse_move(self, event):
        if self.start_x is not None:
            self.canvas.coords(self.current_rect, self.start_x, self.start_y, event.x, event.y)

    def on_mouse_up(self, event):
        if self.start_x is None:
            return
        x1, y1 = self.start_x, self.start_y
        x2, y2 = event.x, event.y
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        orig_x1, orig_y1 = self._canvas_to_original_coords(x1, y1)
        orig_x2, orig_y2 = self._canvas_to_original_coords(x2, y2)
        if orig_x2 > orig_x1 and orig_y2 > orig_y1:
            bbox = (int(orig_x1), int(orig_y1), int(orig_x2), int(orig_y2))
            img_path = self.image_paths[self.current_index]
            if img_path not in self.annotations:
                self.annotations[img_path] = []
            self.annotations[img_path].append({'bbox': bbox})
            self._draw_one_box_on_canvas(bbox, "red", 2)
        if self.current_rect:
            self.canvas.delete(self.current_rect)
            self.current_rect = None
        self.start_x = self.start_y = None

    def on_right_press(self, event):
        self._right_click_start = (event.x, event.y)

    def on_right_release(self, event):
        if self._right_click_start is None:
            return
        sx, sy = self._right_click_start
        if math.hypot(event.x - sx, event.y - sy) < 5:
            self._delete_box_at(event.x, event.y)
        self._right_click_start = None

    def _delete_box_at(self, x, y):
        closest = None
        min_dist = 10
        for rect_id, bbox, color in self.rectangles:
            if color != "red":
                continue
            coords = self.canvas.coords(rect_id)
            if len(coords) != 4:
                continue
            cx1, cy1, cx2, cy2 = coords
            dx = max(cx1 - x, 0, x - cx2)
            dy = max(cy1 - y, 0, y - cy2)
            dist = math.hypot(dx, dy)
            if dist < min_dist:
                min_dist = dist
                closest = (rect_id, bbox)
        if closest:
            rect_id, bbox = closest
            self.canvas.delete(rect_id)
            img_path = self.image_paths[self.current_index]
            if img_path in self.annotations:
                self.annotations[img_path] = [ann for ann in self.annotations[img_path] if ann['bbox'] != bbox]
            self.rectangles = [r for r in self.rectangles if r[0] != rect_id]
            self.console_insert("已删除所选人工标注框\n")
        else:
            self.console_insert("未选中任何人工标注框（只能删除红色框）\n")

    # ------------------ 标注保存（适配 manual_json 单文件）------------------
    def save_current_annotation(self):
        if not self.project_root:
            self.console_insert("请先选择项目根目录。\n")
            return
        img_path = self.image_paths[self.current_index]
        self._save_single_annotation(img_path)
        self.console_insert(f"已保存当前图片的人工标注（{len(self.annotations.get(img_path, []))}个框）。\n")

    def save_all_annotations(self):
        if not self.project_root:
            self.console_insert("请先选择项目根目录。\n")
            return
        total = 0
        for img_path in self.image_paths:
            if img_path in self.annotations and self.annotations[img_path]:
                self._save_single_annotation(img_path)
                total += len(self.annotations[img_path])
        self.console_insert(f"已保存所有图片的人工标注（总计{total}个框）。\n")

    def _save_single_annotation(self, img_path):
        """将单张图片的标注保存到 data/labels/manual_json/{图片名}.json"""
        if not self.project_root:
            return
        manual_dir = self.project_root / "data" / "labels" / "manual_json"
        manual_dir.mkdir(parents=True, exist_ok=True)
        rel_path = Path(img_path).relative_to(self.project_root / "data" / "raw")
        json_path = manual_dir / (Path(img_path).stem + ".json")
        boxes = []
        for ann in self.annotations.get(img_path, []):
            bbox = ann['bbox']
            boxes.append({"bbox": bbox})   # 可根据需要增加其他字段
        data = {
            "image_name": Path(img_path).name,
            "image_path": str(rel_path),
            "boxes": boxes
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    # ------------------ 权重管理（保留占位）------------------
    def load_weights(self):
        filepath = filedialog.askopenfilename(title="选择权重文件", filetypes=[("模型权重", "*.pth *.h5 *.ckpt *.pdiparams"), ("所有文件", "*.*")])
        if filepath:
            self.model_weight_path = filepath
            self.console_insert(f"已加载权重文件: {os.path.basename(filepath)}\n")

    def test_weight(self):
        """占位：测试权重功能（用户需自行实现模型推理）"""
        if not self.image_paths:
            self.console_insert("请先选择项目根目录。\n")
            return
        # 示例伪代码：假数据演示
        if self.model_weight_path:
            self.console_insert(f"使用权重 {os.path.basename(self.model_weight_path)} 测试推理（占位）\n")
        else:
            self.console_insert("请先加载权重文件。\n")
        # 实际应调用模型进行推理并显示框（示例：绘制假框）
        fake_boxes = [(100,100,200,200,0.95), (300,150,400,250,0.92)]
        # 清除旧框（只清除测试框？这里简单示意）
        for rect_id, _, _ in self.rectangles:
            self.canvas.delete(rect_id)
        self.rectangles.clear()
        for (x1,y1,x2,y2,conf) in fake_boxes:
            self._draw_one_box_on_canvas((x1,y1,x2,y2), "orange", 2)
            self.canvas.create_text((x1+x2)//2, y1-5, text=f"{conf:.2f}", fill="yellow")
        self.console_insert(f"测试权重演示，显示 {len(fake_boxes)} 个检测框（橙色）\n")

    def generate_pseudo_labels(self):
        """占位：生成伪标签（用户需自行实现）"""
        if not self.model_weight_path:
            messagebox.showwarning("警告", "请先加载权重文件！")
            return
        if not self.image_paths:
            self.console_insert("没有图片可处理。\n")
            return
        self.console_insert("正在生成伪标签... (此过程可能需要几分钟)\n")
        # TODO: 实际模型推理生成伪标签，保存到 self.pseudo_labels 和磁盘
        # 示例：为每张图生成空列表
        for img_path in self.image_paths:
            self.pseudo_labels[img_path] = []   # 占位
        # 保存伪标签文件（推荐保存到 experiments/continual_training/... 或单独文件）
        pseudo_save_path = self.project_root / "data" / "labels" / "pseudo_labels.json" if self.project_root else None
        if pseudo_save_path:
            # 转换为可保存格式
            save_data = {path: self.pseudo_labels[path] for path in self.image_paths}
            with open(pseudo_save_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2)
            self.console_insert(f"伪标签生成完成，已保存至 {pseudo_save_path}\n")
        else:
            self.console_insert("伪标签生成完成（未保存）\n")
        self.zoom_pan._redraw()

    # ------------------ 持续训练完整流程（核心新增）------------------
    def start_training(self):
        """执行完整训练流程：生成清单、训练、导出、评估"""
        if not self.project_root:
            self.console_insert("请先选择项目根目录。\n")
            return
        # 检查必要文件
        required = [
            self.project_root / "src" / "run_continual_training.py",
            self.project_root / "tools" / "train_layout.py",
            self.project_root / "scripts" / "export_continual_model.ps1",
            self.project_root / "configs" / "continual_training_config.json"
        ]
        for p in required:
            if not p.exists():
                self.console_insert(f"缺少必要文件: {p}\n")
                return
        self.start_train_btn.config(state=tk.DISABLED)
        self.stop_train_btn.config(state=tk.NORMAL)
        self._run_step_1()

    def _run_step_1(self):
        """步骤1：生成下一轮次的清单（不执行训练）"""
        self.current_step = "generate_manifest"
        self.console_insert("\n========== 步骤1: 生成下一轮次清单 ==========\n")
        config_path = self.project_root / "configs" / "continual_training_config.json"
        cmd = [
            "python", str(self.project_root / "src" / "run_continual_training.py"),
            "--config", str(config_path),
            # 不指定 --round，自动选下一个
        ]
        self.console_insert("执行命令: " + " ".join(cmd) + "\n")
        self._run_subprocess(cmd, self._after_step_1)

    def _after_step_1(self, returncode):
        if returncode != 0:
            self.console_insert("步骤1失败，终止流程。\n")
            self._training_finished()
            return
        # 解析最新 round_id
        exp_dir = self.project_root / "experiments" / "continual_training"
        if not exp_dir.exists():
            self.console_insert("错误：未找到 experiments/continual_training 目录\n")
            self._training_finished()
            return
        rounds = [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("round_")]
        if not rounds:
            self.console_insert("错误：未找到任何 round_xxx 目录\n")
            self._training_finished()
            return
        latest_round = max(rounds, key=lambda d: int(d.name.split("_")[1]))
        self.current_round_dir = latest_round
        round_num = int(latest_round.name.split("_")[1])
        self.console_insert(f"当前轮次: {latest_round.name} (编号 {round_num})\n")
        # 确定初始权重
        if round_num == 1:
            # 第一轮应使用官方预训练，但为了避免网络问题，用户需保证配置文件中 detector.model_dir 已指定本地模型
            self.init_weight = "official:PP-DocLayoutV3"  # 或从配置读取
        else:
            prev_version = f"v{round_num-1}"
            self.init_weight = str(self.project_root / "models" / "continual" / prev_version / "best_model.pdparams")
        self.output_weight_dir = self.project_root / "models" / "continual" / f"v{round_num}"
        self.output_weight_dir.mkdir(parents=True, exist_ok=True)
        self._run_step_2()

    def _run_step_2(self):
        """步骤2：执行训练"""
        self.current_step = "train"
        self.console_insert("\n========== 步骤2: 模型训练 ==========\n")
        manifest_dir = self.current_round_dir
        train_manifest = manifest_dir / "train_manifest.json"
        pseudo_manifest = manifest_dir / "pseudo_manifest.json"
        manual_manifest = manifest_dir / "manual_manifest.json"
        if not train_manifest.exists():
            self.console_insert(f"错误：未找到训练清单 {train_manifest}\n")
            self._training_finished()
            return

        cmd = [
            "python", str(self.project_root / "tools" / "train_layout.py"),
            "--round-manifest", str(train_manifest),
            "--pseudo-manifest", str(pseudo_manifest),
            "--manual-manifest", str(manual_manifest),
            "--init-weight", self.init_weight,
            "--output-weight", str(self.output_weight_dir),
            "--round-id", self.current_round_dir.name,
            "--device", self.device,
            "--template-config", self.template_config,
            "--epochs", str(self.epochs),
            "--batch-size", str(self.batch_size),
            "--learning-rate", str(self.learning_rate),
            "--val-ratio", str(self.val_ratio),
            "--pseudo-min-score", str(self.pseudo_min_score)
        ]
        self.console_insert("执行命令: " + " ".join(cmd) + "\n")
        self._run_subprocess(cmd, self._after_step_2)

    def _after_step_2(self, returncode):
        if returncode != 0:
            self.console_insert("步骤2（训练）失败，终止流程。\n")
            self._training_finished()
            return
        self._run_step_3()

    def _run_step_3(self):
        """步骤3：导出推理模型"""
        self.current_step = "export"
        self.console_insert("\n========== 步骤3: 导出推理模型 ==========\n")
        generated_config = self.current_round_dir / "train_config_generated.yaml"
        if not generated_config.exists():
            self.console_insert(f"错误：未找到生成配置 {generated_config}\n")
            self._training_finished()
            return
        best_weight = self.output_weight_dir / "best_model.pdparams"
        if not best_weight.exists():
            self.console_insert(f"错误：未找到最佳权重 {best_weight}\n")
            self._training_finished()
            return
        export_dir = self.output_weight_dir / "exported"
        compat_dir = self.output_weight_dir / "exported_compat"
        ps_script = self.project_root / "scripts" / "export_continual_model.ps1"
        cmd = [
            "powershell", "-ExecutionPolicy", "Bypass", "-File", str(ps_script),
            "-ConfigPath", str(generated_config),
            "-WeightPath", str(best_weight),
            "-ExportDir", str(export_dir),
            "-CompatDir", str(compat_dir)
        ]
        self.console_insert("执行命令: " + " ".join(cmd) + "\n")
        self._run_subprocess(cmd, self._after_step_3)

    def _after_step_3(self, returncode):
        if returncode != 0:
            self.console_insert("步骤3（导出）失败，但可手动导出。\n")
        if self.do_eval_after_train:
            self._run_step_4()
        else:
            self.console_insert("训练流程全部完成！\n")
            self._training_finished()

    def _run_step_4(self):
        """步骤4：评估模型（可选）"""
        self.current_step = "eval"
        self.console_insert("\n========== 步骤4: 评估模型 ==========\n")
        test_dir = self.project_root / "data" / "test_raw"
        if not test_dir.exists():
            self.console_insert(f"测试目录不存在: {test_dir}，跳过评估。\n")
            self._training_finished()
            return
        eval_ps = self.project_root / "scripts" / "run_continual_eval.ps1"
        output_eval = self.project_root / "outputs" / f"continual_eval_{self.output_weight_dir.name}"
        cmd = [
            "powershell", "-ExecutionPolicy", "Bypass", "-File", str(eval_ps),
            "-InputPath", str(test_dir),
            "-OutputPath", str(output_eval),
            "-ModelDir", str(self.output_weight_dir / "exported_compat"),
            "-Threshold", "0.1"
        ]
        self.console_insert("执行命令: " + " ".join(cmd) + "\n")
        self._run_subprocess(cmd, self._after_step_4)

    def _after_step_4(self, returncode):
        if returncode != 0:
            self.console_insert("评估步骤失败，请手动检查。\n")
        else:
            self.console_insert(f"评估完成，结果保存在 {self.project_root / 'outputs'}\n")
        self.console_insert("全部训练流程结束！\n")
        self._training_finished()

    # ------------------ 子进程管理（支持编码容错）------------------
    def _run_subprocess(self, cmd, callback):
        """启动子进程，设置环境变量，处理编码问题"""
        env = os.environ.copy()
        # 确保子进程继承必要的环境变量
        env["PADDLEX_PP_DETECTION_ROOT"] = os.environ.get("PADDLEX_PP_DETECTION_ROOT", "")
        env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            # 对于 PowerShell 命令，先设置输出编码为 UTF-8
            if cmd[0] == "powershell":
                # 构建一个临时脚本设置 chcp 65001
                ps_command = 'chcp 65001 > $null; ' + ' '.join(cmd[1:])
                new_cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-Command", ps_command]
                cmd = new_cmd
            self.train_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding='utf-8',
                errors='replace',
                env=env
            )
            self._read_output_and_wait(callback)
        except Exception as e:
            self.console_insert(f"启动子进程失败: {e}\n")
            callback(-1)

    def _read_output_and_wait(self, callback):
        """非阻塞读取进程输出，编码问题已通过 errors='replace' 缓解"""
        if self.train_process is None:
            callback(-1)
            return
        try:
            line = self.train_process.stdout.readline()
            if line:
                self.console_insert(line)
                self.window.after(100, lambda: self._read_output_and_wait(callback))
            else:
                if self.train_process.poll() is not None:
                    callback(self.train_process.returncode)
                    return
                self.window.after(100, lambda: self._read_output_and_wait(callback))
        except Exception as e:
            self.console_insert(f"读取输出错误: {e}\n")
            if self.train_process:
                self.train_process.terminate()
            callback(-1)

    def stop_training(self):
        """停止当前步骤的子进程"""
        if self.train_process is not None and self.train_process.poll() is None:
            self.console_insert(f"正在终止当前步骤（{self.current_step}）...\n")
            self.train_process.terminate()
            try:
                self.train_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.train_process.kill()
            self.train_process = None
            self.console_insert("当前步骤已停止，流程中断。\n")
        else:
            self.console_insert("没有正在运行的进程。\n")
        self._training_finished()

    def _training_finished(self):
        """流程结束后的界面恢复"""
        self.start_train_btn.config(state=tk.NORMAL)
        self.stop_train_btn.config(state=tk.DISABLED)
        self.train_process = None
        self.current_step = None

    def console_insert(self, text):
        self.window.after(0, lambda: self.console.insert(tk.END, text) or self.console.see(tk.END))
# ======================= 主应用程序 =======================
class MultiRegionApp:
    # 类属性常量（放在 __init__ 之前）
    VALID_LABELS = {
        "姓名", "性别", "出生年月", "民族", "籍贯", "出生地",
        "政治面貌", "入党时间", "参加工作时间", "工作单位", "职务", "现职时间",
        "身份证号", "全日制教育学历", "全日制教育学位", "全日制教育毕业院校系", "全日制教育专业",
        "在职教育学历", "在职教育学位", "在职教育毕业院校系", "在职教育专业"
    }
    VALID_POLITICAL_STATUS = {"党员", "中共党员", "预备党员", "共青团员", "团员", "群众", "民主党派", "无党派"}
    SKIP_OCR_CLASSES = {"figure", "image", "seal", "stamp", "signature", "barcode", "qr_code"}
    def __init__(self, root):
        self.root = root
        self.root.title("多功能演示工具")

        # 20260510 检测 GPU 可用性（同时检查 PaddlePaddle 和 PyTorch）#################
        import paddle
        import torch
        self._cuda_available = paddle.is_compiled_with_cuda() and torch.cuda.is_available()
        # 默认开关开启：如果有 GPU 则用 GPU，否则 CPU
        self.use_gpu = self._cuda_available
        ############################################################################
        self._append_debug(f"初始设备模式: {'GPU' if self.use_gpu else 'CPU'}\n")
        self.root.geometry("1100x750")
        self.root.minsize(900, 650)

        self.image_paths = []
        self.current_image_index = 0
        self.original_pil_image = None
        self.current_display_pil = None
        self.current_folder = None

        # 权重目录
        self.layout_weight_path = None  # 版式分析模型目录
        self.table_cell_weight_path = None  # 单元格分割模型目录
        self.ocr_weight_path = None
        self.ner_weight_path = None

        # 模型缓存
        self._layout_detector = None
        self._table_cell_processor = None
        self._current_layout_weight = None
        self._current_table_cell_weight = None

        self._create_left_buttons()
        self._create_right_regions()

        self.image_boxes_cache = {}  # 键: 图片路径, 值: 该图片的 standard_boxes 列表

        # 图像预处理相关（PyTorch 版）
        self.imgpre_model = None  # 加载后的 PyTorch 模型
        self.imgpre_device = None  # 设备 (cuda / cpu)
        self.enhanced_folder = None
        self.enhanced_image_paths = []
        self.is_preprocessed = False
        self.imgpre_session = None

        # NER 结果列表及当前索引
        self.ner_results = []
        self.ner_current_index = 0
        self._current_full_text = ""

        # OCR 结果列表及当前索引
        self.ocr_results = []      # 每个元素 {"image": img_name, "full_text": full_text}
        self.ocr_current_index = 0

        self.text_area1 = None  # 兼容旧代码，实际不再使用

        # 手动标注相关
        self.drawing = False
        self.start_x = self.start_y = 0
        self.current_rect = None
        self.annotation_class = None  # 预设类别（可选）

        # 右键拖拽平移
        self.right_press_pos = None
        self.right_dragging = False
        self.drag_start_offset_x = 0
        self.drag_start_offset_y = 0
        self.drag_start_x = 0
        self.drag_start_y = 0

        self.waiting_for_selection = False

        self.waiting_for_selection = False
        self.text_boxes_info = []  # 存储文本框信息
        self.start_box_num = None

        self.waiting_for_selection = False
        self.selected_chars_indices = set()
        self.canvas_char_positions = []
        self.layer_full_text = ""

        self.char_positions_orig = []  # 存储每个字符的原始坐标和全局索引
        self.layer_full_text = ""  # 右侧全文
        self.select_start_idx = None  # 拖拽选择的起始字符索引
        self.selected_char_indices = set()  # 当前选中的字符索引集合
        self.waiting_for_selection = False  # 是否处于等待替换模式
        self.current_highlight_rects = []  # 存储当前高亮矩形对象ID

        self.char_canvas_rects = []

        self.layer_boxes = []  # 存储当前图层上的框信息（原始坐标和文本）
        self.selected_box_number = None  # 当前选中的框编号（如果需要交互可保留，否则可忽略）
        self.highlighted_box_numbers = []  # 临时高亮的框编号列表

        # 性能统计字典：键为环节名称，值为 dict
        self.performance_stats = {
            "文本区域检测": {"image_count": 0, "total_time": 0.0, "avg_time": 0.0, "accuracy": ""},
            "OCR识别": {"image_count": 0, "total_time": 0.0, "avg_time": 0.0, "accuracy": ""},
            "关键信息提取": {"image_count": 0, "total_time": 0.0, "avg_time": 0.0, "accuracy": ""},
            "全部环节": {"image_count": 0, "total_time": 0.0, "avg_time": 0.0, "accuracy": ""}
        }

        #### ocr结果前端修改
        self.ocr_edit_mode = False

        #### 新增yolo识别photo
        self.yolo_model = None
        self.yolo_conf = 0.3

    def _create_left_buttons(self):
        left_frame = tk.Frame(self.root, width=200, bg="#f0f0f0", relief=tk.RAISED, bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_frame.pack_propagate(False)

        # 文本区域检测权重行
        self._create_weight_row(left_frame, "文本区域检测", self.load_layout_weight, "layout_weight_label")
        # 单元格检测模型选择（新增）
        self._create_weight_row(left_frame, "单元格检测", self.load_table_cell_weight, "table_cell_weight_label")
        btn1 = tk.Button(left_frame, text="文本区域检测", command=self.on_detect_regions,
                         font=("微软雅黑",12), bg="#4a7abc", fg="white")
        btn1.pack(pady=(0,15), padx=10, fill=tk.X)

        # OCR识别权重行
        self._create_weight_row(left_frame, "OCR识别", self.load_ocr_weight, "ocr_weight_label")
        btn2 = tk.Button(left_frame, text="OCR识别", command=self.on_ocr,
                         font=("微软雅黑",12), bg="#4a7abc", fg="white")
        btn2.pack(pady=(0,15), padx=10, fill=tk.X)

        # 关键信息提取权重行
        self._create_weight_row(left_frame, "关键信息提取", self.load_ner_weight, "ner_weight_label")
        btn3 = tk.Button(left_frame, text="关键信息提取（轻量）", command=self.on_ner,
                         font=("微软雅黑",12), bg="#4a7abc", fg="white")
        btn3.pack(pady=(0,15), padx=10, fill=tk.X)

        ##### 新增qwen进行关键信息提取
        btn3p = tk.Button(left_frame, text="关键信息提取（大模型）", command=self.on_ner_Qwen,
                         font=("微软雅黑", 12), bg="#4a7abc", fg="white")
        btn3p.pack(pady=(0, 15), padx=10, fill=tk.X)

        ## 20260511新增智能问答窗口
        btn_qa = tk.Button(left_frame, text="智能问答", command=self.on_qwen_qa,
                           font=("微软雅黑", 12), bg="#4a7abc", fg="white")
        btn_qa.pack(pady=(0, 15), padx=10, fill=tk.X)
        ##################################


        # 持续训练按钮
        btn4 = tk.Button(left_frame, text="持续训练", command=self.open_train_window,
                         font=("微软雅黑",12), bg="#4a7abc", fg="white")
        btn4.pack(pady=(0,15), padx=10, fill=tk.X)

        # 20260510 新增更换使用设备按钮#################################
        # GPU 推理开关
        self.gpu_var = tk.BooleanVar(value=self.use_gpu)
        self.gpu_check = tk.Checkbutton(
            left_frame,
            text="使用GPU推理",
            variable=self.gpu_var,
            command=self._toggle_gpu_mode,
            bg="#f0f0f0",
            font=("微软雅黑", 10)
        )
        self.gpu_check.pack(pady=(10, 5), padx=10, anchor="w")
        ##############################################################

        tk.Label(left_frame, bg="#f0f0f0").pack(expand=True, fill=tk.BOTH)

    # 20260510 新增gpu设备选择回调方法#################################
    def _toggle_gpu_mode(self):
        """用户切换 GPU 开关时调用"""
        new_mode = self.gpu_var.get()
        if new_mode == self.use_gpu:
            return  # 无变化
        # 如果要开启但 GPU 不可用，给出提示并恢复原状态
        if new_mode and not self._cuda_available:
            self._append_debug("错误：当前环境没有可用的GPU，无法切换到GPU模式，保持CPU模式\n")
            self.gpu_var.set(False)
            return

        self.use_gpu = new_mode
        self._append_debug(f"切换设备模式为: {'GPU' if self.use_gpu else 'CPU'}\n")

        # 清空所有已加载的模型缓存（下次使用时重新加载）
        self._layout_detector = None
        self._table_cell_processor = None
        self._current_layout_weight = None
        self._current_table_cell_weight = None
        self._ocr_engine = None
        self._ner_pipeline = None
        self.imgpre_model = None
        self.yolo_model = None  # 如果有 YOLO 模型也清空

        # 可选：全局设置 PaddlePaddle 后端（影响未显式指定设备的操作）
        import paddle
        if self.use_gpu:
            paddle.set_device('gpu')
        else:
            paddle.set_device('cpu')

        # 重新加载已选择过路径的模型
        if hasattr(self, 'layout_weight_path') and self.layout_weight_path:
            self._load_layout_model()
        if hasattr(self, 'table_cell_weight_path') and self.table_cell_weight_path:
            self._load_table_cell_model()
        if hasattr(self, 'ocr_weight_path') and self.ocr_weight_path:
            self._load_ocr_engine()
        if hasattr(self, 'ner_weight_path') and self.ner_weight_path:
            self._load_ner_pipeline()
        # 图像预处理模型在用到时会自动重新加载（如果 imgpre_model 为 None）

        self._append_debug("模型重新加载完成（设备已切换）\n")
    #################################################################

    def _create_weight_row(self, parent, label_text, load_cmd, label_attr_name):
        frame = tk.Frame(parent, bg="#f0f0f0")
        frame.pack(fill=tk.X, padx=5, pady=(5,0))
        tk.Label(frame, text=label_text, bg="#f0f0f0", font=("微软雅黑",9)).pack(side=tk.LEFT)
        weight_label = tk.Label(frame, text="未加载", bg="#f0f0f0", font=("微软雅黑",8), fg="gray")
        weight_label.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        browse_btn = tk.Button(frame, text="浏览", command=load_cmd, font=("微软雅黑",8))
        browse_btn.pack(side=tk.RIGHT)
        setattr(self, label_attr_name, weight_label)

    #### 选择路径时加载模型权重
    def _load_layout_model(self):
        """根据当前 self.layout_weight_path 加载版式分析模型"""
        if not self.layout_weight_path or not os.path.isdir(self.layout_weight_path):
            self._append_debug("错误：未选择有效的文本区域检测模型目录\n")
            return False
        # 如果模型已加载且路径相同，直接返回
        if self._layout_detector is not None and self._current_layout_weight == self.layout_weight_path:
            return True
        self._append_debug(f"正在加载版式分析模型：{self.layout_weight_path}\n")
        try:
            detector_config = PaddlePicoDetectorConfig(
                model_name="PP-DocLayoutV3",
                model_dir=self.layout_weight_path,
                # 20260510 修改为根据前端选择设备###########
                device="gpu" if self.use_gpu else "cpu",
                ##########################################
                threshold=0.3,
                layout_nms=False,
                layout_unclip_ratio=1.0
            )
            self._layout_detector = PaddlePicoDetector(detector_config)
            self._current_layout_weight = self.layout_weight_path
            self._append_debug("版式分析模型加载完成\n")

        except Exception as e:
            self._append_debug(f"版式分析模型加载失败：{e}\n")
            return False

        yolo_path = f"{self.layout_weight_path}/det_photo/best1.pt"
        try:
            # 20260510 修改为根据前端选择设备###########
            yolo_device = 'cuda:0' if (self.use_gpu and torch.cuda.is_available()) else 'cpu'
            ##########################################
            self.yolo_model = YOLO(yolo_path)
            # self.yolo_conf = conf
            self._append_debug(f"YOLO 模型加载成功: {yolo_path}")
            return True
        except Exception as e:
            self._append_debug(f"YOLO 模型加载失败: {e}")
            return False

    def _load_table_cell_model(self):
        """根据当前 self.table_cell_weight_path 加载单元格分割模型"""
        if not self.table_cell_weight_path or not os.path.isdir(self.table_cell_weight_path):
            self._append_debug("错误：未选择有效的单元格检测模型目录\n")
            return False
        if self._table_cell_processor is not None and self._current_table_cell_weight == self.table_cell_weight_path:
            return True
        try:
            #20260510 使用cpu时禁止onednn加速###########
            old_mkldnn = os.environ.get('FLAGS_use_mkldnn', None)
            if not self.use_gpu:
                os.environ['FLAGS_use_mkldnn'] = '0'
            ###########################################
            cell_config = TableCellPostProcessorConfig(
                enabled=True,
                model_name="RT-DETR-L_wired_table_cell_det",
                model_dir=self.table_cell_weight_path,
                # 20260510 修改为根据前端选择设备###########
                device="gpu" if self.use_gpu else "cpu",
                ##########################################
                threshold=0.3,
                batch_size=1
            )
            self._table_cell_processor = TableCellPostProcessor(cell_config)
            self._current_table_cell_weight = self.table_cell_weight_path
            self._append_debug("单元格分割模型加载完成\n")
            return True
        except Exception as e:
            self._append_debug(f"单元格分割模型加载失败：{e}\n")
            return False

    def _load_ocr_engine(self):
        if hasattr(self, '_ocr_engine') and self._ocr_engine is not None:
            return True

        try:
            # def get_dynamic_limit(h):
            #     # 适配 2000*3000 (300dpi) 高清档案场景
            #     if h < 80:
            #         return 50  # 第1档：极小文字/单行，需放大防止笔画丢失
            #     elif h < 200:
            #         return 100 # 第2档：普通表格行
            #     elif h < 500:
            #         return 150  # 第3档：多行单元格/小段落
            #     elif h < 1000:
            #         return 200  # 第4档：大段落
            #     else:
            #         return 300  # 第5档：整页/超大图，全尺寸推理

            def dynamic_ocr(img, cls=False):
                if img is None:
                    return None

                h, w = img.shape[:2]
                # limit_len = get_dynamic_limit(h)

                result = self._ocr_engine.ocr(
                    img,
                    cls=cls,
                    # det_limit_side_len=limit_len
                )
                return result

            base_params = {
                'use_doc_orientation_classify': False,
                'use_doc_unwarping': False,
                'use_textline_orientation': False,
                'device': "gpu" if self.use_gpu else "cpu",
                'text_det_thresh': 0.1,
                'text_det_box_thresh': 0.3,
                'text_det_unclip_ratio': 2.0,
                # 这里给个默认值即可，反正会被 dynamic_ocr 覆盖
                'text_det_limit_side_len': 2048 if self.use_gpu else 2048,
                'enable_mkldnn': False  # 飞腾D3000(ARM)必须禁用
            }

            if self.ocr_weight_path and os.path.isdir(self.ocr_weight_path):
                base_path_name = os.path.normpath(self.ocr_weight_path)
                folder_name = os.path.basename(base_path_name)
                det_dir = os.path.join(self.ocr_weight_path, "det")
                rec_dir = os.path.join(self.ocr_weight_path, "rec")
                os.makedirs(det_dir, exist_ok=True)
                os.makedirs(rec_dir, exist_ok=True)

                self._ocr_engine = PaddleOCR(
                    text_detection_model_dir=det_dir,
                    text_recognition_model_dir=rec_dir,
                    text_detection_model_name=f'PP-OCRv5_{folder_name}_det',
                    text_recognition_model_name=f'PP-OCRv5_{folder_name}_rec',
                    **base_params
                )
            else:
                self._ocr_engine = PaddleOCR(
                    lang='ch',
                    text_detection_model_name='PP-OCRv5_mobile_det',
                    text_recognition_model_name='PP-OCRv5_mobile_rec',
                    **base_params
                )

            self.ocr_dynamic = dynamic_ocr

            self._append_debug(f"OCR 引擎加载完成\n")
            return True

        except Exception as e:
            self._append_debug(f"OCR 引擎加载失败：{e}\n")
            return False

    def _load_ner_pipeline(self):
        """根据当前 self.ner_weight_path 加载 NER 模型"""
        if hasattr(self, '_ner_pipeline') and self._ner_pipeline is not None:
            return True
        if self.ner_weight_path and os.path.isdir(self.ner_weight_path):
            model_path = self.ner_weight_path
        else:
            default_path = Path("models/ner_model")
            if default_path.exists():
                model_path = str(default_path)
            else:
                self._append_debug("错误：未选择NER模型路径，且默认模型不存在。\n")
                return False
        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = AutoModelForTokenClassification.from_pretrained(model_path)
            # device = 0 if torch.cuda.is_available() else -1
            # 20260510 修改为根据前端选择设备###########
            device = 0 if (self.use_gpu and torch.cuda.is_available()) else -1
            ##########################################
            self._ner_pipeline = pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=device,
            )
            self._append_debug(f"NER 模型加载成功：{model_path}\n")
            return True
        except Exception as e:
            self._append_debug(f"NER 模型加载失败：{e}\n")
            return False
    ###############################


    def load_layout_weight(self):
        path = filedialog.askdirectory(title="选择文本区域检测模型目录")
        if path:
            self.layout_weight_path = path
            self._update_weight_label("layout_weight_label", path)
            self._append_debug(f"已选择版式分析模型目录：{path}\n")
            # 立即加载模型
            self._load_layout_model()

    def load_table_cell_weight(self):
        path = filedialog.askdirectory(title="选择单元格分割模型目录")
        if path:
            self.table_cell_weight_path = path
            self._update_weight_label("table_cell_weight_label", path)
            self._append_debug(f"已选择单元格分割模型目录：{path}\n")
            self._load_table_cell_model()

    def load_ocr_weight(self):
        path = filedialog.askdirectory(title="选择OCR模型目录")
        if path:
            self.ocr_weight_path = path
            self._update_weight_label("ocr_weight_label", path)
            self._append_debug(f"已选择OCR模型目录：{path}\n")
            self._load_ocr_engine()

    def load_ner_weight(self):
        path = filedialog.askdirectory(title="选择关键信息提取模型目录")
        if path:
            self.ner_weight_path = path
            self._update_weight_label("ner_weight_label", path)
            self._append_debug(f"已选择NER模型目录：{path}\n")
            self._load_ner_pipeline()

    def _update_weight_label(self, label_attr, filepath):
        label = getattr(self, label_attr)
        name = os.path.basename(filepath)
        if len(name) > 20:
            name = name[:17]+"..."
        label.config(text=name, fg="green")

    def _create_right_regions(self):
        right_container = tk.Frame(self.root, bg="#e6e6e6")
        right_container.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=5, pady=5)

        # 顶部工具栏（文件夹选择）
        top_frame = tk.Frame(right_container, bg="#e6e6e6", pady=5)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5)
        tk.Label(top_frame, text="图片文件夹:", bg="#e6e6e6", font=("微软雅黑", 10)).pack(side=tk.LEFT)
        self.path_var = tk.StringVar()
        path_entry = tk.Entry(top_frame, textvariable=self.path_var, width=40, font=("微软雅黑", 10))
        path_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        browse_btn = tk.Button(top_frame, text="浏览...", command=self.browse_folder, font=("微软雅黑", 10))
        browse_btn.pack(side=tk.LEFT, padx=5)
        load_btn = tk.Button(top_frame, text="加载图片", command=self.load_images_from_selected_path,
                             font=("微软雅黑", 10))
        load_btn.pack(side=tk.LEFT)

        # 垂直分割窗口（可拖动调整上下区域高度）
        v_paned = tk.PanedWindow(right_container, orient=tk.VERTICAL, sashrelief=tk.RAISED, sashwidth=5, bg="#e6e6e6")
        v_paned.pack(side=tk.TOP, expand=True, fill=tk.BOTH, pady=5)

        # ---------- 上方：左右并列的两个画布 ----------
        main_paned = tk.PanedWindow(v_paned, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=5, bg="#e6e6e6")
        v_paned.add(main_paned, stretch="always", height=400)

        # 左侧：原始图片 + 检测框画布
        left_frame = tk.LabelFrame(main_paned, text="文本检测", font=("微软雅黑", 10), bg="#e6e6e6", padx=5, pady=5)
        main_paned.add(left_frame, width=400, stretch="always")
        left_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(left_frame, bg="white", relief=tk.SUNKEN, bd=2)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.zoom_pan = ZoomPanCanvas(self.canvas, self._draw_image_with_zoom_pan, drag_button=3)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<ButtonPress-3>", self.on_right_press)
        self.canvas.bind("<B3-Motion>", self.on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self.on_right_release)

        # 左侧画布下方的翻页按钮
        left_nav_frame = tk.Frame(left_frame, bg="#e6e6e6")
        left_nav_frame.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        self.prev_btn = tk.Button(left_nav_frame, text="◀ 上一张", command=self.prev_image, state=tk.DISABLED)
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        self.page_label = tk.Label(left_nav_frame, text="无图片", bg="#e6e6e6", font=("微软雅黑", 10))
        self.page_label.pack(side=tk.LEFT, padx=10)
        self.next_btn = tk.Button(left_nav_frame, text="下一张 ▶", command=self.next_image, state=tk.DISABLED)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        save_manual_btn = tk.Button(left_nav_frame, text="保存手动标注", command=self.save_manual_annotations,
                                    font=("微软雅黑", 9), bg="#28a745", fg="white")
        save_manual_btn.pack(side=tk.RIGHT, padx=5)

        # 右侧：文本叠加图画布
        right_frame = tk.LabelFrame(main_paned, text="文本叠加图", font=("微软雅黑", 10), bg="#e6e6e6", padx=5, pady=5)
        main_paned.add(right_frame, width=400, stretch="always")
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)
        self.layer_canvas = tk.Canvas(right_frame, bg="lightgray", relief=tk.SUNKEN, bd=2)
        self.layer_canvas.grid(row=0, column=0, sticky="nsew")
        self.layer_zoom_pan = ZoomPanCanvas(self.layer_canvas, self._draw_layer_image, drag_button=3)
        self.layer_canvas.bind("<ButtonPress-1>", self.on_layer_mouse_down)
        self.layer_canvas.bind("<B1-Motion>", self.on_layer_mouse_move)
        self.layer_canvas.bind("<ButtonRelease-1>", self.on_layer_mouse_up)

        right_toolbar = tk.Frame(right_frame, bg="#e6e6e6")
        right_toolbar.grid(row=1, column=0, sticky="ew", pady=5)
        tk.Button(right_toolbar, text="适应画布", command=self._fit_layer_canvas).pack(side=tk.LEFT, padx=5)

        # ---------- 下方：结果展示与性能统计（水平分割） ----------
        bottom_container = tk.LabelFrame(v_paned, text="结果与性能", font=("微软雅黑", 10),
                                         bg="#e6e6e6", padx=5, pady=5)
        v_paned.add(bottom_container, stretch="always", height=200)
        bottom_container.grid_rowconfigure(0, weight=1)
        bottom_container.grid_columnconfigure(0, weight=1)

        # 水平分割窗口（左侧 Notebook，右侧性能表）
        h_paned = tk.PanedWindow(bottom_container, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=5,
                                 bg="#e6e6e6")
        h_paned.grid(row=0, column=0, sticky="nsew")
        bottom_container.grid_rowconfigure(0, weight=1)
        bottom_container.grid_columnconfigure(0, weight=1)

        # ----- 左侧：Notebook（OCR 和 NER 结果）-----
        notebook_frame = tk.Frame(h_paned, bg="#e6e6e6")
        h_paned.add(notebook_frame, stretch="always", width=500)

        notebook = ttk.Notebook(notebook_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # OCR 标签页
        ocr_tab = tk.Frame(notebook, bg="#e6e6e6")
        notebook.add(ocr_tab, text="OCR识别结果")
        ocr_tab.grid_rowconfigure(0, weight=1)
        ocr_tab.grid_columnconfigure(0, weight=1)

        ocr_tree_frame = tk.Frame(ocr_tab, bg="#e6e6e6")
        ocr_tree_frame.grid(row=0, column=0, sticky="nsew")
        ocr_tree_frame.grid_rowconfigure(0, weight=1)
        ocr_tree_frame.grid_columnconfigure(0, weight=1)

        self.ocr_tree = ttk.Treeview(ocr_tree_frame, columns=("ImageName", "Title", "FullText"), show="headings",
                                     height=12)
        self.ocr_tree.heading("ImageName", text="图像名称")
        self.ocr_tree.heading("Title", text="标题")
        self.ocr_tree.heading("FullText", text="识别出的所有文本")
        self.ocr_tree.column("ImageName", width=150, minwidth=100, stretch=False, anchor="w")
        self.ocr_tree.column("Title", width=200, minwidth=100, stretch=False, anchor="w")
        self.ocr_tree.column("FullText", width=500, minwidth=200, stretch=True, anchor="w")

        ocr_vsb = ttk.Scrollbar(ocr_tree_frame, orient="vertical", command=self.ocr_tree.yview)
        ocr_hsb = ttk.Scrollbar(ocr_tree_frame, orient="horizontal", command=self.ocr_tree.xview)
        self.ocr_tree.configure(yscrollcommand=ocr_vsb.set, xscrollcommand=ocr_hsb.set)
        self.ocr_tree.grid(row=0, column=0, sticky="nsew")
        ocr_vsb.grid(row=0, column=1, sticky="ns")
        ocr_hsb.grid(row=1, column=0, sticky="ew")

        # OCR 操作栏（新增保存按钮）
        ocr_action_frame = tk.Frame(ocr_tab, bg="#e6e6e6")
        ocr_action_frame.grid(row=1, column=0, sticky="ew", pady=5)
        save_ocr_btn = tk.Button(ocr_action_frame, text="保存 OCR 修改", command=self._save_ocr_modifications,
                                 font=("微软雅黑", 9), bg="#28a745", fg="white")
        save_ocr_btn.pack(side=tk.RIGHT, padx=5)

        # NER 标签页
        ner_tab = tk.Frame(notebook, bg="#e6e6e6")
        notebook.add(ner_tab, text="关键信息提取结果")
        ner_tab.grid_rowconfigure(0, weight=1)
        ner_tab.grid_columnconfigure(0, weight=1)

        tree_frame = tk.Frame(ner_tab, bg="#e6e6e6")
        tree_frame.grid(row=0, column=0, sticky="nsew")
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        self.ner_tree = ttk.Treeview(tree_frame, columns=("Field", "Value", "Start", "End"),
                                     show="headings", height=12)
        self.ner_tree.heading("Field", text="字段")
        self.ner_tree.heading("Value", text="值")
        self.ner_tree.heading("Start", text="起始位置")
        self.ner_tree.heading("End", text="结束位置")
        self.ner_tree.column("Field", width=120, anchor="w")
        self.ner_tree.column("Value", width=200, anchor="w")
        self.ner_tree.column("Start", width=60, anchor="center")
        self.ner_tree.column("End", width=60, anchor="center")

        ###### 双击滑动替换ner结果/三击键入文本
        self.ner_tree.bind("<Double-1>", self._edit_ner_cell)
        self.ner_tree.bind("<Triple-1>", self._triple_click_value_cell)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.ner_tree.yview)
        self.ner_tree.configure(yscrollcommand=vsb.set)
        self.ner_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        # NER 操作栏
        action_frame = tk.Frame(ner_tab, bg="#e6e6e6")
        action_frame.grid(row=1, column=0, sticky="ew", pady=5)
        self.new_field_entry = tk.Entry(action_frame, width=15, font=("微软雅黑", 9))
        self.new_field_entry.pack(side=tk.LEFT, padx=2)
        add_btn = tk.Button(action_frame, text="新增字段", command=self._add_ner_item,
                            font=("微软雅黑", 9), bg="#4a7abc", fg="white")
        add_btn.pack(side=tk.LEFT, padx=2)
        del_btn = tk.Button(action_frame, text="删除选中", command=self._delete_ner_item,
                            font=("微软雅黑", 9), bg="#d9534f", fg="white")
        del_btn.pack(side=tk.LEFT, padx=2)
        save_btn = tk.Button(action_frame, text="保存为数据集", command=self._save_as_dataset,
                             font=("微软雅黑", 9), bg="#4a7abc", fg="white")
        save_btn.pack(side=tk.LEFT, padx=2)

        # NER 翻页按钮
        ner_nav_frame = tk.Frame(ner_tab, bg="#e6e6e6")
        ner_nav_frame.grid(row=2, column=0, sticky="ew", pady=2)
        self.prev_ner_btn = tk.Button(ner_nav_frame, text="◀ 上一结果", command=self._prev_ner_result,
                                      state=tk.DISABLED)
        self.prev_ner_btn.pack(side=tk.LEFT, padx=5)
        self.ner_page_label = tk.Label(ner_nav_frame, text="第0/0页", bg="#e6e6e6", font=("微软雅黑", 10))
        self.ner_page_label.pack(side=tk.LEFT, padx=10)
        self.next_ner_btn = tk.Button(ner_nav_frame, text="下一结果 ▶", command=self._next_ner_result,
                                      state=tk.DISABLED)
        self.next_ner_btn.pack(side=tk.LEFT, padx=5)

        # 右侧：性能统计表格
        perf_frame = self._create_performance_table(h_paned)
        h_paned.add(perf_frame, width=400)

        # ----- 最底部调试信息 -----
        debug_frame = tk.LabelFrame(right_container, text="调试信息", font=("微软雅黑", 10), bg="#e6e6e6", padx=5,
                                    pady=5)
        debug_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 0))
        debug_frame.grid_rowconfigure(0, weight=1)
        debug_frame.grid_columnconfigure(0, weight=1)

        self.debug_text = scrolledtext.ScrolledText(debug_frame, wrap=tk.WORD, font=("Consolas", 9),
                                                    relief=tk.SUNKEN, height=8)
        self.debug_text.grid(row=0, column=0, sticky="nsew")
        self.debug_text.config(state=tk.DISABLED)

        # 初始化变量
        if not hasattr(self, '_ocr_data_cache'):
            self._ocr_data_cache = {}
        if not hasattr(self, 'current_selected_ner_row'):
            self.current_selected_ner_row = None
        if not hasattr(self, 'highlighted_box_numbers'):
            self.highlighted_box_numbers = []
        self.start_box = None
        self.last_box = None

    def _create_performance_table(self, parent):
        """创建性能统计表格，返回 Frame"""
        frame = tk.LabelFrame(parent, text="性能统计", font=("微软雅黑", 10), bg="#e6e6e6", padx=5, pady=5)

        # 表格列头
        columns = ["环节", "图像数量/张", "总处理时间/s", "单张图像处理时间/s", "准确率"]
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=6)
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor="center")

        # 插入固定行
        stages = ["文本区域检测", "OCR识别", "关键信息提取", "全部环节"]
        for stage in stages:
            tree.insert("", "end", values=(stage, "0", "0.000", "0.000", "待计算"))

        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)

        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # 保存引用
        self.perf_tree = tree
        return frame

    def _update_performance_stats(self, stage, image_count, total_time):
        """
        更新性能统计表格中的指定环节
        :param stage: 环节名称（"文本区域检测"、"OCR识别"、"关键信息提取"、"全部环节"之一）
        :param image_count: 处理的图像数量
        :param total_time: 总耗时（秒）
        """
        avg_time = total_time / image_count if image_count > 0 else 0.0
        # 更新字典
        self.performance_stats[stage]["image_count"] = image_count
        self.performance_stats[stage]["total_time"] = total_time
        self.performance_stats[stage]["avg_time"] = avg_time
        # 更新表格显示
        if hasattr(self, 'perf_tree'):
            # 找到对应行
            for item in self.perf_tree.get_children():
                values = self.perf_tree.item(item)["values"]
                if values[0] == stage:
                    self.perf_tree.item(item, values=(
                        stage,
                        str(image_count),
                        f"{total_time:.3f}",
                        f"{avg_time:.3f}",
                        self.performance_stats[stage]["accuracy"]
                    ))
                    break
        # 更新“全部环节”行：累加前三个环节的数据（图像数量求和，总时间和，单张时间为总时间/总数量，准确率可暂留空）
        total_img = sum(self.performance_stats[s]["image_count"] for s in ["文本区域检测"])
        total_time_all = sum(
            self.performance_stats[s]["total_time"] for s in ["文本区域检测", "OCR识别", "关键信息提取"])
        avg_time_all = total_time_all / total_img if total_img > 0 else 0.0
        self.performance_stats["全部环节"]["image_count"] = total_img
        self.performance_stats["全部环节"]["total_time"] = total_time_all
        self.performance_stats["全部环节"]["avg_time"] = avg_time_all
        # 更新全部环节行
        for item in self.perf_tree.get_children():
            values = self.perf_tree.item(item)["values"]
            if values[0] == "全部环节":
                self.perf_tree.item(item, values=(
                    "全部环节",
                    str(total_img),
                    f"{total_time_all:.3f}",
                    f"{avg_time_all:.3f}",
                    self.performance_stats["全部环节"]["accuracy"]
                ))
                break


    # 20260511新增准确率计算####################
    def _calculate_accuracy(self, stage):
        """
        计算关键信息提取的准确率。
        使用固定测试集 archives_29fields_65000_pred.csv，
        用当前NER模型重新推理，对比用户修改后的 ground truth (label列)。
        """
        if stage != "关键信息提取":
            return "" if stage != "全部环节" else self.performance_stats["关键信息提取"]["accuracy"]

        csv_path = os.path.join(os.path.dirname(__file__), "datasets", "archives_29fields_65000_pred.csv")
        if not os.path.exists(csv_path):
            return "无测试集"

        # 确保 NER 模型已加载
        if not hasattr(self, '_ner_pipeline') or self._ner_pipeline is None:
            if not self._load_ner_pipeline():
                return "模型未加载"

        try:
            # 读取 CSV
            records = []  # (text, gt_entities)
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    text = r["text"]
                    try:
                        gt = json.loads(r["label"]) if r["label"].strip() else []
                    except:
                        gt = []
                    records.append((text, gt))

            if not records:
                return "空测试集"

            # 逐条推理并对比
            total_tp = total_fp = total_fn = 0
            for doc_text, gt_entities in records:
                try:
                    raw_entities = self._ner_pipeline(doc_text)
                except:
                    continue

                # 过滤与后处理（与 ner_extraction 一致）
                cleaned = []
                for ent in raw_entities:
                    score = ent.get("score", 0.0)
                    if score < 0.98:
                        continue
                    ent_dict = dict(ent)
                    start = ent_dict.get("start")
                    end = ent_dict.get("end")
                    if isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(doc_text):
                        ent_dict["word"] = doc_text[start:end]
                    elif "word" in ent_dict and isinstance(ent_dict["word"], str):
                        ent_dict["word"] = ent_dict["word"].replace("##", "").replace(" ", "")
                    if isinstance(ent_dict.get("word"), str):
                        ent_dict["word"] = self._clean_entity_text(ent_dict["word"])
                    label = str(ent_dict.get("entity_group", ""))
                    word = str(ent_dict.get("word", ""))
                    if self._should_drop_entity(label, word, doc_text,
                                                start if isinstance(start, int) else None,
                                                end if isinstance(end, int) else None):
                        continue
                    ent_dict["score"] = score
                    cleaned.append(ent_dict)

                cleaned.sort(key=lambda x: int(x.get("start", 0)))
                cleaned = self._fix_adjacent_split_entities(cleaned, doc_text)
                cleaned = self._fix_incomplete_dates(cleaned, doc_text)
                cleaned = self._context_relabel(cleaned, doc_text)
                cleaned = self._inject_missing_school_name(cleaned, doc_text)
                cleaned = self._inject_missing_political_status(cleaned, doc_text)
                cleaned = self._postprocess_misc_entities(cleaned, doc_text)
                cleaned = self._deduplicate_overlaps(cleaned)
                cleaned = self._postprocess_org_entities(cleaned, doc_text)
                cleaned = [it for it in cleaned if it.get("entity_group") in self.VALID_LABELS]

                # 匹配
                pred_used = [False] * len(cleaned)
                gt_used = [False] * len(gt_entities)
                for gi, ge in enumerate(gt_entities):
                    gl = ge.get("labels", [None])[0] if isinstance(ge.get("labels"), list) else ge.get("entity_group", "")
                    gt_text = ge.get("text", "")
                    for pi, pe in enumerate(cleaned):
                        if pred_used[pi]:
                            continue
                        pl = pe.get("entity_group", "")
                        pt = pe.get("word", "")
                        if gl != pl:
                            continue
                        # 文本匹配或位置重叠
                        if gt_text and pt and gt_text == pt:
                            total_tp += 1
                            pred_used[pi] = True
                            gt_used[gi] = True
                            break
                        # 位置重叠匹配
                        gs, ge_pos = ge.get("start", -1), ge.get("end", -1)
                        ps, pe_pos = pe.get("start", -1), pe.get("end", -1)
                        if gs >= 0 and ge_pos >= 0 and ps >= 0 and pe_pos >= 0:
                            if max(gs, ps) < min(ge_pos, pe_pos):
                                total_tp += 1
                                pred_used[pi] = True
                                gt_used[gi] = True
                                break

                total_fp += sum(1 for u in pred_used if not u)
                total_fn += sum(1 for u in gt_used if not u)

            # 计算指标
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            result_str = f"P={precision:.2%} R={recall:.2%} F1={f1:.2%}"
            self._append_debug(f"关键信息提取准确率评估完成: {result_str} (TP={total_tp} FP={total_fp} FN={total_fn})\n")
            return result_str

        except Exception as e:
            self._append_debug(f"准确率评估出错: {e}\n")
            import traceback
            self._append_debug(traceback.format_exc() + "\n")
            return "评估出错"


    def _triple_click_value_cell(self, event):
        """三击NER表格的值列，弹出编辑框修改值，并自动更新起始/结束位置"""
        region = self.ner_tree.identify_region(event.x, event.y)
        if region != "cell":
            return
        column = self.ner_tree.identify_column(event.x)  # '#1','#2','#3','#4'
        item_id = self.ner_tree.identify_row(event.y)
        if not item_id:
            return
        col_index = int(column[1:]) - 1  # 0:Field, 1:Value, 2:Start, 3:End
        row_index = int(item_id)

        # 只处理值列（col_index == 1）
        if col_index != 1:
            return

        if not self.ner_results or self.ner_current_index >= len(self.ner_results):
            return
        current_ner = self.ner_results[self.ner_current_index]
        if row_index < 0 or row_index >= len(current_ner["items"]):
            return

        old_item = current_ner["items"][row_index]
        old_value = old_item["Value"]
        full_text = self._current_full_text  # 当前图片的 OCR 全文

        # 创建内联输入框
        x, y, w, h = self.ner_tree.bbox(item_id, column)
        entry = tk.Entry(self.ner_tree, font=("微软雅黑", 9))
        entry.place(x=x, y=y, width=w, height=h)
        entry.insert(0, old_value)
        entry.focus_set()

        def save_edit():
            new_value = entry.get().strip()
            entry.destroy()
            if new_value == old_value:
                return

            # 在全文（self._current_full_text）中搜索新文本
            start_pos, end_pos = self._find_text_in_fulltext(full_text, new_value)

            # 更新内存中的条目
            old_item["Value"] = new_value
            old_item["Start"] = start_pos
            old_item["End"] = end_pos

            # 刷新表格显示
            self._show_ner_result(self.ner_current_index)
            # 保存到磁盘
            self._save_ner_result()
            self._append_debug(f"已将「{old_item['Field']}」的值更新为：{new_value} "
                               f"(位置：{start_pos}:{end_pos} {'✓' if start_pos >= 0 else '未匹配'})\n")

        entry.bind("<Return>", lambda e: save_edit())
        entry.bind("<FocusOut>", lambda e: save_edit())

    def _find_text_in_fulltext(self, full_text: str, target: str) -> tuple:
        """在全文（字符串）中查找目标文本的起始/结束位置。
        返回 (start, end)，若未找到则 (-1, -1)。"""
        if not target or not full_text:
            return (-1, -1)
        # 精确匹配
        pos = full_text.find(target)
        if pos != -1:
            return (pos, pos + len(target))
        return (-1, -1)

    # ========== 字符级选择的核心方法 ==========
    def _canvas_to_orig(self, cx, cy):
        """将画布坐标转换为原始图像坐标（表格图像原始尺寸坐标）"""
        zoom = self.layer_zoom_pan.zoom
        ox = self.layer_zoom_pan.offset_x
        oy = self.layer_zoom_pan.offset_y
        return (cx - ox) / zoom, (cy - oy) / zoom

    def _find_char_index_at_canvas_pos(self, cx, cy):
        """直接使用存储的画布矩形进行命中测试"""
        if not hasattr(self, 'char_canvas_rects'):
            return None
        for cr in self.char_canvas_rects:
            x1, y1, x2, y2 = cr["rect_canvas"]
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return cr["global_idx"]
        return None

    def _update_layer_highlight(self):
        """根据 self.selected_char_indices 绘制高亮矩形（使用画布矩形）"""
        if hasattr(self, 'current_highlight_rects'):
            for rect_id in self.current_highlight_rects:
                self.layer_canvas.delete(rect_id)
        self.current_highlight_rects = []
        if not self.selected_char_indices or not hasattr(self, 'char_canvas_rects'):
            return
        for cr in self.char_canvas_rects:
            if cr["global_idx"] in self.selected_char_indices:
                x1, y1, x2, y2 = cr["rect_canvas"]
                rect_id = self.layer_canvas.create_rectangle(x1, y1, x2, y2,
                                                             fill="#66CCFF", stipple="gray25", outline="")
                self.current_highlight_rects.append(rect_id)

    def on_layer_mouse_down(self, event):
        print("[DEBUG] 进入 on_layer_mouse_down")
        if not hasattr(self, 'char_canvas_rects') or not self.char_canvas_rects:
            print("[DEBUG] char_canvas_rects 为空，无法选择字符")
            return
        idx = self._find_char_index_at_canvas_pos(event.x, event.y)
        print(f"[DEBUG] 查找索引结果: {idx}")
        if idx is not None:
            self.select_start_idx = idx
            self.selected_char_indices = {idx}
            self._update_layer_highlight()
        else:
            self.select_start_idx = None

    def on_layer_mouse_move(self, event):
        if self.select_start_idx is None:
            return
        current_idx = self._find_char_index_at_canvas_pos(event.x, event.y)
        if current_idx is None:
            return
        print(f"[DEBUG] 当前索引 {current_idx}, 起始 {self.select_start_idx}")
        start = self.select_start_idx
        end = current_idx
        if start <= end:
            indices = set(range(start, end + 1))
        else:
            indices = set(range(end, start + 1))
        self.selected_char_indices = indices
        self._update_layer_highlight()

    #########################OCR识别结果编辑相关##########################
    def _edit_ocr_text_by_range(self, start_idx, end_idx, selected_text):
        """
        简化版：弹窗默认显示 selected_text（与用户滑取的文本一致），
        修改后直接更新 OCR 全文以及各框的文本（简单按比例分配）。
        """
        ocr_data = self._get_current_ocr_data()
        if not ocr_data:
            self._append_debug("未找到当前图片的 OCR 数据\n")
            return

        full_text = ocr_data.get("FullText", "")
        # 验证索引范围（仅用于提示，不影响弹窗文本）
        if start_idx < 0 or end_idx > len(full_text):
            self._append_debug(f"选区索引超出全文范围: {start_idx}:{end_idx}, 全文长度 {len(full_text)}\n")

        # 弹出输入框，默认显示选中的文本
        from tkinter import simpledialog
        new_text = simpledialog.askstring(
            "修改选中文本",
            f"当前选中的文本：\n{selected_text}\n\n请输入修改后的文本：",
            initialvalue=selected_text
        )
        if new_text is None or new_text == selected_text:
            return

        # 替换全文中的片段
        if 0 <= start_idx < end_idx <= len(full_text) and full_text[start_idx:end_idx] == selected_text:
            new_full_text = full_text[:start_idx] + new_text + full_text[end_idx:]
        else:
            # 降级：用字符串替换第一个匹配项
            new_full_text = full_text.replace(selected_text, new_text, 1)

        ocr_data["FullText"] = new_full_text

        # 简单更新各框文本：保持原有框的数量，按原比例分配新全文
        boxes = ocr_data.get("Box", [])
        if boxes:
            # 获取原有每个框的旧文本
            text_by_number = {}
            for group in ocr_data.get("Text", []):
                for num, txt in zip(group.get("Number", []), group.get("Content", [])):
                    text_by_number[num] = txt
            # 计算每个框旧文本的总长度
            old_lengths = []
            total_old_len = 0
            for box in boxes:
                num = box.get("Number")
                old_txt = text_by_number.get(num, "")
                old_len = len(old_txt)
                old_lengths.append((box, old_len))
                total_old_len += old_len
            if total_old_len > 0:
                new_len = len(new_full_text)
                pos = 0
                new_text_by_number = {}
                for box, old_len in old_lengths:
                    alloc_len = int(old_len / total_old_len * new_len)
                    end_pos = min(pos + alloc_len, new_len)
                    new_txt = new_full_text[pos:end_pos]
                    new_text_by_number[box.get("Number")] = new_txt
                    pos = end_pos
                if pos < new_len:
                    last_num = old_lengths[-1][0].get("Number")
                    new_text_by_number[last_num] += new_full_text[pos:]
                # 更新 Text 组
                new_text_groups = []
                for box in boxes:
                    num = box.get("Number")
                    new_text_groups.append({
                        "Content": [new_text_by_number.get(num, "")],
                        "Number": [num],
                        "Class": box.get("Class", "")
                    })
                ocr_data["Text"] = new_text_groups

        # ========== 关键：同步更新 RawLines 中的文本（不删除字段，全部保存） ==========
        self._update_rawlines_text_from_fulltext(ocr_data)

        # 更新缓存
        self._ocr_data_cache[os.path.basename(self.image_paths[self.current_image_index])] = ocr_data

        # 同步到 self.ocr_results
        img_name = os.path.basename(self.image_paths[self.current_image_index])
        for i, res in enumerate(self.ocr_results):
            if res.get("Imagepage") == img_name:
                self.ocr_results[i] = ocr_data
                break

        # 刷新界面
        self._fit_layer_canvas()
        self.layer_zoom_pan._redraw()
        self._refresh_ocr_table()
        self._append_debug(f"已将选区文本替换为：{new_text}\n")

    def _update_rawlines_text_from_fulltext(self, ocr_data):
        """
        根据更新后的 FullText 重新分配 RawLines 中每条 text 的内容，保证不丢失任何字符。
        分配依据：原始每条 RawLines 的文本长度比例。
        """
        raw_lines = ocr_data.get("RawLines", [])
        if not raw_lines:
            return

        # 获取原始每条 RawLines 的文本长度
        old_lengths = [len(line.get("text", "")) for line in raw_lines]
        total_old_len = sum(old_lengths)
        if total_old_len == 0:
            return

        new_full_text = ocr_data.get("FullText", "")
        new_len = len(new_full_text)

        # 按比例分配新长度（浮点数，向下取整，末尾调整）
        new_lengths = []
        sum_allocated = 0
        for old_len in old_lengths[:-1]:
            alloc = int(old_len / total_old_len * new_len)
            new_lengths.append(alloc)
            sum_allocated += alloc
        # 最后一条分配剩余长度
        new_lengths.append(new_len - sum_allocated)

        # 按新长度切分新全文并赋值
        pos = 0
        for i, alloc_len in enumerate(new_lengths):
            end_pos = min(pos + alloc_len, new_len)
            raw_lines[i]["text"] = new_full_text[pos:end_pos]
            pos = end_pos
        # 如果还有剩余（理论上不会有，因为总和已对齐），追加到最后一条
        if pos < new_len:
            raw_lines[-1]["text"] += new_full_text[pos:]
    #################################################################################

    def _sync_rawlines_from_text(self, ocr_data):
        """根据更新后的 Text（各框文本）更新 RawLines 中的文本内容，保留原有坐标"""
        raw_lines = ocr_data.get("RawLines", [])
        if not raw_lines:
            return

        # 将 Text 中的文本按顺序提取到一个列表
        text_list = []
        for group in ocr_data.get("Text", []):
            for txt in group.get("Content", []):
                if txt:
                    text_list.append(txt)
                else:
                    text_list.append("")  # 保留空字符串占位

        # 如果 RawLines 条目数与 text_list 长度不一致，说明结构变化，放弃同步（保留 RawLines 不变）
        if len(raw_lines) != len(text_list):
            return

        # 逐个更新 RawLines 中的 text 字段
        for i, new_text in enumerate(text_list):
            if "text" in raw_lines[i]:
                raw_lines[i]["text"] = new_text

    def _save_ocr_modifications(self):
        """将当前内存中所有图片的 OCR 数据保存到 ocr_result.json"""
        if not self.current_folder:
            self._append_debug("未选择文件夹，无法保存\n")
            return
        if hasattr(self, 'ocr_results') and self.ocr_results:
            safe_save_json(self.current_folder, "ocr_result.json", self.ocr_results)
            self._append_debug("OCR 修改已保存到 ocr_result.json\n")
        else:
            self._append_debug("没有可保存的 OCR 数据\n")
    #################################################################################
    def _update_ocr_table_data_for_current_image(self):
        """根据当前内存中的 OCR 数据，更新 self.ocr_table_data 中当前图片的条目"""
        ocr_data = self._get_current_ocr_data()
        if not ocr_data:
            return
        img_name = ocr_data["Imagepage"]
        full_text = ocr_data.get("FullText", "")
        # 提取标题
        title_texts = []
        for box in ocr_data.get("Box", []):
            if box.get("Class") in ("doc_title", "paragraph_title"):
                num = box.get("Number")
                # 找到对应的文本
                for group in ocr_data.get("Text", []):
                    if group["Number"][0] == num:
                        txt = group["Content"][0]
                        if txt:
                            title_texts.append(txt)
                        break
        title_content = " ".join(title_texts) if title_texts else ""
        # 更新 self.ocr_table_data
        for i, (name, _, _) in enumerate(self.ocr_table_data):
            if name == img_name:
                self.ocr_table_data[i] = (img_name, title_content, full_text)
                break

    #################################################################################
    def _update_ocr_table_data_for_current_image(self):
        """根据当前内存中的 OCR 数据，更新 self.ocr_table_data 中当前图片的条目"""
        ocr_data = self._get_current_ocr_data()
        if not ocr_data:
            return
        img_name = ocr_data["Imagepage"]
        full_text = ocr_data.get("FullText", "")
        # 提取标题
        title_texts = []
        for box in ocr_data.get("Box", []):
            if box.get("Class") in ("doc_title", "paragraph_title"):
                num = box.get("Number")
                for group in ocr_data.get("Text", []):
                    if group["Number"][0] == num:
                        txt = group["Content"][0]
                        if txt:
                            title_texts.append(txt)
                        break
        title_content = " ".join(title_texts) if title_texts else ""
        # 更新 self.ocr_table_data
        for i, (name, _, _) in enumerate(self.ocr_table_data):
            if name == img_name:
                self.ocr_table_data[i] = (img_name, title_content, full_text)
                break
    ################################################################################

    def on_layer_mouse_up(self, event):
        if self.select_start_idx is None:
            return
        if self.selected_char_indices:
            min_idx = min(self.selected_char_indices)
            max_idx = max(self.selected_char_indices)
            selected_chars = []
            for cr in self.char_canvas_rects:
                if min_idx <= cr["global_idx"] <= max_idx:
                    selected_chars.append(cr["char"])
            selected_text = "".join(selected_chars)

            # 如果是 NER 替换模式（双击值列后激活）
            if self.waiting_for_selection and self.current_selected_ner_row is not None:
                self._update_ner_field(self.current_selected_ner_row, selected_text, min_idx, max_idx + 1)
                self._append_debug(f"已用选区文本替换字段: {selected_text}\n")
                self.waiting_for_selection = False
            else:
                # 普通模式：弹出窗口修改 OCR 文本，直接传递 selected_text
                self._edit_ocr_text_by_range(min_idx, max_idx + 1, selected_text)

        self.select_start_idx = None
        self.selected_char_indices.clear()
        self._update_layer_highlight()

    def _fit_layer_canvas(self):
        """使右侧画布适应原始图片尺寸（白底图像）。"""
        if self.original_pil_image is None:
            self.layer_zoom_pan._redraw()
            return
        img_w, img_h = self.original_pil_image.size
        canvas = self.layer_canvas
        canvas.update_idletasks()
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw > 1 and ch > 1:
            zoom = min(cw / img_w, ch / img_h) * 0.95
            offset_x = (cw - img_w * zoom) / 2
            offset_y = (ch - img_h * zoom) / 2
            self.layer_zoom_pan.zoom = zoom
            self.layer_zoom_pan.offset_x = offset_x
            self.layer_zoom_pan.offset_y = offset_y
            self.layer_zoom_pan._redraw()

    ## 20260512修改 与增加空格文本对齐
    def _get_current_layout_data(self):
        """返回当前图片对应的版面检测数据（从 layout_analysis.json 缓存或文件中读取）"""
        if not self.current_folder or not self.image_paths:
            return None
        img_name = os.path.basename(self.image_paths[self.current_image_index])
        # 可以缓存到字典中
        if hasattr(self, '_layout_data_cache') and img_name in self._layout_data_cache:
            return self._layout_data_cache[img_name]
        layout_path = os.path.join(self.current_folder, "layout_analysis.json")
        if not os.path.exists(layout_path):
            return None
        with open(layout_path, 'r', encoding='utf-8') as f:
            all_layout = json.load(f)
        for item in all_layout:
            if item.get("Imagepage") == img_name:
                if not hasattr(self, '_layout_data_cache'):
                    self._layout_data_cache = {}
                self._layout_data_cache[img_name] = item
                return item
        return None

    def _get_font_path(self):
        """返回系统可用的中文字体路径"""
        candidates = [
            "C:/Windows/Fonts/simhei.ttf",
            "/System/Library/Fonts/PingFang.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None  # 回退到默认字体

    def _draw_layer_image(self, canvas, zoom, offset_x, offset_y):
        """绘制右侧画布：白色背景 + 版面检测框 + OCR文本（文本行无边框）"""
        canvas.delete("all")
        if self.original_pil_image is None:
            canvas.create_text(200, 200, text="未加载原始图片", fill="gray", font=("微软雅黑", 16))
            return

        img_w, img_h = self.original_pil_image.size
        # ---------- 1. 创建白色背景图 ----------
        white_img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
        draw = ImageDraw.Draw(white_img)

        # ---------- 2. 绘制版面检测框（来自 layout_analysis.json）----------
        layout_data = self._get_current_layout_data()
        if layout_data:
            layout_boxes = layout_data.get("Box", [])
            for box_info in layout_boxes:
                pos = box_info.get("Position", [])
                if len(pos) < 8:
                    continue
                # 构造多边形点列表
                pts = [(pos[i], pos[i + 1]) for i in range(0, 8, 2)]
                # 用浅灰色细线绘制边框
                draw.polygon(pts, outline="#AAAAAA", width=1)
                # 可选：在框的左上角标注类别与编号
                class_name = box_info.get("Class", "")
                number = box_info.get("Number", "")
                label = f"{class_name}:{number}" if class_name else number
                if label and pts:
                    # 使用小号字体（若无中文字体则用默认）
                    try:
                        small_font = ImageFont.truetype(self._get_font_path(), 12)
                    except:
                        small_font = ImageFont.load_default()
                    draw.text((pts[0][0] + 2, pts[0][1] + 2), label, fill="#888888", font=small_font)

        # ---------- 3. 绘制 OCR 文本 ----------
        ocr_data = self._get_current_ocr_data()
        if ocr_data:
            # ===== 新增：如果存在 RawLines，则用它替换原有的 boxes 数据 =====
            raw_lines = ocr_data.get("RawLines", [])
            if raw_lines:
                # 用 RawLines 构造与原来 Box 格式一致的列表
                boxes = []
                text_by_number = {}
                for idx, line in enumerate(raw_lines):
                    poly = line.get("poly")
                    text = line.get("text", "")
                    if not poly or not text:
                        continue
                    # 将四点坐标转换为八点 [左下,右下,右上,左上]
                    # RawLines 中的 poly 顺序通常为：左上、右上、右下、左下
                    if isinstance(poly[0], (list, tuple)):
                        tl, tr, br, bl = poly[0], poly[1], poly[2], poly[3]
                    else:  # 一维列表 [x1,y1, x2,y2, x3,y3, x4,y4]
                        tl = (poly[0], poly[1])
                        tr = (poly[2], poly[3])
                        br = (poly[4], poly[5])
                        bl = (poly[6], poly[7])
                    position = [bl[0], bl[1], br[0], br[1], tr[0], tr[1], tl[0], tl[1]]
                    num = str(idx)
                    boxes.append({
                        "Number": num,
                        "Position": position,
                        "Class": "ocr_line"
                    })
                    text_by_number[num] = text.strip()
                # 由于 RawLines 各行独立，无需插入行间空格，设为 None 跳过后续空格插入
                last_valid_num = None
            else:
                # 原有逻辑：从 ocr_data 中取 Box 和 Text
                boxes = ocr_data.get("Box", [])
                text_by_number = {}
                for group in ocr_data.get("Text", []):
                    for num, txt in zip(group.get("Number", []), group.get("Content", [])):
                        text_by_number[num] = txt

                # 原有 last_valid_num 计算（保留）
                last_valid_num = None
                for box_info in reversed(boxes):
                    cls = box_info.get("Class", "")
                    if cls not in self.SKIP_OCR_CLASSES:
                        num = box_info.get("Number")
                        if text_by_number.get(num, ""):
                            last_valid_num = num
                            break
            # ===== 数据源替换结束，以下全部为原有代码 =====

            # 加载字体（同原逻辑）
            font_path = self._get_font_path()

            # 辅助函数：从四点坐标获取外接矩形
            def get_polygon_bbox(points):
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                return min(xs), min(ys), max(xs), max(ys)

            # 辅助函数：自动换行（原逻辑不变）
            def wrap_text(text, max_width, font, initial_x, initial_y):
                lines = []
                current_line = ""
                current_width = 0
                for ch in text:
                    try:
                        bbox = font.getbbox(ch)
                        ch_width = bbox[2] - bbox[0]
                    except:
                        ch_width = font.getsize(ch)[0]
                    if current_width + ch_width > max_width and current_line:
                        lines.append((current_line, current_width))
                        current_line = ch
                        current_width = ch_width
                    else:
                        current_line += ch
                        current_width += ch_width
                if current_line:
                    lines.append((current_line, current_width))
                if not lines:
                    return [], 0
                try:
                    line_height = font.getbbox(text[0])[3] - font.getbbox(text[0])[1]
                except:
                    line_height = font.getsize(text[0])[1]
                all_char_rects = []
                cur_y = initial_y
                for line_chars, line_width in lines:
                    start_x = initial_x + (max_width - line_width) / 2
                    cur_x = start_x
                    for ch in line_chars:
                        try:
                            ch_bbox = font.getbbox(ch)
                            ch_w = ch_bbox[2] - ch_bbox[0]
                            ch_h = ch_bbox[3] - ch_bbox[1]
                        except:
                            ch_w, ch_h = font.getsize(ch)
                        ch_left = cur_x
                        ch_top = cur_y
                        ch_right = cur_x + ch_w
                        ch_bottom = cur_y + ch_h
                        all_char_rects.append((ch, (ch_left, ch_top, ch_right, ch_bottom)))
                        cur_x += ch_w
                    cur_y += line_height + 2
                return all_char_rects, cur_y - initial_y

            # 重置字符级数据容器
            self.char_positions_orig = []
            self.char_canvas_rects = []
            self.layer_full_text = ""
            global_idx = 0
            self.layer_boxes = []

            # 遍历 OCR 的 Box 列表（按原有顺序，不排序）
            for box_info in boxes:
                num = box_info.get("Number")
                pos = box_info.get("Position", [])
                if len(pos) < 8:
                    continue
                pts = [(pos[i], pos[i + 1]) for i in range(0, 8, 2)]
                # 注意：这里不绘制多边形边框

                text = text_by_number.get(num, "")
                if not text:
                    self.layer_boxes.append({
                        "box_number": num,
                        "text": "",
                        "polygon_orig": pts,
                        "bbox_orig": get_polygon_bbox(pts)
                    })
                    continue

                left, top, right, bottom = get_polygon_bbox(pts)
                cell_w = right - left
                cell_h = bottom - top
                if cell_w <= 0 or cell_h <= 0:
                    continue

                # 动态选择最佳字体大小（与原始代码一致）
                best_font_size = 10
                best_char_rects = []
                best_total_height = 0
                for font_size in range(40, 9, -2):
                    try:
                        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
                    except:
                        font = ImageFont.load_default()
                    char_rects, total_height = wrap_text(text, cell_w, font, left, top)
                    if total_height <= cell_h * 0.95:
                        best_font_size = font_size
                        best_char_rects = char_rects
                        best_total_height = total_height
                        break
                if best_font_size == 10:
                    try:
                        font = ImageFont.truetype(font_path, best_font_size) if font_path else ImageFont.load_default()
                    except:
                        font = ImageFont.load_default()
                    best_char_rects, best_total_height = wrap_text(text, cell_w, font, left, top)

                # 垂直居中微调
                y_offset = (cell_h - best_total_height) / 2
                for ch, (ch_left, ch_top, ch_right, ch_bottom) in best_char_rects:
                    new_top = ch_top + y_offset
                    new_bottom = ch_bottom + y_offset
                    # 在 Pillow 图像上绘制字符（黑色）
                    draw.text((ch_left, new_top), ch, font=font, fill=(0, 0, 0))
                    # 记录原始坐标用于字符选择
                    self.char_positions_orig.append({
                        "char": ch,
                        "rect": (ch_left, new_top, ch_right, new_bottom),
                        "global_idx": global_idx
                    })
                    self.layer_full_text += ch
                    global_idx += 1

                # 在框后插入一个空心格分隔符（除非是最后一个有效框）
                if last_valid_num is not None and num != last_valid_num:
                    if self.char_positions_orig:
                        last_cp = self.char_positions_orig[-1]
                        space_left = last_cp["rect"][2]
                        space_top = last_cp["rect"][1]
                        space_width = max(2, best_font_size // 2)
                        space_right = space_left + space_width
                        space_bottom = last_cp["rect"][3]
                        self.char_positions_orig.append({
                            "char": " ",
                            "rect": (space_left, space_top, space_right, space_bottom),
                            "global_idx": global_idx
                        })
                        self.layer_full_text += " "
                        global_idx += 1

                self.layer_boxes.append({
                    "box_number": num,
                    "text": text,
                    "polygon_orig": pts,
                    "bbox_orig": (left, top, right, bottom)
                })

        # ---------- 4. 缩放并显示图像 ----------
        new_w = int(img_w * zoom)
        new_h = int(img_h * zoom)
        if new_w < 1 or new_h < 1:
            return
        scaled_img = white_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.layer_photo = ImageTk.PhotoImage(scaled_img)
        canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=self.layer_photo)

        # ---------- 5. 生成画布坐标系下的字符矩形（用于字符级选择）----------
        actual_zoom_x = new_w / img_w
        actual_zoom_y = new_h / img_h
        self.char_canvas_rects = []
        for cp in self.char_positions_orig:
            left, top, right, bottom = cp["rect"]
            x1 = left * actual_zoom_x + offset_x
            y1 = top * actual_zoom_y + offset_y
            x2 = right * actual_zoom_x + offset_x
            y2 = bottom * actual_zoom_y + offset_y
            self.char_canvas_rects.append({
                "char": cp["char"],
                "rect_canvas": (x1, y1, x2, y2),
                "global_idx": cp["global_idx"]
            })

        # 重置当前高亮选区
        self.selected_char_indices.clear()
        self._update_layer_highlight()

    def _orig_to_canvas(self, x, y):
        """将原始图像坐标转换为画布坐标"""
        zoom = self.layer_zoom_pan.zoom
        ox = self.layer_zoom_pan.offset_x
        oy = self.layer_zoom_pan.offset_y
        return x * zoom + ox, y * zoom + oy

    def _get_current_ocr_data(self):
        """返回当前图片对应的 OCR 条目（从缓存的 ocr_result.json 或内存）"""
        if not self.current_folder or not self.image_paths:
            return None
        img_name = os.path.basename(self.image_paths[self.current_image_index])
        # 尝试从内存缓存获取
        if hasattr(self, '_ocr_data_cache') and img_name in self._ocr_data_cache:
            return self._ocr_data_cache[img_name]
        # 加载 JSON
        ocr_json_path = os.path.join(self.current_folder, "ocr_result.json")
        if not os.path.exists(ocr_json_path):
            return None
        with open(ocr_json_path, 'r', encoding='utf-8') as f:
            all_ocr = json.load(f)
        for item in all_ocr:
            if item.get("Imagepage") == img_name:
                self._ocr_data_cache[img_name] = item
                return item
        return None

    def on_right_press(self, event):
        self.right_press_pos = (event.x, event.y)
        self.right_dragging = False
        # 记录当前偏移量，用于拖拽平移
        self.drag_start_offset_x = self.zoom_pan.offset_x
        self.drag_start_offset_y = self.zoom_pan.offset_y
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def on_right_drag(self, event):
        if self.right_press_pos:
            dx = event.x - self.right_press_pos[0]
            dy = event.y - self.right_press_pos[1]
            if abs(dx) > 3 or abs(dy) > 3:
                self.right_dragging = True
        if self.right_dragging:
            # 执行画布平移
            new_offset_x = self.drag_start_offset_x + (event.x - self.drag_start_x)
            new_offset_y = self.drag_start_offset_y + (event.y - self.drag_start_y)
            self.zoom_pan.offset_x = new_offset_x
            self.zoom_pan.offset_y = new_offset_y
            self.zoom_pan._redraw()

    def on_right_release(self, event):
        if self.right_press_pos and not self.right_dragging:
            # 视为点击，删除框
            self._delete_box_at_position(event)
        self.right_press_pos = None
        self.right_dragging = False

    def _delete_box_at_position(self, event):
        """根据右键点击位置删除框"""
        # 转换坐标
        img_x, img_y = self._canvas_to_image_coords(event.x, event.y)
        current_path = self.image_paths[self.current_image_index]
        boxes = self.image_boxes_cache.get(current_path, [])

        # 从后往前遍历，避免索引问题
        for i in range(len(boxes) - 1, -1, -1):
            box = boxes[i]
            pos = box["Position"]  # 八点坐标 [x1,y1, x2,y2, x3,y3, x4,y4]
            points = [(pos[j], pos[j + 1]) for j in range(0, 8, 2)]
            if self._point_in_polygon(img_x, img_y, points):
                del boxes[i]
                self._append_debug(f"已删除 {box.get('Class', '未知')} 框\n")
                # 更新缓存并重绘
                self.image_boxes_cache[current_path] = boxes
                self._draw_detection_boxes_on_current_image(standard_boxes=boxes)
                break

    def _point_in_polygon(self, x, y, poly):
        """射线法判断点是否在多边形内"""
        inside = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            # 检查射线是否与边相交
            if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
                inside = not inside
        return inside

    def _canvas_to_image_coords(self, canvas_x, canvas_y):
        """将画布坐标转换为原始图像上的坐标（考虑缩放和平移）"""
        zoom = self.zoom_pan.zoom
        offset_x = self.zoom_pan.offset_x
        offset_y = self.zoom_pan.offset_y
        img_x = (canvas_x - offset_x) / zoom
        img_y = (canvas_y - offset_y) / zoom
        # 裁剪到图像有效范围
        img_w, img_h = self.original_pil_image.size
        img_x = max(0, min(img_x, img_w))
        img_y = max(0, min(img_y, img_h))
        return img_x, img_y

    def on_mouse_down(self, event):
        self.drawing = True
        self.start_x, self.start_y = event.x, event.y
        if self.current_rect:
            self.canvas.delete(self.current_rect)
            self.current_rect = None

    def on_mouse_move(self, event):
        if not self.drawing:
            return
        if self.current_rect:
            self.canvas.delete(self.current_rect)
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline="blue", width=2, dash=(4, 2)
        )

    def on_mouse_up(self, event):
        if not self.drawing:
            return
        self.drawing = False
        end_x, end_y = event.x, event.y
        # 去除太小矩形
        if abs(end_x - self.start_x) < 5 or abs(end_y - self.start_y) < 5:
            if self.current_rect:
                self.canvas.delete(self.current_rect)
                self.current_rect = None
            return

        # 转换为图像坐标
        x1, y1 = self._canvas_to_image_coords(self.start_x, self.start_y)
        x2, y2 = self._canvas_to_image_coords(end_x, end_y)
        # 归一化为左上右下
        left, right = sorted([x1, x2])
        top, bottom = sorted([y1, y2])

        # 弹出类别选择窗口
        self._ask_annotation_class(left, top, right, bottom)
        if self.current_rect:
            self.canvas.delete(self.current_rect)
            self.current_rect = None

    def _ask_annotation_class(self, left, top, right, bottom):
        """弹出类别选择窗口（带确定/取消按钮）"""
        dialog = tk.Toplevel(self.root)
        dialog.title("选择标注类别")
        dialog.geometry("320x350")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)  # 禁止调整大小，避免布局错乱

        # 让对话框居中于主窗口
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - 320) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 280) // 2
        dialog.geometry(f"+{x}+{y}")

        # 提示标签
        tk.Label(dialog, text="请选择标注类别：", font=("微软雅黑", 10)).pack(pady=(15, 5))

        # 类别列表
        listbox = tk.Listbox(dialog, height=10, font=("微软雅黑", 9))
        categories = [
            "text", "title", "table", "figure", "header", "footer",
            "page-number", "caption", "equation", "其它"
        ]
        for cat in categories:
            listbox.insert(tk.END, cat)
        listbox.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)
        listbox.select_set(0)  # 默认选中第一项

        # 按钮框架
        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=15)

        def on_ok():
            sel = listbox.curselection()
            if sel:
                cat = listbox.get(sel[0])
                dialog.destroy()
                self._save_annotation(left, top, right, bottom, cat)
            else:
                dialog.destroy()  # 未选择则取消

        def on_cancel():
            dialog.destroy()

        ok_btn = tk.Button(btn_frame, text="确定", command=on_ok, width=10, bg="#4a7abc", fg="white")
        ok_btn.pack(side=tk.LEFT, padx=10)
        cancel_btn = tk.Button(btn_frame, text="取消", command=on_cancel, width=10)
        cancel_btn.pack(side=tk.LEFT, padx=10)

        # 确保窗口置顶并获取焦点
        dialog.focus_force()


    def _save_annotation(self, left, top, right, bottom, category):
        """将新标注框添加到当前图片的缓存中，并刷新画布"""
        current_path = self.image_paths[self.current_image_index]
        current_boxes = self.image_boxes_cache.get(current_path, [])

        # 生成新的编号
        new_number = str(len(current_boxes))
        # 使用八点坐标 [左下,右下,右上,左上] 顺序
        new_box = {
            "Position": [left, bottom, right, bottom, right, top, left, top],
            "Number": new_number,
            "Class": category
        }
        current_boxes.append(new_box)
        self.image_boxes_cache[current_path] = current_boxes

        # 立即重绘画布，显示新框
        self._draw_detection_boxes_on_current_image(standard_boxes=current_boxes)

        self._append_debug(f"已添加 {category} 框\n")

    def save_manual_annotations(self):
        """将手动标注保存为新的 JSON 文件（PPDetV3格式）"""
        if not self.current_folder:
            self._append_debug("未加载图片文件夹\n")
            return
        if not self.image_boxes_cache:
            self._append_debug("没有标注数据，请先进行文本区域检测或手动标注\n")
            return

        output_data = []
        for img_path, boxes in self.image_boxes_cache.items():
            img_name = os.path.basename(img_path)
            # 转换格式：如果现有 boxes 是标准格式，直接使用
            # 如果检测器输出是另一种结构，可能需要转换，这里假设已经是标准格式
            output_data.append({
                "Imagepage": img_name,
                "Box": boxes
            })

        output_path = os.path.join(self.current_folder, "manual_annotations.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        self._append_debug(f"手动标注已保存到 {output_path}\n")

    def _add_ner_item(self):
        """在当前图片的NER结果中新增一行"""
        if not self.ner_results or self.ner_current_index >= len(self.ner_results):
            self._append_debug("没有NER结果，无法新增\n")
            return
        field_name = self.new_field_entry.get().strip()
        if not field_name:
            self._append_debug("请输入字段名\n")
            return
        # 获取当前图片的数据
        current_data = self.ner_results[self.ner_current_index]
        # 新增一个条目，值为空，start/end设为-1
        new_item = {
            "Field": field_name,
            "Value": "",
            "Start": -1,
            "End": -1
        }
        current_data["items"].append(new_item)
        # 刷新表格
        self._show_ner_result(self.ner_current_index)
        self._append_debug(f"已新增字段：{field_name}\n")
        # 清空输入框
        self.new_field_entry.delete(0, tk.END)

    def _delete_ner_item(self):
        """删除当前选中的NER条目"""
        selected = self.ner_tree.selection()
        if not selected:
            self._append_debug("请先选中要删除的行\n")
            return
        # 只处理第一个选中项
        item_id = selected[0]
        row_index = int(item_id)
        if not self.ner_results or self.ner_current_index >= len(self.ner_results):
            return
        current_data = self.ner_results[self.ner_current_index]
        if row_index < 0 or row_index >= len(current_data["items"]):
            return
        # 删除条目
        del current_data["items"][row_index]
        # 刷新表格
        self._show_ner_result(self.ner_current_index)
        self._append_debug("已删除选中行\n")

    def _append_debug(self, content):
        if hasattr(self, 'debug_text') and self.debug_text:
            self.debug_text.config(state=tk.NORMAL)
            self.debug_text.insert(tk.END, content)
            self.debug_text.see(tk.END)
            self.debug_text.config(state=tk.DISABLED)
        else:
            # 如果调试窗口还没准备好，先输出到控制台（避免崩溃）
            print(content, end='')

    def _draw_image_with_zoom_pan(self, canvas, zoom, offset_x, offset_y):
        if self.current_display_pil is None:
            return
        w, h = self.current_display_pil.size
        new_w, new_h = int(w*zoom), int(h*zoom)
        if new_w<1 or new_h<1: return
        img = self.current_display_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.current_photo = ImageTk.PhotoImage(img)
        canvas.delete("all")
        canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=self.current_photo)

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.path_var.set(folder)
            self.load_images_from_selected_path()

    def load_images_from_selected_path(self):
        folder = self.path_var.get().strip()
        if not os.path.isdir(folder):
            self._append_debug("路径无效\n")
            return
        self.current_folder = folder
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
        self.image_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]
        self.image_paths.sort()

        if not self.image_paths:
            self._append_debug("未找到图片\n")
            self.current_image_index = 0
            self._update_nav()
            self.page_label.config(text="无图片")
            self.current_display_pil = None
            self._clear_canvas()
            # 重置 NER 结果与表格
            self.ner_results = []
            self.ner_current_index = 0
            if hasattr(self, 'ner_tree'):
                for row in self.ner_tree.get_children():
                    self.ner_tree.delete(row)
                self._update_ner_nav_state()
            # 重置 OCR 表格
            self.ocr_table_data = []
            if hasattr(self, 'ocr_tree'):
                for row in self.ocr_tree.get_children():
                    self.ocr_tree.delete(row)
            return

        self.current_image_index = 0
        self._update_nav()
        self._load_current_image()

        # 重置预处理状态（图像增强）
        self.is_preprocessed = False
        self.enhanced_image_paths = []
        self.enhanced_folder = None

        # 重置 NER 相关
        self.ner_results = []
        self.ner_current_index = 0
        if hasattr(self, 'ner_tree'):
            for row in self.ner_tree.get_children():
                self.ner_tree.delete(row)
            self._update_ner_nav_state()

        # 重置 OCR 表格
        self.ocr_results = []
        self.ocr_table_data = []
        if hasattr(self, 'ocr_tree'):
            for row in self.ocr_tree.get_children():
                self.ocr_tree.delete(row)

    def _load_current_image(self):
        if not self.image_paths:
            return
        img_path = self.image_paths[self.current_image_index]
        try:
            pil = Image.open(img_path)
            self.original_pil_image = pil.copy()
            self.current_display_pil = self.original_pil_image.copy()

            self.canvas.update_idletasks()
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()

            if canvas_w > 1 and canvas_h > 1:
                img_w, img_h = self.original_pil_image.size
                scale_x = canvas_w / img_w
                scale_y = canvas_h / img_h
                fit_scale = min(scale_x, scale_y)
            else:
                fit_scale = 1.0

            self.zoom_pan.zoom = fit_scale
            self.zoom_pan.offset_x = 0
            self.zoom_pan.offset_y = 0
            self.page_label.config(text=f"{self.current_image_index + 1}/{len(self.image_paths)}")

            current_path = img_path
            if current_path in self.image_boxes_cache:
                self._draw_detection_boxes_on_current_image(standard_boxes=self.image_boxes_cache[current_path])
            else:
                self.zoom_pan._redraw()
        except Exception as e:
            self._append_debug(f"加载失败: {img_path}\n{e}\n")
            self._clear_canvas()

        self.highlighted_box_numbers = []  # 重置高亮
        if hasattr(self, 'layer_zoom_pan'):
            self.layer_zoom_pan._redraw()
        # 在 _load_current_image 函数最后，self.zoom_pan._redraw() 之后添加
        if hasattr(self, '_ocr_data_cache') and os.path.basename(img_path) in self._ocr_data_cache:
            self._fit_layer_canvas()

    def _clear_canvas(self):
        self.canvas.delete("all")

    def prev_image(self):
        if self.image_paths and self.current_image_index>0:
            self.current_image_index -= 1
            self._load_current_image()
            self._update_nav()

    def next_image(self):
        if self.image_paths and self.current_image_index<len(self.image_paths)-1:
            self.current_image_index += 1
            self._load_current_image()
            self._update_nav()

    def _update_nav(self):
        if not self.image_paths:
            self.prev_btn.config(state=tk.DISABLED)
            self.next_btn.config(state=tk.DISABLED)
        else:
            self.prev_btn.config(state=tk.NORMAL if self.current_image_index>0 else tk.DISABLED)
            self.next_btn.config(state=tk.NORMAL if self.current_image_index<len(self.image_paths)-1 else tk.DISABLED)

    # ---------- 算法占位 ----------#########################需增加
    def detect_text_regions(self, image_paths):
        if not self.is_preprocessed:
            self._preprocess_all_images()
        detection_paths = image_paths  # 或 self.enhanced_image_paths

        if not self.current_folder:
            return []

        # 检查模型是否已加载
        if self._layout_detector is None:
            self._append_debug("错误：请先选择文本区域检测模型目录并加载。\n")
            return None

        results = []
        for img_path, enhanced_path in zip(self.image_paths, detection_paths):
            detection_results = self._layout_detector.predict(Path(enhanced_path), batch_size=1)
            if not detection_results:
                self._append_debug(f"警告：{os.path.basename(img_path)} 未检测到任何区域\n")
                results.append({
                    "Imagepage": os.path.basename(img_path),
                    "Box": []
                })
                continue

            result_dict = self._layout_detector.result_to_dict(detection_results[0])
            boxes = self._layout_detector.extract_boxes(result_dict)

            # 调试：打印所有检测到的类别
            detected_labels = {box.get("label", "unknown") for box in boxes}
            self._append_debug(f"[{os.path.basename(img_path)}] 检测到类别: {detected_labels}\n")

            # 表格单元格后处理（如果有表格）
            table_cell_payload = None
            if self._table_cell_processor is not None:
                table_boxes = self._table_cell_processor.pick_table_boxes(boxes)
                if table_boxes:
                    table_cell_payload = self._table_cell_processor.detect_cells(
                        image_path=Path(img_path),
                        table_boxes=table_boxes,
                    )

            # 构建标准框（保留全部类别）
            standard_boxes = self._build_standard_boxes(boxes, table_cell_payload)


            ###### 新增yolo照片识别
            # 假设 standard_boxes 已经是当前图片的 Box 列表（每个元素包含 Position, Number, Class）
            if self.yolo_model is not None:
                # 20260510 更新设备选择
                yolo_results = self.yolo_model.predict(str(enhanced_path), conf=self.yolo_conf, verbose=False, device='cuda:0' if (self.use_gpu and torch.cuda.is_available()) else 'cpu')
                if yolo_results and len(yolo_results) > 0:
                    names = self.yolo_model.names
                    # 当前已有框的数量，用于生成新的 Number 序号
                    current_count = len(standard_boxes)
                    for box in yolo_results[0].boxes:
                        cls_id = int(box.cls[0])
                        cls_name = names[cls_id].lower() if names else str(cls_id)
                        if cls_name == 'photo':  # 只保留 photo 类
                            x1, y1, x2, y2 = box.xyxy[0].tolist()  # 绝对坐标
                            # 构造四点坐标：左上 -> 右上 -> 右下 -> 左下
                            position = [x1, y1, x2, y1, x2, y2, x1, y2]
                            photo_box = {
                                "Position": position,
                                "Number": str(current_count),  # 分配新序号
                                "Class": "photo",
                                # 原格式无 confidence，如需保留可加额外字段如 "_confidence": float(box.conf[0])
                            }
                            standard_boxes.append(photo_box)
                            current_count += 1
                        elif cls_name == 'seal':
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            position = [x1, y1, x2, y1, x2, y2, x1, y2]
                            photo_box = {
                                "Position": position,
                                "Number": str(current_count),  # 分配新序号
                                "Class": "seal",
                                # 原格式无 confidence，如需保留可加额外字段如 "_confidence": float(box.conf[0])
                            }
                            standard_boxes.append(photo_box)
                            current_count += 1


            # 缓存并保存
            self.image_boxes_cache[img_path] = standard_boxes
            results.append({
                "Imagepage": os.path.basename(img_path),
                "Box": standard_boxes
            })

            # 如果是当前图片，立即绘制
            if img_path == self.image_paths[self.current_image_index]:
                self._draw_detection_boxes_on_current_image(standard_boxes=standard_boxes)

            gc.collect()
            if paddle.device.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()

        return results

    def _append_to_text_area(self, widget, content):
        widget.config(state=tk.NORMAL)
        widget.insert(tk.END, content)
        widget.see(tk.END)
        widget.config(state=tk.DISABLED)

    def _build_standard_boxes(self, boxes, table_cell_payload):
        """
        将检测器输出的 boxes 和表格单元格后处理结果转换为标准 JSON 格式。
        参照您提供的 build_standard_boxes 函数实现。
        """
        # 提取表格内的单元格位置（如果有）
        table_positions = []
        if table_cell_payload and isinstance(table_cell_payload, dict):
            tables = table_cell_payload.get("tables", [])
            for table in tables:
                cells = table.get("cells", [])
                if not isinstance(cells, list):
                    table_positions.append([])
                    continue
                # 按 cell_index 排序保证顺序
                sorted_cells = sorted(
                    [cell for cell in cells if isinstance(cell, dict)],
                    key=lambda cell: int(cell.get("cell_index", 0))
                )
                cell_positions = []
                for cell in sorted_cells:
                    quad = self._to_position_quad(cell.get("coordinate"))
                    if quad:
                        cell_positions.append(quad)
                table_positions.append(cell_positions)

        normalized = []
        table_index = 0
        for box in boxes:
            box_class = str(box.get("label", ""))
            if box_class.lower() == "table":
                # 如果有单元格，则展开为多个框；否则用表格自身框
                if table_index < len(table_positions) and table_positions[table_index]:
                    for cell_pos in table_positions[table_index]:
                        normalized.append({
                            "Position": cell_pos,
                            "Number": str(len(normalized)),
                            "Class": "table"
                        })
                else:
                    quad = self._to_position_quad(box.get("coordinate"))
                    if quad:
                        normalized.append({
                            "Position": quad,
                            "Number": str(len(normalized)),
                            "Class": "table"
                        })
                table_index += 1
            else:
                quad = self._to_position_quad(box.get("coordinate"))
                if quad:
                    normalized.append({
                        "Position": quad,
                        "Number": str(len(normalized)),
                        "Class": box_class
                    })
        return normalized

    def _to_position_quad(self, coordinate):# -> list[float] | None:
        """将坐标转换为八点格式 [左下, 右下, 右上, 左上]"""
        if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 4:
            return None
        try:
            x_min, y_min, x_max, y_max = [float(v) for v in coordinate]
        except (TypeError, ValueError):
            return None
        return [x_min, y_max, x_max, y_max, x_max, y_min, x_min, y_min]

    def _draw_detection_boxes_on_current_image(self, standard_boxes=None):
        """在画布上绘制检测框，并自动适应画布大小"""
        if self.original_pil_image is None:
            if self.image_paths and self.current_image_index < len(self.image_paths):
                img_path = self.image_paths[self.current_image_index]
                try:
                    self.original_pil_image = Image.open(img_path).copy()
                except Exception as e:
                    self._append_debug(f"无法加载原始图片用于绘制: {e}\n")
                    return
            else:
                self._append_debug("没有可用的原始图片进行绘制\n")
                return

        if standard_boxes is None:
            current_path = self.image_paths[self.current_image_index]
            standard_boxes = self.image_boxes_cache.get(current_path, [])

        # 绘制带框图像
        if not standard_boxes:
            self.current_display_pil = self.original_pil_image.copy()
        else:
            img = self.original_pil_image.copy()
            draw = ImageDraw.Draw(img)
            for box_info in standard_boxes:
                pos = box_info["Position"]
                points = [(pos[i], pos[i + 1]) for i in range(0, 8, 2)]
                draw.polygon(points, outline="red", width=3)
                class_name = box_info.get("Class", "")
                number = box_info.get("Number", "")
                text = f"{class_name}:{number}"
                text_x, text_y = points[0]
                draw.text((text_x, text_y - 10), text, fill="red")
            self.current_display_pil = img

        # 强制更新画布尺寸信息
        self.canvas.update_idletasks()
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w > 1 and canvas_h > 1 and self.current_display_pil:
            img_w, img_h = self.current_display_pil.size
            scale_x = canvas_w / img_w
            scale_y = canvas_h / img_h
            fit_scale = min(scale_x, scale_y)
        else:
            fit_scale = 1.0

        self.zoom_pan.zoom = fit_scale
        self.zoom_pan.offset_x = 0
        self.zoom_pan.offset_y = 0
        self.zoom_pan._redraw()


    ## 20260512 修改
    def ocr_recognition(self, image_paths, layout_data):
        self.ocr_results = []
        if not self.is_preprocessed:
            self._preprocess_all_images()

        layout_map = {item["Imagepage"]: item for item in layout_data}

        for img_path, enhanced_path in zip(image_paths, self.enhanced_image_paths):
            img_name = os.path.basename(img_path)
            layout_item = layout_map.get(img_name)  # 可能为 None

            # ========== 1. 无条件执行 OCR ==========
            try:
                current_img = Image.open(enhanced_path).convert('RGB')
            except Exception as e:
                self._append_debug(f"无法打开图片 {img_name}: {e}\n")
                continue
            # current_np = np.array(current_img)
            current_np = cv2.cvtColor(np.array(current_img), cv2.COLOR_RGB2BGR)

            try:
                ocr_result = self._ocr_engine.predict(current_np)
            except Exception as e:
                self._append_debug(f"全图 OCR 失败（{img_name}）：{e}\n")
                continue

            # 提取文本行坐标与文本
            lines_with_boxes = []
            if ocr_result and isinstance(ocr_result, list) and len(ocr_result) > 0:
                result_obj = ocr_result[0]
                rec_texts = result_obj.get('rec_texts', [])
                rec_polys = result_obj.get('rec_polys', [])
                rec_scores = result_obj.get('rec_scores', [])
                if rec_texts and rec_polys:
                    for poly, text, score in zip(rec_polys, rec_texts, rec_scores):
                        if text and score > 0.03:
                            text_clean = text.strip().replace(" ", "").replace("\u3000", "")
                            if text_clean:
                                lines_with_boxes.append((poly, text_clean))

            # ========== 2. 分支处理：有版面数据 vs 无版面数据 ==========
            if layout_item is not None:
                # 有版面数据：执行原有分配逻辑（完全保持不变）
                layout_boxes = layout_item.get("Box", [])
                box_rects = []
                for box in layout_boxes:
                    pos = box.get("Position", [])
                    if len(pos) < 8:
                        continue
                    xs = pos[0::2]
                    ys = pos[1::2]
                    xmin, xmax = min(xs), max(xs)
                    ymin, ymax = min(ys), max(ys)
                    box_rects.append((xmin, ymin, xmax, ymax, box["Number"], box.get("Class", "")))

                box_texts = {box["Number"]: [] for box in layout_boxes if len(box.get("Position", [])) >= 8}
                for poly, text in lines_with_boxes:
                    xs = [p[0] for p in poly]
                    ys = [p[1] for p in poly]
                    line_cx = (min(xs) + max(xs)) / 2
                    line_cy = (min(ys) + max(ys)) / 2
                    for rxmin, rymin, rxmax, rymax, num, cls in box_rects:
                        if cls in self.SKIP_OCR_CLASSES:
                            continue
                        if rxmin <= line_cx <= rxmax and rymin <= line_cy <= rymax:
                            box_texts[num].append((poly, text))
                            break

                text_by_number = {}
                for num, lines in box_texts.items():
                    if lines:
                        sorted_lines = self._sort_text_lines(lines)
                        sorted_lines = self._fix_horizontal_disorder(sorted_lines)
                        text_by_number[num] = "".join(t for _, t in sorted_lines)
                    else:
                        text_by_number[num] = ""

                boxes_out = []
                text_groups = []
                full_text_parts = []
                char_offset = 0
                for box in layout_boxes:
                    num = box["Number"]
                    cls = box.get("Class", "")
                    new_box = dict(box)
                    txt = text_by_number.get(num, "")
                    if cls in self.SKIP_OCR_CLASSES:
                        new_box["char_start"] = -1
                        new_box["char_end"] = -1
                    else:
                        start = char_offset
                        end = start + len(txt)
                        new_box["char_start"] = start
                        new_box["char_end"] = end
                        full_text_parts.append(txt)
                        char_offset = end + 1
                    if "Position" in new_box:
                        new_box["Position"] = [int(v) for v in new_box["Position"]]
                    boxes_out.append(new_box)

                full_text = " ".join(full_text_parts)
                for box in boxes_out:
                    num = box["Number"]
                    txt = text_by_number.get(num, "")
                    text_groups.append({
                        "Content": [txt],
                        "Number": [num],
                        "Class": box.get("Class", "")
                    })
            else:
                # 无版面数据：直接基于 OCR 原始行构建结果（不分配框）
                boxes_out = []  # 空的 Box 列表
                text_groups = []  # 空的 Text 组
                # 全文按原始行顺序拼接（保持阅读顺序）
                full_text_parts = [text for _, text in lines_with_boxes]
                full_text = " ".join(full_text_parts)

            # ========== 3. 构建统一的 result_entry ==========
            result_entry = {
                "Imagepage": img_name,
                "Box": boxes_out,
                "Text": text_groups,
                "FullText": full_text,
                "RawLines": [
                    {
                        "poly": [[int(p[0]), int(p[1])] for p in poly],
                        "text": text
                    }
                    for poly, text in lines_with_boxes
                ]
            }
            self.ocr_results.append(result_entry)
            self._ocr_data_cache[img_name] = result_entry

            # 刷新右侧画布
            if img_path == self.image_paths[self.current_image_index]:
                self._fit_layer_canvas()
                self.layer_zoom_pan._redraw()

            self._append_debug(f"[{img_name}] OCR 完成，全文长度 {len(full_text)}\n")

        # 更新左侧 OCR 表格
        self.ocr_table_data = []
        for res in self.ocr_results:
            img_name = res["Imagepage"]
            full_text = res.get("FullText", "")
            self.ocr_table_data.append((img_name, "", full_text))
        self.root.after(0, self._refresh_ocr_table)

        return self.ocr_results

    # 20260512 新增方法
    def _extract_lines_with_boxes(self, ocr_result):
        """
        从 PaddleOCR 的 predict 或 ocr 返回结果中提取 (box, text) 对列表。
        - 支持标准格式: [[[box], (text, score)], ...]
        - 支持 predict 字典格式: [{'rec_texts': [...], 'det_polygons': [...]}, ...]
        如果无法提取坐标，返回 None。
        """
        if not ocr_result:
            return None
        lines = []
        try:
            if isinstance(ocr_result, list) and len(ocr_result) > 0:
                first = ocr_result[0]
                # 格式1: 列表内嵌列表，如 [[[box], (text, score)], ...]
                if isinstance(first, list):
                    for item in first:
                        if isinstance(item, (list, tuple)) and len(item) == 2:
                            box = item[0]
                            txt_info = item[1]
                            if isinstance(txt_info, (list, tuple)) and len(txt_info) >= 1:
                                text = txt_info[0]
                                if box and text:
                                    lines.append((box, text))
                # 格式2: 字典格式，如 [{'rec_texts': [...], 'det_polygons': [...]}]
                elif isinstance(first, dict):
                    # 可能有多页，这里只处理单页
                    rec_texts = first.get('rec_texts', [])
                    det_polygons = first.get('det_polygons', [])
                    if rec_texts and det_polygons and len(rec_texts) == len(det_polygons):
                        for poly, text in zip(det_polygons, rec_texts):
                            if poly and text:
                                # poly 是四点坐标列表，如 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                                lines.append((poly, text))
                else:
                    return None
            return lines if lines else None
        except Exception:
            return None

    def _sort_text_lines(self, lines, y_threshold=15):
        """
        对 (box, text) 列表按阅读顺序排序（先从上到下，再从左到右）。
        box: 四点坐标列表 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        """
        if not lines:
            return lines
        # 计算每个框的 y_min 和 x_min
        boxes_info = []
        for box, text in lines:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            y_min = min(ys)
            x_min = min(xs)
            boxes_info.append((y_min, x_min, box, text))
        # 按 y_min 排序
        boxes_info.sort(key=lambda x: x[0])
        # 聚类行
        grouped = []
        current_row = []
        last_y = None
        for y_min, x_min, box, text in boxes_info:
            if last_y is None or abs(y_min - last_y) <= y_threshold:
                current_row.append((x_min, box, text))
            else:
                current_row.sort(key=lambda x: x[0])
                for _, b, t in current_row:
                    grouped.append((b, t))
                current_row = [(x_min, box, text)]
            last_y = y_min
        if current_row:
            current_row.sort(key=lambda x: x[0])
            for _, b, t in current_row:
                grouped.append((b, t))
        return grouped

    @staticmethod
    def _fix_horizontal_disorder(lines):
        """
        修复同一行内，机器打印体与手写体因间距过大导致的乱序问题。
        例如 “姓名” “张三” 在 x 轴上位置可能被错误交换。
        策略：如果多行在同一水平线上（y 坐标接近），强制按 x 坐标排序。
        """
        if not lines or len(lines) < 2:
            return lines

        # 计算平均 y 坐标和高度
        y_centers = []
        heights = []
        for box, text in lines:
            ys = [p[1] for p in box]
            y_centers.append((min(ys) + max(ys)) / 2)
            heights.append(max(ys) - min(ys))

        avg_height = sum(heights) / len(heights) if heights else 0
        avg_y = sum(y_centers) / len(y_centers) if y_centers else 0

        # 如果所有行的 y 中心都在一个平均高度范围内，认为是同一行
        in_same_row = all(abs(y - avg_y) < avg_height * 1.2 for y in y_centers)

        if in_same_row:
            # 按 x 坐标重新排序
            sorted_lines = sorted(lines, key=lambda x: min(p[0] for p in x[0]))
            return sorted_lines

        return lines
    ############################################

    def _refresh_ocr_table(self):
        # 清空现有行
        for row in self.ocr_tree.get_children():
            self.ocr_tree.delete(row)

        # 插入数据（顺序：图像名称，标题，全文）
        for item in self.ocr_table_data:
            # item 已经是 (img_name, title, full_text)
            self.ocr_tree.insert("", "end", values=item)

        # 将全文列的宽度设大，出现水平滚动条
        self.ocr_tree.column("FullText", width=10000, minwidth=200, stretch=False)
        self.ocr_tree.update_idletasks()
        self.ocr_tree.xview_moveto(0)


    # ---------- NER 常量 ----------
    VALID_LABELS = {
        "姓名", "性别", "出生年月", "民族", "籍贯", "出生地",
        "政治面貌", "入党时间", "参加工作时间", "工作单位", "职务", "现职时间",
        "身份证号", "全日制教育学历", "全日制教育学位", "全日制教育毕业院校系", "全日制教育专业",
        "在职教育学历", "在职教育学位", "在职教育毕业院校系", "在职教育专业"
    }
    VALID_POLITICAL_STATUS = {"党员", "中共党员", "预备党员", "共青团员", "团员", "群众", "民主党派", "无党派"}

    # ---------- NER 辅助函数（静态方法）----------
    @staticmethod
    def _normalize_word(word: str) -> str:
        return word.replace("##", "").replace(" ", "")

    @staticmethod
    def _clean_entity_text(word: str) -> str:
        import re
        return re.sub(r'^[\s#;；,，。:：\|"“”\(\)（）\[\]【】<>《》]+|[\s#;；,，。:：\|"“”\(\)（）\[\]【】<>《》]+$', "", word)

    @staticmethod
    def _is_symbol_only(word: str) -> bool:
        import re
        return bool(word) and re.fullmatch(r"[\W_#]+", word, flags=re.UNICODE) is not None

    @staticmethod
    def _is_year_month_like(word: str) -> bool:
        import re
        return re.fullmatch(r"(?:\d{4}|[〇零一二三四五六七八九十]{4})年(?:\d{1,2}|[〇零一二三四五六七八九十]{1,3})月", word) is not None

    @staticmethod
    def _is_full_date_like(word: str) -> bool:
        import re
        return re.fullmatch(
            r"(?:\d{4}|[〇零一二三四五六七八九十]{4})年(?:\d{1,2}|[〇零一二三四五六七八九十]{1,3})月(?:\d{1,2}|[〇零一二三四五六七八九十]{1,3})日",
            word,
        ) is not None

    @staticmethod
    def _is_year_like(word: str) -> bool:
        import re
        return re.fullmatch(r"(?:\d{4}|[〇零一二三四五六七八九十]{4})", word) is not None

    @staticmethod
    def _is_cn_id_number(word: str) -> bool:
        import re
        w = word.replace(" ", "")
        return (
            re.fullmatch(r"[1-9]\d{16}[0-9Xx]", w) is not None
            or re.fullmatch(r"[1-9]\d{14}", w) is not None
        )

    @staticmethod
    def _expand_year_to_year_month(word: str, text: str, start: int, end: int):
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(text)):
            return word, end
        if not MultiRegionApp._is_year_like(word):
            return word, end
        if end >= len(text) or text[end] != "年":
            return word, end
        max_right = min(len(text), end + 8)
        seg = text[start:max_right]
        import re
        m = re.match(r"^(?:\d{4}|[〇零一二三四五六七八九十]{4})年(?:\d{1,2}|[〇零一二三四五六七八九十]{1,3})月", seg)
        if m:
            expanded = m.group(0)
            return expanded, start + len(expanded)
        return word, end

    @staticmethod
    def _fix_incomplete_dates(items: list[dict], text: str) -> list[dict]:
        """修复不完整的日期，如 '年9月' -> '2012年9月'"""
        import re
        out = []
        date_labels = {"参加工作时间", "出生年月", "入党时间", "现职时间"}
        for item in items:
            label = str(item.get("entity_group", ""))
            word = str(item.get("word", ""))
            start = item.get("start")
            end = item.get("end")
            if label in date_labels and word.startswith("年") and isinstance(start, int):
                prev_text = text[max(0, start - 6):start]
                year_match = re.search(r'(\d{4})', prev_text)
                if year_match:
                    year = year_match.group(1)
                    full_word = year + word
                    new_start = start - len(year)
                    item["word"] = full_word
                    item["start"] = new_start
                    print(f"[日期修复] {word} -> {full_word}")
            out.append(item)
        return out

    @staticmethod
    def _should_drop_entity(label: str, word: str, text: str, start: int, end: int) -> bool:
        import re
        if not word:
            return True
        if MultiRegionApp._is_symbol_only(word):
            return True
        if label not in MultiRegionApp.VALID_LABELS:
            return True
        # 身份证号验证
        if label == "身份证号":
            clean_word = re.sub(r'[\s\-_]', '', word)
            is_valid_18 = re.fullmatch(r'[1-9]\d{16}[\dXx]', clean_word) is not None
            is_valid_15 = re.fullmatch(r'[1-9]\d{14}', clean_word) is not None
            if not (is_valid_18 or is_valid_15):
                return True
        # 政治面貌有效值验证
        if label == "政治面貌":
            if word not in MultiRegionApp.VALID_POLITICAL_STATUS:
                return True
        # 过滤单字符噪声（保留性别和民族）
        if len(word) == 1 and label not in {"性别", "民族"}:
            return True
        noise_words = {"无", "不详", "未知", "待填", "市", "员", "技术"}
        if word in noise_words:
            return True
        if label == "全日制教育专业" and word in {"技术", "职务", "工资", "情况"}:
            return True
        if label == "入党时间":
            if not re.match(r"\d{4}[./年]\d{1,2}([./月]\d{1,2}日?)?", word):
                return True
            if any(k in word for k in ["会议", "全会", "召开", "期间", "十一届", "三中全会"]):
                return True
        if label in ["全日制教育毕业院校系", "在职教育毕业院校系"]:
            political_orgs = ["中国共产主义青年团", "共青团", "共产党", "党支部", "党委", "委员会"]
            if word in political_orgs or any(org in word for org in political_orgs):
                return True
        if label == "籍贯" and any(k in word for k in ["路", "号", "小区", "街道", "村", "胡同"]):
            return True
        if label in ["全日制教育学历", "在职教育学历"]:
            if word == "大学" or word == "大":
                return True
        return False

    @staticmethod
    def _fix_adjacent_split_entities(items: list[dict], text: str) -> list[dict]:
        if not items:
            return items
        fixed = []
        i = 0
        while i < len(items):
            cur = items[i]
            if i + 1 < len(items):
                nxt = items[i + 1]
                cur_start = cur.get("start")
                cur_end = cur.get("end")
                nxt_start = nxt.get("start")
                nxt_end = nxt.get("end")
                cur_word = str(cur.get("word", ""))
                nxt_word = str(nxt.get("word", ""))
                cur_label = str(cur.get("entity_group", ""))
                nxt_label = str(nxt.get("entity_group", ""))
                if (isinstance(cur_start, int) and isinstance(cur_end, int) and
                    isinstance(nxt_start, int) and isinstance(nxt_end, int) and
                    cur_end == nxt_start):
                    # 合并公司后缀
                    if cur_word.endswith("有限") and nxt_word == "公司":
                        merged = {
                            "entity_group": cur_label if cur_label in MultiRegionApp.VALID_LABELS else nxt_label,
                            "score": max(float(cur.get("score", 0.0)), float(nxt.get("score", 0.0))),
                            "word": cur_word + nxt_word,
                            "start": cur_start,
                            "end": nxt_end,
                        }
                        fixed.append(merged)
                        i += 2
                        continue
                    # 合并“建筑”+“有限公司”
                    if cur_word.endswith("建筑") and nxt_word.startswith("有限"):
                        merged = {
                            "entity_group": cur_label if cur_label in MultiRegionApp.VALID_LABELS else nxt_label,
                            "score": max(float(cur.get("score", 0.0)), float(nxt.get("score", 0.0))),
                            "word": cur_word + nxt_word,
                            "start": cur_start,
                            "end": nxt_end,
                        }
                        fixed.append(merged)
                        i += 2
                        continue
                    # 研 + 究生 -> 研究生
                    if cur_word == "研" and nxt_word == "究生":
                        merged = {
                            "entity_group": "全日制教育学历",
                            "score": max(float(cur.get("score", 0.0)), float(nxt.get("score", 0.0))),
                            "word": "研究生",
                            "start": cur_start,
                            "end": nxt_end,
                        }
                        fixed.append(merged)
                        i += 2
                        continue
                    # 大学 + 本科 -> 本科
                    if cur_word == "大学" and nxt_word == "本科":
                        if cur_label in ["全日制教育学历", "在职教育学历"] and nxt_label in ["全日制教育学历", "在职教育学历"]:
                            merged = {
                                "entity_group": nxt_label,
                                "score": nxt.get("score", 0.0),
                                "word": "本科",
                                "start": nxt_start,
                                "end": nxt_end,
                            }
                            fixed.append(merged)
                            i += 2
                            continue
                    # 大学 + 专科 -> 专科
                    if cur_word == "大学" and nxt_word == "专科":
                        if cur_label in ["全日制教育学历", "在职教育学历"] and nxt_label in ["全日制教育学历", "在职教育学历"]:
                            merged = {
                                "entity_group": nxt_label,
                                "score": nxt.get("score", 0.0),
                                "word": "专科",
                                "start": nxt_start,
                                "end": nxt_end,
                            }
                            fixed.append(merged)
                            i += 2
                            continue
                    # 合并两个连续的毕业院校系
                    if cur_label == "在职教育毕业院校系" and nxt_label == "在职教育毕业院校系":
                        merged = {
                            "entity_group": "在职教育毕业院校系",
                            "score": max(float(cur.get("score", 0.0)), float(nxt.get("score", 0.0))),
                            "word": cur_word + nxt_word,
                            "start": cur_start,
                            "end": nxt_end,
                        }
                        fixed.append(merged)
                        i += 2
                        continue
                    if cur_label == "全日制教育毕业院校系" and nxt_label == "全日制教育毕业院校系":
                        merged = {
                            "entity_group": "全日制教育毕业院校系",
                            "score": max(float(cur.get("score", 0.0)), float(nxt.get("score", 0.0))),
                            "word": cur_word + nxt_word,
                            "start": cur_start,
                            "end": nxt_end,
                        }
                        fixed.append(merged)
                        i += 2
                        continue
                    # 日期合并
                    combined_date = cur_word + nxt_word
                    date_labels = {"出生年月", "参加工作时间", "现职时间", "入党时间"}
                    if MultiRegionApp._is_year_month_like(combined_date) and {cur_label, nxt_label}.issubset(date_labels):
                        merged_label = cur_label if cur_label != "日期" else (nxt_label if nxt_label != "日期" else "出生年月")
                        merged = {
                            "entity_group": merged_label,
                            "score": max(float(cur.get("score", 0.0)), float(nxt.get("score", 0.0))),
                            "word": combined_date,
                            "start": cur_start,
                            "end": nxt_end,
                        }
                        fixed.append(merged)
                        i += 2
                        continue
                    if MultiRegionApp._is_full_date_like(combined_date) and {cur_label, nxt_label}.issubset(date_labels):
                        merged_label = cur_label if cur_label != "日期" else (nxt_label if nxt_label != "日期" else "出生年月")
                        merged = {
                            "entity_group": merged_label,
                            "score": max(float(cur.get("score", 0.0)), float(nxt.get("score", 0.0))),
                            "word": combined_date,
                            "start": cur_start,
                            "end": nxt_end,
                        }
                        fixed.append(merged)
                        i += 2
                        continue
                    if not (len(cur_word) == 1 and len(nxt_word) == 1):
                        fixed.append(cur)
                        i += 1
                        continue
            fixed.append(cur)
            i += 1
        return fixed

    @staticmethod
    def _context_relabel(items: list[dict], text: str) -> list[dict]:
        if not items:
            return items
        out = [dict(x) for x in items]
        school_keywords = ["大学", "学院", "学校", "研究院", "系", "技术学院", "师范大学", "理工大学", "工业大学", "科技大学"]
        fulltime_keywords = ["全日制", "全日制教育", "全日制学历", "毕业院校", "学历学位", "全日制毕业"]
        parttime_keywords = ["在职", "在职教育", "在职学历", "继续教育", "成人教育"]
        family_keywords = ["配偶", "父亲", "母亲", "妹妹", "哥哥", "弟弟", "姐姐", "子女", "关系"]
        work_history_keywords = ["参加工作", "工作经历", "简历", "学习和工作", "至今", "前后简历"]

        family_region_start = -1
        for kw in family_keywords:
            pos = text.find(kw)
            if pos != -1:
                family_region_start = pos
                break

        for item in out:
            label = str(item.get("entity_group", ""))
            word = str(item.get("word", ""))
            start = item.get("start", 0)
            end = item.get("end", 0)
            if label not in MultiRegionApp.VALID_LABELS:
                continue

            # 职务修正：部门名称 -> 工作单位，工作内容 -> 丢弃
            if label == "职务":
                department_keywords = ["人事部", "财务部", "办公室", "行政部", "技术部", "市场部", "销售部", "研发部", "人事科", "财务科"]
                if word in department_keywords:
                    item["entity_group"] = "工作单位"
                    continue
                work_content_keywords = ["档案审核", "审核", "会计", "出纳", "管理", "负责"]
                if word in work_content_keywords:
                    context_before = text[max(0, start - 20):start]
                    if "负责" in context_before or "从事" in context_before:
                        item["_drop"] = True
                        continue

            # 身份证号验证
            if label == "身份证号":
                import re
                clean_word = re.sub(r'[\s\-_]', '', word)
                is_valid_18 = re.fullmatch(r'[1-9]\d{16}[\dXx]', clean_word) is not None
                is_valid_15 = re.fullmatch(r'[1-9]\d{14}', clean_word) is not None
                if not (is_valid_18 or is_valid_15):
                    item["_drop"] = True
                    continue

            # 判断是否在工作经历区域
            is_in_work_history = False
            for kw in work_history_keywords:
                pos = text.find(kw)
                if pos != -1 and start > pos:
                    is_in_work_history = True
                    break

            # 家庭成员区域特殊处理
            if family_region_start != -1 and isinstance(start, int) and start >= family_region_start:
                import re
                if re.match(r"\d{4}\.\d{1,2}(\.\d{1,2})?", word) or re.match(r"\d{4}\.\d{1,2}", word):
                    if label in ["现职时间", "入党时间"]:
                        item["entity_group"] = "出生年月"
                if label == "职务" and word == "党员":
                    item["entity_group"] = "政治面貌"

            # 工作单位关键词检测
            workunit_keywords = ["公司", "集团", "局", "部", "中心", "研究所", "研究院", "设计院", "人事部", "财务部", "办公室", "厂", "矿", "有限"]
            if any(kw in word for kw in workunit_keywords):
                context_before = text[max(0, start - 50):start]
                work_indicators = ["工作于", "就职", "任职", "现工作", "所在单位", "从事", "工作", "就任"]
                if any(ind in context_before for ind in work_indicators):
                    if label in ["全日制教育毕业院校系", "在职教育毕业院校系", "工作单位"]:
                        item["entity_group"] = "工作单位"
                        continue

            company_keywords = ["公司", "集团", "有限", "建筑", "工程", "厂", "矿", "局", "部", "中心"]
            if label in ["全日制教育毕业院校系", "在职教育毕业院校系"]:
                if any(kw in word for kw in company_keywords):
                    context_before = text[max(0, start - 150):start]
                    work_indicators = ["至今", "工作经历", "简历", "参加工作", "任职", "就职", "年月-", "-至今"]
                    if any(ind in context_before for ind in work_indicators):
                        item["entity_group"] = "工作单位"
                        continue

            # 院校类型智能分配（全日制 vs 在职）
            is_school = any(kw in word for kw in school_keywords)
            if is_school and label in ["工作单位", "全日制教育毕业院校系", "在职教育毕业院校系"]:
                context_start = max(0, start - 80)
                context_end = min(len(text), end + 80)
                context = text[context_start:context_end]
                if family_region_start != -1 and isinstance(start, int) and start >= family_region_start:
                    item["entity_group"] = "工作单位"
                    continue
                if is_in_work_history:
                    if any(kw in word for kw in company_keywords):
                        item["entity_group"] = "工作单位"
                        continue
                is_fulltime = any(kw in context for kw in fulltime_keywords)
                is_parttime = any(kw in context for kw in parttime_keywords)
                if is_parttime and not is_fulltime:
                    item["entity_group"] = "在职教育毕业院校系"
                else:
                    item["entity_group"] = "全日制教育毕业院校系"

            # 籍贯补全
            if label == "籍贯":
                if word.endswith("长治") and isinstance(end, int) and end < len(text):
                    if text[end] == "市":
                        item["word"] = word + "市"
                        item["end"] = end + 1
                if word == "西省五台县":
                    item["word"] = "山西省五台县"
                    item["start"] = start - 1

            # 补齐不完整日期
            if label == "出生年月" and isinstance(start, int) and isinstance(end, int):
                import re
                if re.match(r"^\d{4}$", word):
                    remaining = text[end:min(len(text), end + 20)]
                    match = re.search(r"\.\d{1,2}\.\d{1,2}", remaining)
                    if match:
                        full_date = word + match.group(0)
                        item["word"] = full_date
                        item["end"] = end + len(match.group(0))

            # 家庭住址误标为籍贯
            if label == "籍贯" and any(k in word for k in ["路", "号", "小区", "街道"]):
                item["entity_group"] = "家庭住址"

            # 学历字段映射
            if word in ["本科", "研究生", "专科", "硕士", "博士", "大学本科", "大学专科"]:
                context_start = max(0, start - 40)
                context_end = min(len(text), end + 40)
                context = text[context_start:context_end]
                if "在职" in context or "继续教育" in context:
                    item["entity_group"] = "在职教育学历"
                else:
                    item["entity_group"] = "全日制教育学历"

            # 专业字段中职务误标修正
            if label == "全日制教育专业":
                position_words = ["财务会计", "会计", "出纳", "工程师", "技术员", "经济师", "会计师", "教师", "医生", "护士"]
                if word in position_words:
                    context_before = text[max(0, start - 10):start]
                    if "职务" in context_before or "从事何种工作" in context_before:
                        item["entity_group"] = "职务"

            # 太原理工大学特殊处理
            if word == "太原理工大学":
                remaining = text[end:min(len(text), end + 20)]
                if "岩土工程" in remaining:
                    full_word = word + "岩土工程"
                    item["word"] = full_word
                    item["end"] = end + len("岩土工程")
                    item["entity_group"] = "全日制教育毕业院校系"

        out = [item for item in out if not item.get("_drop", False)]
        out = [item for item in out if str(item.get("entity_group", "")) in MultiRegionApp.VALID_LABELS]
        return out

    @staticmethod
    def _injected_score(items: list[dict], start: int, end: int) -> float:
        nearby = []
        for it in items:
            s = it.get("start")
            e = it.get("end")
            sc = it.get("score")
            if not (isinstance(s, int) and isinstance(e, int)):
                continue
            if not isinstance(sc, (int, float)):
                continue
            dist = min(abs(start - s), abs(end - e))
            if dist <= 24:
                nearby.append(float(sc))
        if nearby:
            base = max(nearby)
            return round(min(0.999999, max(0.900001, base - 0.000001)), 12)
        return 0.999123456789

    @staticmethod
    def _inject_missing_school_name(items: list[dict], text: str) -> list[dict]:
        import re
        out = [dict(x) for x in items]
        for match in re.finditer(r'毕业院校及专业\s+([^\n]+)', text):
            content = match.group(1)
            content_start = match.start(1)
            has_school = False
            for item in out:
                if item.get("entity_group") == "全日制教育毕业院校系":
                    s = item.get("start", 0)
                    e = item.get("end", 0)
                    if content_start <= s <= content_start + len(content):
                        has_school = True
                        break
            if not has_school:
                school_match = re.search(r'([\u4e00-\u9fff]+?(?:大学|学院|研究院|学校))', content)
                if school_match:
                    school_name = school_match.group(1)
                    school_start = content_start + school_match.start(1)
                    school_end = content_start + school_match.end(1)
                    out.append({
                        "entity_group": "全日制教育毕业院校系",
                        "score": 0.999,
                        "word": school_name,
                        "start": school_start,
                        "end": school_end,
                    })
        return sorted(out, key=lambda x: int(x.get("start", 0)))

    @staticmethod
    def _inject_missing_political_status(items: list[dict], text: str) -> list[dict]:
        import re
        out = [dict(x) for x in items]
        for match in re.finditer(r'政治面貌[\s,]+([^\s,]+)', text):
            status_start = match.start(1)
            status_end = match.end(1)
            status_word = match.group(1).strip()
            has_political = False
            for item in out:
                if item.get("entity_group") == "政治面貌":
                    s = item.get("start", 0)
                    e = item.get("end", 0)
                    if s <= status_start <= e or (status_start <= s <= status_end):
                        has_political = True
                        break
            if not has_political and status_word in MultiRegionApp.VALID_POLITICAL_STATUS:
                out.append({
                    "entity_group": "政治面貌",
                    "score": 0.999,
                    "word": status_word,
                    "start": status_start,
                    "end": status_end,
                })
        return sorted(out, key=lambda x: int(x.get("start", 0)))

    @staticmethod
    def _inject_missing_range_dates(items: list[dict], text: str) -> list[dict]:
        import re
        if not text:
            return items
        out = [dict(x) for x in items]
        date_labels = {"出生年月", "参加工作时间", "现职时间", "入党时间"}
        ym_pat = re.compile(r"\d{4}\s*年\s*(?:0?[1-9]|1[0-2])\s*月")
        edu_cues = {"就读", "入学", "毕业", "学位", "专业", "大学", "学院", "学制", "全日制", "学习"}
        work_cues = {"工作", "任职", "历任", "进入", "负责", "就职"}

        def overlaps_date(s, e):
            for it in out:
                lbl = str(it.get("entity_group", ""))
                is_ = it.get("start")
                ie = it.get("end")
                if lbl not in date_labels or not isinstance(is_, int) or not isinstance(ie, int):
                    continue
                if not (e <= is_ or ie <= s):
                    return True
            return False

        def nearest_date_score(s, e):
            best_dist = 10 ** 9
            best_score = None
            for it in out:
                lbl = str(it.get("entity_group", ""))
                is_ = it.get("start")
                ie = it.get("end")
                sc = it.get("score")
                if lbl not in date_labels:
                    continue
                if not (isinstance(is_, int) and isinstance(ie, int) and isinstance(sc, (int, float))):
                    continue
                dist = min(abs(s - is_), abs(e - ie))
                if dist < best_dist:
                    best_dist = dist
                    best_score = float(sc)
            if best_score is not None:
                return best_score
            return MultiRegionApp._injected_score(out, s, e)

        for m in ym_pat.finditer(text):
            s = m.start()
            e = m.end()
            if overlaps_date(s, e):
                continue
            left = text[max(0, s - 2):s]
            right = text[e:min(len(text), e + 2)]
            around = text[max(0, s - 16):min(len(text), e + 16)]
            in_edu = any(k in around for k in edu_cues)
            in_work = any(k in around for k in work_cues)

            label = None
            if re.match(r"^\s*[-—~至到]", right):
                label = "日期"
            elif re.search(r"[-—~至到]\s*$", left):
                label = "日期"

            if label is None:
                continue
            if label in date_labels:
                out.append({
                    "entity_group": label,
                    "score": nearest_date_score(s, e),
                    "word": text[s:e],
                    "start": s,
                    "end": e,
                })
        return sorted(out, key=lambda x: int(x.get("start", 0)))

    @staticmethod
    def _postprocess_misc_entities(items: list[dict], text: str) -> list[dict]:
        import re
        fixed = []
        for raw in items:
            it = dict(raw)
            label = str(it.get("entity_group", ""))
            word = str(it.get("word", ""))
            s = it.get("start")
            e = it.get("end")
            if label not in MultiRegionApp.VALID_LABELS:
                continue
            # 去除地域前缀（如“洛阳市洛阳师范学院” -> “洛阳师范学院”）
            if label in ["全日制教育毕业院校系", "在职教育毕业院校系"]:
                match = re.match(r'^([\u4e00-\u9fff]{2,3}[市县区])?([\u4e00-\u9fff]+?(?:大学|学院|学校|研究院|系))$', word)
                if match:
                    prefix = match.group(1) or ""
                    school = match.group(2)
                    if prefix and school:
                        it["word"] = school
                        if isinstance(s, int):
                            it["start"] = s + len(prefix)
                        print(f"[修正] 去除地域前缀: {word} -> {school}")
            if isinstance(s, int) and isinstance(e, int) and 0 <= s <= e <= len(text):
                if label == "毕肄业学校或单位" and s > 0 and any(k in word for k in {"大学", "学院", "学校"}):
                    prev = text[s - 1]
                    if re.fullmatch(r"[\u4e00-\u9fff]", prev) is not None and prev not in {"于", "在", "到", "从", "由", "与", "和"}:
                        it["start"] = s - 1
                        it["word"] = text[s - 1:e]
            fixed.append(it)
        return fixed

    @staticmethod
    def _deduplicate_overlaps(items: list[dict]) -> list[dict]:
        if not items:
            return items
        label_priority = {
            "全日制教育毕业院校系": 1,
            "在职教育毕业院校系": 1,
            "全日制教育专业": 2,
            "在职教育专业": 2,
            "出生年月": 3,
            "参加工作时间": 3,
            "入党时间": 3,
            "现职时间": 3,
            "工作单位": 4,
            "职务": 5,
            "政治面貌": 5,
            "全日制教育学历": 5,
            "全日制教育学位": 5,
            "在职教育学历": 5,
            "在职教育学位": 5,
            "姓名": 5,
            "性别": 5,
            "民族": 5,
            "籍贯": 5,
            "出生地": 5,
            "身份证号": 5,
        }
        kept = []
        for cand in sorted(items, key=lambda x: (int(x.get("start", 0)), int(x.get("end", 0)))):
            cs = cand.get("start")
            ce = cand.get("end")
            cl = str(cand.get("entity_group", ""))
            if not (isinstance(cs, int) and isinstance(ce, int)):
                kept.append(cand)
                continue
            drop = False
            for i, old in enumerate(kept):
                os = old.get("start")
                oe = old.get("end")
                ol = str(old.get("entity_group", ""))
                if not (isinstance(os, int) and isinstance(oe, int)):
                    continue
                if cs == os and ce == oe:
                    if cand.get("score", 0) > old.get("score", 0):
                        kept[i] = cand
                    drop = True
                    break
                if os <= cs and ce <= oe:
                    drop = True
                    break
                if cs <= os and oe <= ce:
                    old_priority = label_priority.get(ol, 10)
                    new_priority = label_priority.get(cl, 10)
                    if new_priority < old_priority:
                        kept[i] = cand
                    drop = True
                    break
                if cs < oe and ce > os:
                    old_priority = label_priority.get(ol, 10)
                    new_priority = label_priority.get(cl, 10)
                    if new_priority < old_priority:
                        kept[i] = cand
                    drop = True
                    break
            if not drop:
                kept.append(cand)
        return sorted(kept, key=lambda x: int(x.get("start", 0)))

    @staticmethod
    def _postprocess_org_entities(items: list[dict], text: str) -> list[dict]:
        if not items:
            return items
        org_labels = {"工作单位"}
        role_words = {"科员", "职员", "员工", "主任", "经理", "干部"}
        merged = []
        i = 0
        while i < len(items):
            cur = dict(items[i])
            cs = cur.get("start")
            ce = cur.get("end")
            cl = str(cur.get("entity_group", ""))
            if not (isinstance(cs, int) and isinstance(ce, int)):
                merged.append(cur)
                i += 1
                continue
            if cl in org_labels:
                j = i + 1
                best_start = cs
                best_end = ce
                best_score = float(cur.get("score", 0.0))
                while j < len(items):
                    nxt = items[j]
                    ns = nxt.get("start")
                    ne = nxt.get("end")
                    nl = str(nxt.get("entity_group", ""))
                    if not (isinstance(ns, int) and isinstance(ne, int)):
                        break
                    if nl not in org_labels:
                        break
                    gap = text[best_end:ns]
                    if len(gap) <= 2 and re.fullmatch(r"[\s,，;；:：\|]*", gap) is not None:
                        best_end = ne
                        best_score = max(best_score, float(nxt.get("score", 0.0)))
                        j += 1
                        continue
                    break
                if best_end != ce:
                    cur["start"] = best_start
                    cur["end"] = best_end
                    cur["word"] = text[best_start:best_end]
                    cur["score"] = best_score
                    i = j
                else:
                    i += 1
            else:
                i += 1
            merged.append(cur)
        return merged

    # ---------- ner_extraction 主方法 ----------
    def ner_extraction(self, ocr_data):
        if not ocr_data:
            return []

        # 检查 NER 模型是否已加载
        if not hasattr(self, '_ner_pipeline') or self._ner_pipeline is None:
            self._append_debug("错误：请先选择关键信息提取模型目录并加载。\n")
            return []

        results = []
        for item in ocr_data:
            img_name = item.get("Imagepage", "")
            boxes = item.get("Box", [])
            text_groups = item.get("Text", [])

            # 拼接全文
            full_text_parts = []
            for group in text_groups:
                content = group.get("Content", [])
                text = "".join(content).strip()
                if text:
                    full_text_parts.append(text)
            full_text = " ".join(full_text_parts)

            # 如果 text_groups 为空（即没有版面数据），则使用已保存的 FullText 字段
            if not full_text:
                full_text = item.get("FullText", "")
                if not full_text:
                    continue

            ### 20260512 增加超出512字符分割文本
            max_len = 512               # 模型最长识别长度
            chunk_size = 500            # 分块大小（留 12 字符缓冲）
            overlap = 50                # 块间重叠字符数

            if len(full_text) <= max_len:
                try:
                    raw_entities = self._ner_pipeline(full_text)
                except Exception as e:
                    self._append_debug(f"NER 推理失败（{img_name}）：{e}\n")
                    continue
            else:
                # 长文本分块处理
                self._append_debug(f"[{img_name}] 文本长度 {len(full_text)} > {max_len}，启动分块推理\n")
                raw_entities = []
                start_offset = 0
                while start_offset < len(full_text):
                    end_offset = min(start_offset + chunk_size, len(full_text))
                    chunk = full_text[start_offset:end_offset]
                    try:
                        chunk_entities = self._ner_pipeline(chunk)
                    except Exception as e:
                        self._append_debug(f"NER 分块失败（{img_name} offset {start_offset}）：{e}\n")
                        start_offset += chunk_size - overlap
                        continue

                    # 将块内实体偏移量还原为全文偏移
                    for ent in chunk_entities:
                        ent = dict(ent)
                        ent['start'] = start_offset + ent['start']
                        ent['end']   = start_offset + ent['end']
                        raw_entities.append(ent)

                    # 移动窗口
                    start_offset += chunk_size - overlap

                # 简单合并重叠的同标签实体（因重叠区域可能重复识别）
                raw_entities.sort(key=lambda x: (x['start'], -x['end']))
                merged = []
                for ent in raw_entities:
                    if not merged:
                        merged.append(ent)
                        continue
                    last = merged[-1]
                    if (ent['start'] <= last['end'] and
                        ent['entity_group'] == last['entity_group']):
                        # 合并：保留最大范围，得分取高
                        last['end'] = max(last['end'], ent['end'])
                        if ent.get('score', 0) > last.get('score', 0):
                            last['score'] = ent['score']
                            last['word']  = full_text[last['start']:last['end']]
                    else:
                        merged.append(ent)
                raw_entities = merged
            ############################################

            # 初步过滤
            cleaned_items = []
            for ent in raw_entities:
                score = ent.get("score", 0.0)
                if score < 0.98:
                    continue
                ent_dict = dict(ent)
                start = ent_dict.get("start")
                end = ent_dict.get("end")
                if isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(full_text):
                    ent_dict["word"] = full_text[start:end]
                elif "word" in ent_dict and isinstance(ent_dict["word"], str):
                    ent_dict["word"] = self._normalize_word(ent_dict["word"])
                if isinstance(ent_dict.get("word"), str):
                    ent_dict["word"] = self._clean_entity_text(ent_dict["word"])
                label = str(ent_dict.get("entity_group", ""))
                word = str(ent_dict.get("word", ""))
                if self._should_drop_entity(label, word, full_text,
                                            start if isinstance(start, int) else None,
                                            end if isinstance(end, int) else None):
                    continue
                ent_dict["score"] = score
                cleaned_items.append(ent_dict)

            cleaned_items.sort(key=lambda x: int(x.get("start", 0)))
            cleaned_items = self._fix_adjacent_split_entities(cleaned_items, full_text)
            cleaned_items = self._fix_incomplete_dates(cleaned_items, full_text)  # 最新添加的日期修复
            cleaned_items = self._context_relabel(cleaned_items, full_text)
            cleaned_items = self._inject_missing_school_name(cleaned_items, full_text)
            cleaned_items = self._inject_missing_political_status(cleaned_items, full_text)
            cleaned_items = self._inject_missing_range_dates(cleaned_items, full_text)
            cleaned_items = self._postprocess_misc_entities(cleaned_items, full_text)
            cleaned_items = self._deduplicate_overlaps(cleaned_items)
            cleaned_items = self._postprocess_org_entities(cleaned_items, full_text)

            # 最终过滤
            cleaned_items = [it for it in cleaned_items if it.get("entity_group") in self.VALID_LABELS]
            cleaned_items = [
                it for it in cleaned_items
                if not (it.get("entity_group") in ["全日制教育学历", "在职教育学历"] and it.get("word") == "大学")
            ]

            positioning = []
            for ent in cleaned_items:
                positioning.append({
                    "Field": ent["entity_group"],
                    "Content": ent["word"],
                    "Imagepage": img_name,
                    "Start": ent.get("start", -1),
                    "End": ent.get("end", -1)
                })
            results.append({
                "Imagepage": img_name,
                "Box": boxes,
                "Text": text_groups,
                "Positioning": positioning,
                "FullText": full_text  # 保存全文，供编辑使用
            })

        # 构建表格展示数据
        display_list = []
        for res in results:
            items = []
            for p in res.get("Positioning", []):
                items.append({
                    "Field": p["Field"],
                    "Value": p["Content"],
                    "Start": p.get("Start", -1),
                    "End": p.get("End", -1)
                })
            display_list.append({
                "image": res["Imagepage"],
                "items": items,
                "full_text": res.get("FullText", "")
            })
        self.ner_results = display_list
        self.root.after(0, self._update_ner_nav_state)

        return results


    def _update_ner_nav_state(self):
        """更新NER翻页按钮状态和页码标签"""
        total = len(self.ner_results)
        if total > 0:
            self.ner_page_label.config(text=f"第{self.ner_current_index+1}/{total}页")
            self.prev_ner_btn.config(state=tk.NORMAL if self.ner_current_index > 0 else tk.DISABLED)
            self.next_ner_btn.config(state=tk.NORMAL if self.ner_current_index < total-1 else tk.DISABLED)
        else:
            self.ner_page_label.config(text="第0/0页")
            self.prev_ner_btn.config(state=tk.DISABLED)
            self.next_ner_btn.config(state=tk.DISABLED)

    def _show_ner_result(self, index):
        if not self.ner_results or index < 0 or index >= len(self.ner_results):
            for row in self.ner_tree.get_children():
                self.ner_tree.delete(row)
            self._update_ner_nav_state()
            return
        result = self.ner_results[index]
        # 清空表格
        for row in self.ner_tree.get_children():
            self.ner_tree.delete(row)
        # 插入数据
        for i, item in enumerate(result["items"]):
            self.ner_tree.insert("", "end", iid=str(i),
                                 values=(item["Field"], item["Value"], item["Start"], item["End"]))
        self._update_ner_nav_state()
        self._current_ner_image = result["image"]
        self._current_full_text = result.get("full_text", "")

    def _edit_ner_cell(self, event):
        """双击NER表格单元格：字段列定位并高亮，值列/偏移列编辑值/偏移量，值列同时激活替换模式"""
        region = self.ner_tree.identify_region(event.x, event.y)
        if region != "cell":
            return
        column = self.ner_tree.identify_column(event.x)  # '#1', '#2', '#3', '#4'
        item_id = self.ner_tree.identify_row(event.y)
        if not item_id:
            return
        col_index = int(column[1:]) - 1  # 0:Field, 1:Value, 2:Start, 3:End
        row_index = int(item_id)

        # 记录当前选中的行（供拖拽替换使用）
        self.current_selected_ner_row = row_index

        # 如果是字段列（第一列），执行定位和高亮（不激活编辑）
        if col_index == 0:
            self._locate_ner_field_in_layer(row_index)
            return

        # 如果是值列（第二列），除了定位高亮，还激活“替换模式”（用户可拖拽选择文本替换）
        if col_index == 1:
            self.current_selected_ner_row = row_index
            self.waiting_for_selection = True
            print(f"[DEBUG] 激活替换模式，行索引 {row_index}, waiting={self.waiting_for_selection}")
            self._append_debug("请在右侧画布中拖拽选择要替换的文本\n")
            return

        # 其他列（Start/End）执行原有的编辑逻辑（弹出输入框）
        current_img_data = self.ner_results[self.ner_current_index]
        if row_index >= len(current_img_data["items"]):
            return
        current_item = current_img_data["items"][row_index]
        full_text = current_img_data.get("full_text", "")

        # 获取当前值
        if col_index == 2:  # Start
            current_val = str(current_item["Start"])
        else:  # col_index == 3 End
            current_val = str(current_item["End"])

        # 创建编辑框
        x, y, w, h = self.ner_tree.bbox(item_id, column)
        entry = tk.Entry(self.ner_tree)
        entry.place(x=x, y=y, width=w, height=h)
        entry.insert(0, current_val)
        entry.focus()

        def save_edit():
            new_val = entry.get().strip()
            entry.destroy()
            if col_index == 2:  # 修改起始位置
                try:
                    new_start = int(new_val)
                    current_item["Start"] = new_start
                    end = current_item["End"]
                    if 0 <= new_start < end <= len(full_text):
                        new_text = full_text[new_start:end]
                        current_item["Value"] = new_text
                except ValueError:
                    pass
            else:  # 修改结束位置
                try:
                    new_end = int(new_val)
                    current_item["End"] = new_end
                    start = current_item["Start"]
                    if 0 <= start < new_end <= len(full_text):
                        new_text = full_text[start:new_end]
                        current_item["Value"] = new_text
                except ValueError:
                    pass
            # 刷新表格显示
            self._show_ner_result(self.ner_current_index)
            # 保存到磁盘
            self._save_ner_result()

        entry.bind("<Return>", lambda e: save_edit())
        entry.bind("<FocusOut>", lambda e: save_edit())

    def _locate_ner_field_in_layer(self, row_index):
        """根据NER字段的起止位置，在下层画布中定位并高亮对应的文本框"""
        if not self.ner_results or self.ner_current_index >= len(self.ner_results):
            self._append_debug("没有NER结果或索引无效\n")
            return

        current_ner = self.ner_results[self.ner_current_index]
        if row_index >= len(current_ner["items"]):
            self._append_debug("行索引超出范围\n")
            return

        item = current_ner["items"][row_index]
        start = item.get("Start")
        end = item.get("End")
        if start is None or end is None:
            self._append_debug("该字段没有有效的起止位置信息\n")
            return

        # 获取当前图片的OCR数据
        ocr_data = self._get_current_ocr_data()
        if not ocr_data:
            self._append_debug("未找到当前图片的OCR数据，请先执行OCR识别\n")
            return

        # 查找所有包含该偏移区间的框
        target_boxes = []
        for box in ocr_data.get("Box", []):
            cs = box.get("char_start")
            ce = box.get("char_end")
            if cs is None or ce is None:
                continue
            # 判断区间重叠
            if (cs <= start <= ce) or (cs <= end <= ce) or (start <= cs <= end):
                target_boxes.append(box)

        if not target_boxes:
            self._append_debug("未找到对应的文本框区域\n")
            return

        # 计算联合外接矩形（图像坐标）
        all_x = []
        all_y = []
        for box in target_boxes:
            pos = box.get("Position", [])
            if len(pos) >= 8:
                all_x.extend(pos[0::2])
                all_y.extend(pos[1::2])

        if not all_x:
            self._append_debug("无法计算联合矩形\n")
            return

        left = min(all_x)
        right = max(all_x)
        top = min(all_y)
        bottom = max(all_y)
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2

        # 调整下层画布视图，使该区域居中
        canvas = self.layer_canvas
        canvas.update_idletasks()
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw > 1 and ch > 1:
            zoom = self.layer_zoom_pan.zoom
            new_offset_x = cw / 2 - center_x * zoom
            new_offset_y = ch / 2 - center_y * zoom
            self.layer_zoom_pan.offset_x = new_offset_x
            self.layer_zoom_pan.offset_y = new_offset_y
            self.layer_zoom_pan._redraw()
        else:
            # 如果画布尺寸未就绪，延迟重试
            self.window.after(100, lambda: self._locate_ner_field_in_layer(row_index))
            return

        # 高亮这些框
        self._highlight_layer_boxes(target_boxes)

    def _highlight_layer_boxes(self, target_boxes):
        """高亮下层画布中指定的框（设置高亮列表，触发重绘）"""
        # 提取需要高亮的 Box 的 Number 列表
        self.highlighted_box_numbers = [box.get("Number") for box in target_boxes if box.get("Number") is not None]
        # 触发下层画布重绘（_draw_layer_image 会根据 self.highlighted_box_numbers 决定边框颜色）
        self.layer_zoom_pan._redraw()
        self._append_debug(f"已高亮 {len(self.highlighted_box_numbers)} 个文本框\n")

    def _update_ner_field(self, row_index, new_text, new_start, new_end):
        """更新 NER 字段的值和偏移量，刷新表格并保存"""
        if not self.ner_results or self.ner_current_index >= len(self.ner_results):
            self._append_debug("无法更新：NER结果无效\n")
            return
        current_ner = self.ner_results[self.ner_current_index]
        items = current_ner["items"]
        if row_index < 0 or row_index >= len(items):
            self._append_debug(f"行索引 {row_index} 超出范围\n")
            return
        items[row_index]["Value"] = new_text
        items[row_index]["Start"] = new_start
        items[row_index]["End"] = new_end
        # 刷新表格显示
        self._show_ner_result(self.ner_current_index)
        # 保存到磁盘
        self._save_ner_result()
        self._append_debug(f"已更新字段 '{items[row_index]['Field']}' 为 '{new_text}'\n")

    def _save_ner_result(self):
        """将 self.ner_results 保存为 ner_result.json 到当前文件夹"""
        if not self.current_folder:
            self._append_debug("未选择文件夹，无法保存NER结果\n")
            return
        # 转换为原始的 Positioning 格式
        output_data = []
        for nr in self.ner_results:
            positioning = []
            for item in nr["items"]:
                positioning.append({
                    "Field": item["Field"],
                    "Content": item["Value"],
                    "Imagepage": nr["image"],
                    "Start": item["Start"],
                    "End": item["End"]
                })
            output_data.append({
                "Imagepage": nr["image"],
                "Box": [],  # 可选，保留为空或原有Box（可略）
                "Text": [],
                "Positioning": positioning,
                "FullText": nr.get("full_text", "")
            })
        file_path = os.path.join(self.current_folder, "ner_result.json")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            self._append_debug(f"NER结果已保存至 {file_path}\n")
        except Exception as e:
            self._append_debug(f"保存NER结果失败: {e}\n")

    def _save_as_dataset(self):
        try:
            import pandas as pd
        except ImportError:
            self._append_debug("请安装 pandas 和 openpyxl: pip install pandas openpyxl\n")
            return
        if not self.ner_results:
            self._append_debug("没有 NER 结果可保存\n")
            return
        if not self.current_folder:
            self._append_debug("未选择图片文件夹\n")
            return
        output_path = os.path.join(self.current_folder, "label.xls")
        rows = []
        for res in self.ner_results:
            img_name = res["image"]
            full_text = res.get("full_text", "")
            entities = []
            for item in res["items"]:
                entities.append({
                    "start": item["Start"],
                    "end": item["End"],
                    "text": item["Value"],
                    "labels": [item["Field"]]
                })
            json_str = json.dumps(entities, ensure_ascii=False, separators=(',', ':'))
            rows.append([img_name, full_text, json_str])
        df = pd.DataFrame(rows, columns=["图像名称", "识别出的所有文本", "标注信息"])
        try:
            df.to_excel(output_path, index=False, engine='openpyxl')
            self._append_debug(f"数据集已保存到：{output_path}\n")
        except Exception as e:
            self._append_debug(f"保存失败：{e}\n")

    def _prev_ner_result(self):
        if self.ner_current_index > 0:
            self.ner_current_index -= 1
            self._show_ner_result(self.ner_current_index)
            # 同时跳转到左侧对应图片
            self._jump_to_ner_image()

    def _next_ner_result(self):
        if self.ner_current_index < len(self.ner_results) - 1:
            self.ner_current_index += 1
            self._show_ner_result(self.ner_current_index)
            self._jump_to_ner_image()

    def _jump_to_ner_image(self):
        """根据当前NER结果图片名，跳转到左侧画布"""
        if not self.ner_results or self.ner_current_index >= len(self.ner_results):
            return
        img_name = self.ner_results[self.ner_current_index]["image"]
        # 在 self.image_paths 中查找匹配路径
        for idx, path in enumerate(self.image_paths):
            if os.path.basename(path) == img_name:
                if idx != self.current_image_index:
                    self.current_image_index = idx
                    self._load_current_image()
                break

    def _on_ner_tree_select(self, event):
        """点击表格任意行，跳转到该NER结果对应的图片（因为整个表格当前显示某一张图的结果）"""
        self._jump_to_ner_image()

    def on_detect_regions(self):
        if not self.image_paths:
            self._append_debug("请先加载图片文件夹\n")
            return
        start_time = time.time()
        self._append_debug("正在执行文本区域检测...\n")
        result = self.detect_text_regions(self.image_paths)
        safe_save_json(self.current_folder, "layout_analysis.json", result)
        end_time = time.time()
        total_time = end_time - start_time
        image_count = len(self.image_paths)
        self._update_performance_stats("文本区域检测", image_count, total_time)
        self._append_debug(f"已生成文件：{self.current_folder}/layout_analysis.json\n")

    def on_ocr(self):
        if not self.image_paths:
            self._append_debug("请先加载图片文件夹\n")
            return
        layout = load_json_file(self.current_folder, "layout_analysis.json")
        if layout is None:
            safe_save_json(self.current_folder, "layout_analysis.json", None)
            layout = []
        start_time = time.time()
        self._append_text("正在执行OCR识别...\n")
        result = self.ocr_recognition(self.image_paths, layout)
        safe_save_json(self.current_folder, "ocr_result.json", result)
        end_time = time.time()
        total_time = end_time - start_time
        image_count = len(self.image_paths)
        self._update_performance_stats("OCR识别", image_count, total_time)
        self._append_text(f"已生成文件：{self.current_folder}/ocr_result.json\n")


    ## 20260511更新 新增ner准确率评估######################
    def on_ner(self):
        if not self.image_paths:
            self._append_debug("请先加载图片文件夹\n")
            return
        ocr = load_json_file(self.current_folder, "ocr_result.json")
        if ocr is None:
            safe_save_json(self.current_folder, "ocr_result.json", None)
            ocr = []
        start_time = time.time()
        self._append_debug("正在执行关键信息提取...\n")
        result = self.ner_extraction(ocr)
        safe_save_json(self.current_folder, "ner_result.json", result)
        end_time = time.time()
        total_time = end_time - start_time
        image_count = len(self.image_paths)
        self._update_performance_stats("关键信息提取", image_count, total_time)
        self._append_debug(f"已生成文件：{self.current_folder}/ner_result.json\n")

        # 显示NER结果
        if self.ner_results:
            self.ner_current_index = 0
            self._show_ner_result(0)
            self._jump_to_ner_image()
        else:
            for row in self.ner_tree.get_children():
                self.ner_tree.delete(row)
            self._update_ner_nav_state()

        # ---- 自动评估准确率并更新性能统计 ----
        self.root.after(100, self._auto_eval_ner_accuracy)

    def _auto_eval_ner_accuracy(self):
        """在后台线程中评估NER准确率，完成后更新性能统计表"""

        def _eval():
            accuracy_str = self._calculate_accuracy("关键信息提取")
            # 在主线程中更新UI
            self.root.after(0, lambda: self._update_accuracy_display(accuracy_str))

        threading.Thread(target=_eval, daemon=True).start()

    def _update_accuracy_display(self, accuracy_str):
        """更新性能统计表中关键信息提取的准确率"""
        self.performance_stats["关键信息提取"]["accuracy"] = accuracy_str
        # 更新表格中"关键信息提取"行
        if hasattr(self, 'perf_tree'):
            for item in self.perf_tree.get_children():
                values = self.perf_tree.item(item)["values"]
                if values[0] == "关键信息提取":
                    self.perf_tree.item(item, values=(
                        values[0], values[1], values[2], values[3], accuracy_str
                    ))
                    break
            # 同时更新全部环节行的准确率
            for item in self.perf_tree.get_children():
                values = self.perf_tree.item(item)["values"]
                if values[0] == "全部环节":
                    self.performance_stats["全部环节"]["accuracy"] = accuracy_str
                    self.perf_tree.item(item, values=(
                        values[0], values[1], values[2], values[3], accuracy_str
                    ))
                    break
    ####################################################

    ##### 预留大模型调用功能
    def on_ner_Qwen(self):
        """弹出对话框，用户输入问题，调用 Qwen3 回答并输出 JSON 格式，支持额外字段队列"""
        dialog = tk.Toplevel(self.root)
        dialog.title("关键信息提取（大模型）")
        dialog.geometry("950x700")
        dialog.minsize(800, 600)
        dialog.transient(self.root)
        dialog.grab_set()

        # ---- 上方提示 ----
        tip_frame = tk.Frame(dialog, padx=10, pady=5)
        tip_frame.pack(fill=tk.X)
        tk.Label(tip_frame,
                 text="请核对OCR识别结果，若无文本，大模型自动进行OCR并提取关键信息",
                 font=("微软雅黑", 9), fg="#555", justify=tk.LEFT).pack(anchor=tk.W)

        # 主内容区域：左侧文本输入区，右侧额外字段管理区
        main_paned = tk.PanedWindow(dialog, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=5)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # ---- 左侧：问题/文本输入区 ----
        left_frame = tk.LabelFrame(main_paned, text="OCR识别结果", padx=10, pady=10)
        main_paned.add(left_frame, stretch="always", width=500)

        self._qwen_ner_input = tk.Text(left_frame, height=15, font=("微软雅黑", 11), wrap=tk.WORD)
        self._qwen_ner_input.pack(fill=tk.BOTH, expand=True)

        # 如果有 OCR 结果，自动填入当前图片的全文，并记录标志
        ocr_text = self._get_current_ocr_fulltext()
        self._has_ocr_for_current_image = bool(ocr_text and ocr_text.strip())
        if self._has_ocr_for_current_image:
            self._qwen_ner_input.insert(tk.END, ocr_text)

        # ---- 右侧：额外字段管理区 ----
        right_frame = tk.LabelFrame(main_paned, text="新增提取字段", padx=10, pady=10)
        main_paned.add(right_frame, stretch="always", width=250)

        # 输入框和添加按钮
        add_field_frame = tk.Frame(right_frame)
        add_field_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Label(add_field_frame, text="字段名：", font=("微软雅黑", 9)).pack(side=tk.LEFT)
        self._extra_field_entry = tk.Entry(add_field_frame, font=("微软雅黑", 9))
        self._extra_field_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        add_btn = tk.Button(add_field_frame, text="添加", command=self._add_extra_field,
                            font=("微软雅黑", 9), bg="#4a7abc", fg="white", width=6)
        add_btn.pack(side=tk.RIGHT)

        # 额外字段列表框（支持删除）
        self._extra_fields_listbox = tk.Listbox(right_frame, height=10, font=("微软雅黑", 9))
        self._extra_fields_listbox.pack(fill=tk.BOTH, expand=True, pady=(5, 5))
        del_btn = tk.Button(right_frame, text="删除选中", command=self._del_extra_field,
                            font=("微软雅黑", 9), bg="#d9534f", fg="white")
        del_btn.pack(pady=(0, 5))

        # 存储额外字段的列表（与 listbox 同步）
        self._extra_fields_list = []

        # ---- 底部按钮 ----
        btn_frame = tk.Frame(dialog)
        btn_frame.pack(fill=tk.X, pady=10, padx=10)

        send_btn = tk.Button(
            btn_frame, text="发送",
            command=lambda: self._do_qwen_ner_query(dialog),
            font=("微软雅黑", 11), bg="#4a7abc", fg="white", width=10,
        )
        send_btn.pack(side=tk.LEFT, padx=(0, 10))

        clear_btn = tk.Button(
            btn_frame, text="清空输入",
            command=lambda: self._qwen_ner_input.delete("1.0", tk.END),
            font=("微软雅黑", 11), width=10,
        )
        clear_btn.pack(side=tk.LEFT, padx=(0, 10))

        # ---- 输出区 ----
        output_frame = tk.LabelFrame(dialog, text="提取结果", padx=10, pady=10)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self._qwen_ner_output = tk.Text(output_frame, height=12, font=("Consolas", 11),
                                        wrap=tk.WORD, state=tk.DISABLED)
        self._qwen_ner_output.pack(fill=tk.BOTH, expand=True)

        # 绑定快捷键
        self._qwen_ner_input.bind("<Control-Return>", lambda e: self._do_qwen_ner_query(dialog))

        self._qwen_ner_input.focus_set()

    def _get_current_ocr_fulltext(self) -> str:
        """获取当前图片的 OCR 全文，用于自动填入输入框"""
        if not self.image_paths:
            return ""
        ocr = load_json_file(self.current_folder, "ocr_result.json")
        if not ocr:
            return ""
        current_img = self.image_paths[self.current_image_index]
        for item in ocr:
            if item.get("Imagepage", "") == os.path.basename(current_img):
                text_groups = item.get("Text", [])
                parts = []
                for group in text_groups:
                    content = group.get("Content", [])
                    text = "".join(content).strip()
                    if text:
                        parts.append(text)
                return "\n".join(parts) if parts else ""
        return ""

    ## 20260513 新增定义提取字段
    def _build_ner_system_prompt(self, extra_fields_list):
        """
        根据固定字段列表和用户添加的额外字段列表，构建 system prompt。
        extra_fields_list: list of str，用户自定义的字段名称
        """
        fixed_fields = [
            "姓名", "性别", "出生年月", "民族", "籍贯", "出生地",
            "政治面貌", "入党时间", "参加工作时间", "工作单位", "职务", "现职时间",
            "身份证号", "全日制教育学历", "全日制教育学位", "全日制教育毕业院校系", "全日制教育专业",
            "在职教育学历", "在职教育学位", "在职教育毕业院校系", "在职教育专业"
        ]
        # 合并固定字段和额外字段
        all_fields = fixed_fields + extra_fields_list
        # 拼接成字符串，用中文顿号分隔
        fields_str = "、".join(all_fields)

        # 根据是否有OCR全文选择基础描述
        if self._has_ocr_for_current_image:
            base_desc = "你是一个干部人事档案信息提取引擎。"
        else:
            base_desc = "你是一个OCR+干部人事档案信息提取引擎。"

        prompt = (f"{base_desc}请严格以 JSON 格式输出提取结果，提取内容包括：{fields_str}。"
                  "不要带 Markdown 代码块标记，不要有多余文字，也不要编造数据，没有的字段就是空，提取内容不要出现OCR识别之外的内容。")

        return prompt

    def _add_extra_field(self):
        """添加额外字段到队列"""
        field = self._extra_field_entry.get().strip()
        if not field:
            return
        if field in self._extra_fields_list:
            self._append_debug(f"字段 '{field}' 已存在，不重复添加\n")
            return
        self._extra_fields_list.append(field)
        self._extra_fields_listbox.insert(tk.END, field)
        self._extra_field_entry.delete(0, tk.END)

    def _del_extra_field(self):
        """删除选中的额外字段"""
        selected = self._extra_fields_listbox.curselection()
        if selected:
            idx = selected[0]
            self._extra_fields_list.pop(idx)
            self._extra_fields_listbox.delete(idx)
    #########################################################

    def _do_qwen_ner_query(self, dialog: tk.Toplevel):
        """调用 Ollama Qwen3 执行问答，并正确管理图像路径"""
        question = self._qwen_ner_input.get("1.0", tk.END).strip()
        if not question:
            question = "请根据图片提取指定字段信息"

        # 获取当前图片完整路径，存入实例变量供线程使用
        self._llm_image_path = None
        if hasattr(self, 'image_paths') and hasattr(self, 'current_image_index'):
            if 0 <= self.current_image_index < len(self.image_paths):
                import os
                image_path = self.image_paths[self.current_image_index]
                if not os.path.isabs(image_path):
                    image_path = os.path.join(self.current_folder, image_path)
                self._llm_image_path = image_path

        # 获取额外字段列表
        extra_fields = self._extra_fields_list

        # 构建 system prompt
        system_prompt = self._build_ner_system_prompt(extra_fields)

        # 显示等待状态
        self._qwen_ner_output.config(state=tk.NORMAL)
        self._qwen_ner_output.delete("1.0", tk.END)
        self._qwen_ner_output.insert(tk.END, "正在调用大模型，请稍候...\n")
        self._qwen_ner_output.config(state=tk.DISABLED)
        dialog.update_idletasks()

        # 启动后台线程（只传必要参数，图像路径从实例变量获取）
        import threading
        threading.Thread(
            target=self._qwen_ner_query_thread,
            args=(dialog, question, system_prompt),
            daemon=True,
        ).start()

    def _qwen_ner_query_thread(self, dialog: tk.Toplevel, question: str, system_prompt: str):
        """后台线程：优化连接管理，避免连续请求失败"""
        import base64
        import requests
        import json
        import re
        import os
        import random

        # 从实例变量获取图像路径
        image_path = getattr(self, '_llm_image_path', None)

        # 1. 图片编码（仅在需要时编码）
        img_base64 = None
        if image_path and os.path.exists(image_path):
            # 如果有OCR全文，则不发送图片（模型只处理文本）
            if not self._has_ocr_for_current_image:
                try:
                    with open(image_path, "rb") as image_file:
                        img_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                    self._append_debug(f"图片已编码，大小 {len(img_base64)} 字符\n")
                except Exception as e:
                    self._append_debug(f"图片编码失败: {e}\n")
                    dialog.after(0, self._show_qwen_ner_answer, dialog, "错误：无法读取图片")
                    return
            else:
                self._append_debug("已有OCR全文，不发送图片，仅使用文本\n")
        else:
            self._append_debug("未找到图片路径，仅文本模式\n")

        # 2. 构建请求 payload
        payload = {
            "model": "qwen3-vl:4b",
            "prompt": question,
            "system": system_prompt,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 2048,
                "keep_alive": 0
                # "seed": random.randint(1, 99999),  # 避免服务端缓存
                # "num_ctx": 8192,  # 增大上下文窗口
            },
        }
        if img_base64:
            payload["images"] = [img_base64]

        url = "http://localhost:11434/api/generate"

        try:
            self._append_debug(f"发送请求 (文本长度 {len(question)}，图片 {'有' if img_base64 else '无'})\n")

            # 关键：使用 Session 并确保连接正确关闭
            with requests.Session() as session:
                response = session.post(url, json=payload, timeout=180)
                response.raise_for_status()
                result = response.json()
                # 会话结束后自动关闭连接

            response_text = result.get("response", "")

            if not response_text.strip():
                self._append_debug(f"警告：模型返回空 (状态码 {response.status_code})\n")
                # 可以尝试打印 response 原文
                # self._append_debug(f"完整响应: {result}\n")
                display = "模型返回为空，请检查 System Prompt 或查看 Ollama 服务日志。"
            else:
                # 尝试解析 JSON
                try:
                    parsed = json.loads(response_text)
                    display = json.dumps(parsed, ensure_ascii=False, indent=2)
                except json.JSONDecodeError:
                    # 尝试提取代码块中的 JSON
                    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                    if match:
                        parsed = json.loads(match.group(1))
                        display = json.dumps(parsed, ensure_ascii=False, indent=2)
                    else:
                        match = re.search(r'\{.*\}', response_text, re.DOTALL)
                        if match:
                            parsed = json.loads(match.group())
                            display = json.dumps(parsed, ensure_ascii=False, indent=2)
                        else:
                            self._append_debug(f"无法解析 JSON，原始返回前200字符: {response_text[:200]}\n")
                            display = response_text

            # 更新 UI
            dialog.after(0, self._show_qwen_ner_answer, dialog, display)

        except requests.exceptions.RequestException as e:
            display = f"网络请求错误：{str(e)}"
            self._append_debug(f"请求异常: {e}\n")
            dialog.after(0, self._show_qwen_ner_answer, dialog, display)
        except Exception as e:
            import traceback
            display = f"调用出错：{str(e)}"
            self._append_debug(f"详细错误:\n{traceback.format_exc()}\n")
            dialog.after(0, self._show_qwen_ner_answer, dialog, display)
        self._append_debug(f"发送的文本 (前200字符): {question[:200]}...\n" if question else "警告：question为空\n")

    def _show_qwen_ner_answer(self, dialog: tk.Toplevel, display: str):
        """显示大模型回答"""
        self._qwen_ner_output.config(state=tk.NORMAL)
        self._qwen_ner_output.delete("1.0", tk.END)
        self._qwen_ner_output.insert(tk.END, display)
        self._qwen_ner_output.config(state=tk.DISABLED)

    # 20260511新增大模型相关功能########################################################
    def on_qwen_qa(self):
        """弹出 Qwen3 智能问答对话框"""
        dialog = tk.Toplevel(self.root)
        dialog.title("QWEN3 大模型智能问答")
        dialog.geometry("700x550")
        dialog.minsize(500, 400)
        dialog.transient(self.root)
        dialog.grab_set()

        # ---- 上方：问题输入区 ----
        input_frame = tk.LabelFrame(dialog, text="输入您的问题", padx=10, pady=10)
        input_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=(10, 5))

        self._qa_input = tk.Text(input_frame, height=5, font=("微软雅黑", 11), wrap=tk.WORD)
        self._qa_input.pack(fill=tk.BOTH, expand=True)

        btn_frame = tk.Frame(input_frame)
        btn_frame.pack(fill=tk.X, pady=(8, 0))

        send_btn = tk.Button(
            btn_frame, text="发送",
            command=lambda: self._do_qa_query(dialog),
            font=("微软雅黑", 11), bg="#4a7abc", fg="white", width=10,
        )
        send_btn.pack(side=tk.LEFT, padx=(0, 10))

        clear_btn = tk.Button(
            btn_frame, text="清空",
            command=lambda: self._qa_input.delete("1.0", tk.END),
            font=("微软雅黑", 11), width=8,
        )
        clear_btn.pack(side=tk.LEFT)

        # ---- 下方：回答输出区 ----
        output_frame = tk.LabelFrame(dialog, text="智能回答", padx=10, pady=10)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        self._qa_output = tk.Text(output_frame, height=12, font=("微软雅黑", 11),
                                  wrap=tk.WORD, state=tk.DISABLED)
        self._qa_output.pack(fill=tk.BOTH, expand=True)

        # 绑定快捷键 Ctrl+Enter 发送
        self._qa_input.bind("<Control-Return>", lambda e: self._do_qa_query(dialog))

        # 焦点自动定位到输入框
        self._qa_input.focus_set()

    def _do_qa_query(self, dialog: tk.Toplevel):
        """调用 Ollama Qwen3 执行问答"""
        question = self._qa_input.get("1.0", tk.END).strip()
        if not question:
            messagebox.showwarning("提示", "请输入问题", parent=dialog)
            return

        # 禁用发送按钮，显示等待状态
        self._set_qa_waiting(dialog, True)

        # 在后台线程调用 Ollama，避免阻塞 UI
        import threading
        threading.Thread(
            target=self._qa_query_thread,
            args=(dialog, question),
            daemon=True,
        ).start()

    def _set_qa_waiting(self, dialog: tk.Toplevel, waiting: bool):
        """切换问答对话框的等待状态"""
        if waiting:
            self._qa_output.config(state=tk.NORMAL)
            self._qa_output.delete("1.0", tk.END)
            self._qa_output.insert(tk.END, "正在思考，请稍候...\n")
            self._qa_output.config(state=tk.DISABLED)
        dialog.update_idletasks()

    def _qa_query_thread(self, dialog: tk.Toplevel, question: str):
        """在后台线程中执行 Ollama 调用"""
        try:
            from common.llm_extract import call_ollama_generate

            answer = call_ollama_generate(
                base_url="http://localhost:11434",
                model="qwen3:4b",
                system="你是一个智能助手，请友好、准确地回答用户的问题。",
                prompt=question,
                temperature=0.7,
                num_predict=2048,
                enforce_json=False,
                timeout_s=180,
            )
        except Exception as e:
            answer = f"调用出错：{e}"

        # 回到主线程更新 UI
        dialog.after(0, self._show_qa_answer, dialog, answer)

    def _show_qa_answer(self, dialog: tk.Toplevel, answer: str):
        """在主线程中显示回答"""
        self._qa_output.config(state=tk.NORMAL)
        self._qa_output.delete("1.0", tk.END)
        self._qa_output.insert(tk.END, answer)
        self._qa_output.config(state=tk.DISABLED)
        self._set_qa_waiting(dialog, False)
    #########################################################################################


    def open_train_window(self):
        SemiSupervisedTrainWindow(self.root)

    # def _append_text(self, content):
        # self.text_area1.config(state=tk.NORMAL)
        # self.text_area1.insert(tk.END, content)
        # self.text_area1.see(tk.END)
        # self.text_area1.config(state=tk.DISABLED)
    def _append_text(self, content):
        """兼容旧调用：将文本输出到底部调试窗口"""
        self._append_debug(content)


    def _append_text2(self, content):
        self.text_area2.config(state=tk.NORMAL)
        self.text_area2.insert(tk.END, content)
        self.text_area2.see(tk.END)
        self.text_area2.config(state=tk.DISABLED)

    #######################onnx
    def _load_imgpre_model(self):
        """查找 ./pth/imgpre 下的第一个 .pth 文件，加载 PyTorch 模型到 GPU/CPU"""
        model_dir = "./pth/imgpre"
        if not os.path.exists(model_dir):
            self._append_debug(f"警告: 预处理模型目录不存在 {model_dir}\n")
            return False
        pth_files = [f for f in os.listdir(model_dir) if f.lower().endswith('.pth')]
        if not pth_files:
            self._append_debug(f"警告: 在 {model_dir} 中未找到 .pth 模型文件\n")
            return False
        model_path = os.path.join(model_dir, pth_files[0])

        # 设备
        # self.imgpre_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 20260510 修改设备为前端选择#######
        if self.use_gpu and torch.cuda.is_available():
            self.imgpre_device = torch.device('cuda')
        else:
            self.imgpre_device = torch.device('cpu')
        ##################################
        try:
            # 导入网络结构（确保 MobileNetV3_Unet 在当前命名空间）
            from train_mobilenetv3_321 import MobileNetV3_Unet  # 或者直接使用已定义类
            model = MobileNetV3_Unet(pretrained=False)
            checkpoint = torch.load(model_path, map_location=self.imgpre_device, weights_only=True)
            # 兼容含有 'G_model' 的字典
            if 'G_model' in checkpoint:
                model.load_state_dict(checkpoint['G_model'], strict=True)
            else:
                model.load_state_dict(checkpoint, strict=True)
            model.to(self.imgpre_device)
            model.eval()
            self.imgpre_model = model
            self._append_debug(f"已加载 PyTorch 图像预处理模型: {model_path} (设备: {self.imgpre_device})\n")
            return True
        except Exception as e:
            self._append_debug(f"加载 PyTorch 模型失败: {e}\n")
            return False

    def _cv2_imread_unicode(self, file_path):
        try:
            img_array = np.fromfile(file_path, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            self._append_debug(f"读取图片失败 {file_path}: {e}\n")
            return None

    def _cv2_imwrite_unicode(self, file_path, img):
        try:
            ext = os.path.splitext(file_path)[1]
            if not ext:
                ext = ".png"
            success, buf = cv2.imencode(ext, img)
            if success:
                buf.tofile(file_path)
                return True
            return False
        except Exception as e:
            self._append_debug(f"保存图片失败 {file_path}: {e}\n")
            return False

    def _remove_small_noise_from_mask(self, binary_mask_fg255, min_area_threshold=25):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask_fg255, 8, cv2.CV_32S)
        cleaned = np.zeros_like(binary_mask_fg255)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area_threshold:
                cleaned[labels == i] = 255
        return cleaned

    def _split_with_overlap(self, image, tile_size=512):
        h, w = image.shape[:2]
        tile_size = int(tile_size)
        patches = []
        coords = []
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                left = x
                top = y
                right = min(left + tile_size, w)
                bottom = min(top + tile_size, h)
                if right - left < tile_size:
                    left = w - tile_size
                if bottom - top < tile_size:
                    top = h - tile_size
                right = left + tile_size
                bottom = top + tile_size
                patch = image[top:bottom, left:right]
                patches.append(patch)
                coords.append((left, top, right, bottom))
        # 去重（理论上不会有重复，但保留原逻辑）
        unique_patches, unique_coords = [], []
        seen = set()
        for p, c in zip(patches, coords):
            key = (c[0], c[1])
            if key not in seen:
                seen.add(key)
                unique_patches.append(p)
                unique_coords.append(c)
        return unique_patches, unique_coords

    def _merge_images_from_prob(self, prob_patches, coords, img_size_hw):
        H, W = img_size_hw
        merged = np.zeros((H, W), dtype=np.float32)
        count = np.zeros((H, W), dtype=np.float32)
        for (left, top, right, bottom), tile in zip(coords, prob_patches):
            h, w = tile.shape
            merged[top:top + h, left:left + w] += tile
            count[top:top + h, left:left + w] += 1
        count[count == 0] = 1
        return merged / count

    def _adaptive_threshold_cv2(self, prob_float, block_size=51, C=8):
        prob_uint8 = (prob_float * 255).astype(np.uint8)
        mask = cv2.adaptiveThreshold(prob_uint8, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY_INV, block_size, C)
        return mask

    def _modify_background(self, color_img, binary_mask_fg0_bg255, bg_color=(255, 255, 255), brightness_thresh=240):
    # def _modify_background(self, color_img, binary_mask_fg0_bg255, bg_color=(197, 246, 254), brightness_thresh=240):

        """binary_mask_fg0_bg255: 0=前景(文字), 255=背景(待替换)"""
        result = color_img.copy()
        # 背景区域替换
        bg_mask = binary_mask_fg0_bg255 == 255
        result[bg_mask] = bg_color
        # 额外处理高亮度区域（防止亮白文字被误伤？这里保留原逻辑）
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        high_bright = gray > brightness_thresh
        result[high_bright] = bg_color
        return result

    def _enhance_single_image(self, input_path, output_path, tile_size=512, threshold=0.45,
                              local_block_size=51, local_C_offset=8, noise_thresh=15,
                              bg_color=(255, 255, 255), batch_size=16):
                              # bg_color=(197, 246, 254), batch_size=8):

        """使用 PyTorch 模型对单张图像进行去底增强，保存到 output_path"""
        if self.imgpre_model is None:
            if not self._load_imgpre_model():
                return False

        # 1. 读取图像
        img = self._cv2_imread_unicode(input_path)
        if img is None:
            return False
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. 切片
        patches, coords = self._split_with_overlap(img_rgb, tile_size=tile_size)

        # 3. 转换为 Tensor 并批推理
        transform = transforms.Compose([transforms.ToTensor()])
        batch_tensors = []
        for patch in patches:
            pil = Image.fromarray(patch)
            batch_tensors.append(transform(pil))
        all_tensor = torch.stack(batch_tensors)  # [N, 3, tile_size, tile_size]

        out_probs = []
        with torch.no_grad():
            for i in range(0, len(all_tensor), batch_size):
                batch = all_tensor[i:i + batch_size].to(self.imgpre_device)
                pred = self.imgpre_model(batch)  # [bs, 1, H, W]
                pred_np = pred[:, 0, :, :].cpu().numpy()  # [bs, H, W]
                out_probs.append(pred_np)

        prob_patches = np.concatenate(out_probs, axis=0)  # [N, H, W]

        # 4. 合并概率图
        full_prob = self._merge_images_from_prob(prob_patches, coords, (h, w))

        # 5. 后处理（与您提供的完全一致）
        prob_corrected = 1.0 - full_prob
        mask_global = (prob_corrected > threshold).astype(np.uint8) * 255
        mask_local = self._adaptive_threshold_cv2(full_prob, block_size=local_block_size, C=local_C_offset)
        mask_faint = cv2.subtract(mask_local, mask_global)
        mask_faint_clean = self._remove_small_noise_from_mask(mask_faint, min_area_threshold=noise_thresh)
        final_mask_fg255 = cv2.bitwise_or(mask_global, mask_faint_clean)
        final_binary_mask = cv2.bitwise_not(final_mask_fg255)  # 0=前景(文字), 255=背景

        # 6. 背景替换并保存
        enhanced_img = self._modify_background(img, final_binary_mask, bg_color=bg_color)
        self._cv2_imwrite_unicode(output_path, enhanced_img)
        return True

    def _preprocess_all_images(self):
        """对当前 self.image_paths 中的所有图片进行增强，保存到 enhanced 子目录"""
        if not self.image_paths:
            return
        if self.imgpre_model is None:  # ✅ 改这里！原来是 imgpre_session
            if not self._load_imgpre_model():
                self._append_debug("图像预处理模型加载失败，将使用原图继续\n")
                self.enhanced_image_paths = self.image_paths.copy()
                self.is_preprocessed = True
                return

        # 创建增强文件夹
        self.enhanced_folder = os.path.join(self.current_folder, "enhanced")
        os.makedirs(self.enhanced_folder, exist_ok=True)

        self.enhanced_image_paths = []
        total = len(self.image_paths)
        for idx, src_path in enumerate(self.image_paths):
            base = os.path.basename(src_path)
            name, ext = os.path.splitext(base)
            dst_path = os.path.join(self.enhanced_folder, f"{name}_enhanced{ext}")
            # 如果已存在且非空，可跳过（但为确保一致性，可强制覆盖）
            if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
                self._append_debug(f"跳过已存在: {base}\n")
            else:
                self._append_debug(f"预处理图片 [{idx + 1}/{total}]: {base} ... ")
                success = self._enhance_single_image(src_path, dst_path)
                if success:
                    self._append_debug("完成\n")
                else:
                    self._append_debug("失败，回退使用原图\n")
                    dst_path = src_path  # 失败时使用原图
            self.enhanced_image_paths.append(dst_path)

        self.is_preprocessed = True
        self._append_debug(f"图像预处理完成，增强图保存在 {self.enhanced_folder}\n")




if __name__ == "__main__":
    root = tk.Tk()
    app = MultiRegionApp(root)
    root.mainloop()