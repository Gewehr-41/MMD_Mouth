"""Tkinter front-end: lyrics -> rhythm matching -> VMD, outside Blender.

    .\\.venv-worker\\Scripts\\python.exe tools\\rhythm_vmd_gui.py

Left side is the timed lyric table (edit start/end/text per line), right side is
the raw lyric text.  "自动对齐" runs the CTC aligner, "生成 VMD" writes the
motion file next to an editable LRC.
"""

from __future__ import annotations

import queue
from pathlib import Path
import threading
import time
import traceback

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import winsound
except ImportError:  # pragma: no cover - only Windows has winsound
    winsound = None

import rhythm_vmd as core


class RhythmApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("MMD 歌词节奏匹配 → VMD")
        self.root.geometry("1180x720")
        self.rows: list[core.LyricRow] = []
        self.wav: Path | None = None
        self.duration = 0.0
        self.queue: queue.Queue = queue.Queue()
        self.busy = False
        self.playing = False
        self.play_offset = 0.0
        self.play_started = 0.0
        self.play_counter = 0
        self.current_line: int | None = None
        self.seeking = False
        self.follow_var = tk.BooleanVar(value=True)
        self.scale_var = tk.StringVar(value="1.00")
        self.min_var = tk.StringVar(value="0.00")
        self.max_var = tk.StringVar(value="1.00")
        self.limit_hint = tk.StringVar(value="")
        self.channel_limits: dict[str, tuple[float, float]] = {}
        self.channel_variants: dict[str, int] = {}

        self.audio_var = tk.StringVar()
        self.lyrics_var = tk.StringVar()
        self.out_var = tk.StringVar(value=str(Path("vmd_out").resolve()))
        self.name_var = tk.StringVar()
        self.model_var = tk.StringVar(
            value=str(core.DEFAULT_MODEL) if core.DEFAULT_MODEL.is_file() else ""
        )
        self.language_var = tk.StringVar(value="zh-CN")
        self.fps_var = tk.StringVar(value="30")
        self.status_var = tk.StringVar(value="就绪")

        self._build()
        self.root.after(120, self._poll)
        self.root.after(80, self._tick)

    # ---------------------------------------------------------------- layout
    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        # Packed first with side="bottom" so a tall table or log cannot squeeze
        # the status bar out of the window.
        status = ttk.Frame(outer)
        status.pack(side="bottom", fill="x", pady=(4, 0))
        self.row_status = tk.StringVar(value="未载入")
        ttk.Label(status, textvariable=self.row_status).pack(side="left")
        ttk.Label(
            status,
            text="双击改单元格 · 空格试听 · Del 删除 · Ctrl+S 保存 · Ctrl+O 载入 · Ctrl+Enter 生成",
            foreground="#888",
        ).pack(side="right")

        files = ttk.LabelFrame(outer, text="输入 / 输出", padding=6)
        files.pack(fill="x")
        self._file_row(files, 0, "音频", self.audio_var, self._pick_audio)
        self._file_row(files, 1, "歌词", self.lyrics_var, self._pick_lyrics)
        self._file_row(files, 2, "输出目录", self.out_var, self._pick_out, directory=True)

        ttk.Label(files, text="名称").grid(row=3, column=0, sticky="w")
        ttk.Entry(files, textvariable=self.name_var, width=24).grid(row=3, column=1, sticky="w")
        ttk.Label(files, text="模型").grid(row=3, column=2, sticky="e")
        model_entry = ttk.Entry(files, textvariable=self.model_var, width=52)
        model_entry.grid(row=3, column=3, sticky="we")
        ttk.Button(files, text="…", width=3, command=self._pick_model).grid(row=3, column=4)
        ttk.Label(files, text="语言").grid(row=4, column=0, sticky="w")
        ttk.Combobox(
            files,
            textvariable=self.language_var,
            values=("自动", "zh-CN", "ja-JP", "en-US"),
            width=10,
            state="readonly",
        ).grid(row=4, column=1, sticky="w")
        ttk.Label(files, text="FPS").grid(row=4, column=2, sticky="e")
        ttk.Entry(files, textvariable=self.fps_var, width=8).grid(row=4, column=3, sticky="w")
        files.columnconfigure(3, weight=1)

        actions = ttk.Frame(outer, padding=(0, 6))
        actions.pack(fill="x")
        ttk.Button(actions, text="载入歌词并分割", command=self.load_lyrics).pack(side="left")
        ttk.Button(actions, text="自动对齐（CTC）", command=self.run_align).pack(side="left", padx=4)
        ttk.Button(actions, text="生成 VMD", command=self.run_generate).pack(side="left", padx=4)
        ttk.Button(actions, text="保存 LRC", command=self.save_lrc).pack(side="left", padx=4)
        self.progress = ttk.Progressbar(actions, mode="indeterminate", length=140)
        self.progress.pack(side="right")
        ttk.Label(actions, textvariable=self.status_var).pack(side="right", padx=8)

        player = ttk.Frame(outer, padding=(0, 4))
        player.pack(fill="x")
        self.play_label = tk.StringVar(value="▶ 播放全曲")
        ttk.Button(player, textvariable=self.play_label, command=self.toggle_playback).pack(side="left")
        ttk.Button(player, text="■ 停止", command=self.stop_playback).pack(side="left", padx=4)
        self.time_var = tk.StringVar(value="00:00 / 00:00")
        ttk.Label(player, textvariable=self.time_var, width=15).pack(side="left", padx=6)
        self.seek = ttk.Scale(player, from_=0.0, to=1.0, command=self._on_seek_drag)
        self.seek.pack(side="left", fill="x", expand=True, padx=6)
        self.seek.bind("<ButtonPress-1>", self._begin_seek)
        self.seek.bind("<ButtonRelease-1>", self._commit_seek)
        ttk.Checkbutton(player, text="跟随滚动", variable=self.follow_var).pack(side="left")

        limits_row = ttk.Frame(outer, padding=(0, 2))
        limits_row.pack(fill="x")
        ttk.Label(limits_row, text="开口缩放").pack(side="left")
        ttk.Spinbox(
            limits_row, textvariable=self.scale_var, from_=0.2, to=1.5, increment=0.05, width=6
        ).pack(side="left", padx=(2, 10))
        ttk.Label(limits_row, text="下限").pack(side="left")
        ttk.Spinbox(
            limits_row, textvariable=self.min_var, from_=0.0, to=0.5, increment=0.05, width=6
        ).pack(side="left", padx=(2, 10))
        ttk.Label(limits_row, text="上限").pack(side="left")
        ttk.Spinbox(
            limits_row, textvariable=self.max_var, from_=0.1, to=1.0, increment=0.05, width=6
        ).pack(side="left", padx=(2, 10))
        ttk.Button(limits_row, text="每通道…", command=self.edit_channel_limits).pack(side="left")
        ttk.Label(limits_row, textvariable=self.limit_hint, foreground="#888").pack(
            side="left", padx=8
        )

        body = ttk.Panedwindow(outer, orient="horizontal")
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body)
        columns = ("index", "start", "end", "length", "text")
        table_frame = ttk.Frame(left)
        table_frame.pack(fill="both", expand=True)
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings", height=18)
        for key, title, width, stretch in (
            ("index", "#", 40, False),
            ("start", "开始(s)", 80, False),
            ("end", "结束(s)", 80, False),
            ("length", "时长", 60, False),
            ("text", "歌词", 300, True),
        ):
            self.table.heading(key, text=title)
            self.table.column(
                key,
                width=width,
                minwidth=40,
                stretch=stretch,
                anchor="w" if key == "text" else "center",
            )
        self.table_scroll = ttk.Scrollbar(
            table_frame, orient="vertical", command=self.table.yview
        )
        self.table.configure(yscrollcommand=self.table_scroll.set)
        self.table.pack(side="left", fill="both", expand=True)
        self.table_scroll.pack(side="right", fill="y")
        self.table.bind("<<TreeviewSelect>>", self._on_select)
        self.table.bind("<space>", self._on_space)
        self.table.bind("<Double-1>", self._on_double_click)
        self.table.bind("<Button-3>", self._show_context_menu)
        self.table.bind("<Delete>", lambda _event: self.delete_row())
        self.table.tag_configure("playing", background="#2f5d8a")
        body.add(left, weight=3)

        edit = ttk.LabelFrame(left, text="编辑选中行", padding=6)
        edit.pack(fill="x", pady=(6, 0))
        self.start_var = tk.StringVar()
        self.end_var = tk.StringVar()
        self.text_var = tk.StringVar()
        ttk.Label(edit, text="开始").grid(row=0, column=0)
        ttk.Entry(edit, textvariable=self.start_var, width=9).grid(row=0, column=1)
        ttk.Label(edit, text="结束").grid(row=0, column=2)
        ttk.Entry(edit, textvariable=self.end_var, width=9).grid(row=0, column=3)
        ttk.Label(edit, text="歌词").grid(row=0, column=4)
        ttk.Entry(edit, textvariable=self.text_var, width=30).grid(row=0, column=5, sticky="we")
        ttk.Button(edit, text="应用", command=self.apply_edit).grid(row=0, column=6, padx=4)
        ttk.Button(edit, text="删除行", command=self.delete_row).grid(row=0, column=7)
        ttk.Button(edit, text="跳到本行", command=self.seek_to_selected).grid(
            row=0, column=9, padx=(6, 0)
        )
        edit.columnconfigure(5, weight=1)

        nudge = ttk.Frame(edit)
        nudge.grid(row=0, column=8, padx=(6, 0))
        ttk.Button(nudge, text="尾 −0.1", width=7, command=lambda: self.nudge_end(-0.1)).pack()
        ttk.Button(nudge, text="尾 +0.1", width=7, command=lambda: self.nudge_end(0.1)).pack()

        self.auto_preview = tk.BooleanVar(value=False)
        self.refine_var = tk.BooleanVar(value=True)
        self.blend_var = tk.BooleanVar(value=True)
        self.dense_var = tk.BooleanVar(value=True)
        self.style_var = tk.StringVar(value="经典")
        self.preview_counter = 0
        preview = ttk.Frame(edit)
        preview.grid(row=1, column=0, columnspan=8, sticky="we", pady=(6, 0))
        ttk.Button(preview, text="▶ 试听本行", command=self.play_selected).pack(side="left")
        ttk.Button(preview, text="■ 停止", command=self.stop_preview).pack(side="left", padx=4)
        ttk.Checkbutton(
            preview,
            text="选中即试听",
            variable=self.auto_preview,
        ).pack(side="left", padx=8)
        ttk.Checkbutton(
            preview,
            text="逐句精对齐",
            variable=self.refine_var,
        ).pack(side="left", padx=8)
        ttk.Checkbutton(
            preview,
            text="合成口型",
            variable=self.blend_var,
        ).pack(side="left", padx=4)
        ttk.Checkbutton(
            preview,
            text="逐帧关键帧",
            variable=self.dense_var,
        ).pack(side="left", padx=4)
        ttk.Label(preview, text="风格").pack(side="left", padx=(8, 2))
        ttk.Combobox(
            preview,
            textvariable=self.style_var,
            values=("经典", "自然"),
            width=6,
            state="readonly",
        ).pack(side="left")
        ttk.Label(preview, text="（表格里按空格也可以试听）").pack(side="left")

        right = ttk.LabelFrame(body, text="歌词原文（可直接修改）", padding=6)
        text_frame = ttk.Frame(right)
        text_frame.pack(fill="both", expand=True)
        self.text_area = tk.Text(text_frame, wrap="word", undo=True, width=34)
        self.text_scroll = ttk.Scrollbar(
            text_frame, orient="vertical", command=self.text_area.yview
        )
        self.text_area.configure(yscrollcommand=self.text_scroll.set)
        self.text_area.pack(side="left", fill="both", expand=True)
        self.text_scroll.pack(side="right", fill="y")
        ttk.Button(right, text="按文本重建行（重新分配时间）", command=self.rebuild_from_text).pack(
            fill="x", pady=4
        )
        body.add(right, weight=2)

        log = ttk.LabelFrame(outer, text="日志", padding=4)
        log.pack(fill="both", expand=False)
        self.log = tk.Text(log, height=5, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True)

    def _file_row(self, parent, row, label, variable, command, directory=False) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Entry(parent, textvariable=variable, width=52).grid(
            row=row, column=1, columnspan=3, sticky="we", padx=4
        )
        ttk.Button(
            parent,
            text="…",
            width=3,
            command=lambda: command(directory),
        ).grid(row=row, column=4)

    # ------------------------------------------------------------- utilities
    def _log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")
        self.status_var.set(message)

    def _thread(self, target) -> None:
        if self.busy:
            messagebox.showinfo("请稍候", "上一个任务还在运行")
            return
        self.busy = True
        self.progress.start(12)

        def runner():
            try:
                target()
            except Exception as exc:  # noqa: BLE001 - surface everything to the UI
                self.queue.put(("error", f"{exc}\n{traceback.format_exc()}"))
            finally:
                self.queue.put(("done", ""))

        threading.Thread(target=runner, daemon=True).start()

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "error":
                    self.progress.stop()
                    self.busy = False
                    self._log("失败：" + payload.splitlines()[0])
                    messagebox.showerror("出错", payload[:1200])
                elif kind == "rows":
                    self.rows = payload
                    self._refresh_rows()
                    self._fill_text()
                    self._sync_scale()
                    if self.rows:
                        self.table.selection_set("0")
                        self.table.focus("0")
                        self._on_select()
                elif kind == "done":
                    self.progress.stop()
                    self.busy = False
        except queue.Empty:
            pass
        self.root.after(120, self._poll)

    def _refresh_rows(self) -> None:
        self.current_line = None
        self.table.delete(*self.table.get_children())
        for index, row in enumerate(self.rows):
            self.table.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    index + 1,
                    f"{row.start:.2f}",
                    f"{row.end:.2f}",
                    f"{row.duration:.2f}",
                    row.text,
                ),
            )
        self._update_row_status()

    def _update_row_status(self) -> None:
        index = self._selected()
        if index is None:
            self.row_status.set(f"共 {len(self.rows)} 行")
            return
        row = self.rows[index]
        self.row_status.set(
            f"第 {index + 1} / {len(self.rows)} 行 · "
            f"{row.start:.2f}–{row.end:.2f}s（{row.duration:.2f}s）"
        )

    def _fill_text(self) -> None:
        self.text_area.delete("1.0", "end")
        self.text_area.insert("1.0", "\n".join(row.text for row in self.rows))

    def _language(self) -> str:
        """Selected language, with 自动 meaning detect per character."""

        value = self.language_var.get()
        return "" if value == "自动" else value

    def _selected(self) -> int | None:
        selection = self.table.selection()
        return int(selection[0]) if selection else None

    # ---------------------------------------------------------------- actions
    def _pick_audio(self, _directory=False) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("音频", "*.m4a *.mp3 *.wav *.flac *.ogg"), ("全部", "*.*")]
        )
        if path:
            self.audio_var.set(path)

    def _pick_lyrics(self, _directory=False) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("歌词", "*.txt *.lrc *.json"), ("全部", "*.*")]
        )
        if path:
            self.lyrics_var.set(path)

    def _pick_out(self, directory=False) -> None:
        path = filedialog.askdirectory()
        if path:
            self.out_var.set(path)

    def _pick_model(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("ONNX", "*.onnx"), ("全部", "*.*")])
        if path:
            self.model_var.set(path)

    def load_lyrics(self) -> None:
        lyrics_path = Path(self.lyrics_var.get().strip())
        if not lyrics_path.is_file():
            messagebox.showerror("缺少歌词", "请先选择歌词文件")
            return

        def work():
            audio = self.audio_var.get().strip()
            duration = 0.0
            if audio:
                self.queue.put(("log", "准备音频…"))
                self.wav, duration = core.prepare_audio(
                    audio, Path(self.out_var.get()) / "_work"
                )
                self.duration = duration
            texts = core.split_text(lyrics_path.read_text(encoding="utf-8-sig"))
            self.queue.put(("log", f"分割出 {len(texts)} 行"))
            self.queue.put(("rows", core.uniform_rows(texts, 0.0, duration or len(texts) * 3.0)))

        self._thread(work)

    def run_align(self) -> None:
        lyrics_path = Path(self.lyrics_var.get().strip())
        audio = self.audio_var.get().strip()
        if not lyrics_path.is_file() or not audio:
            messagebox.showerror("缺少输入", "需要音频和歌词文件")
            return

        def work():
            if self.wav is None:
                self.queue.put(("log", "准备音频…"))
                self.wav, duration = core.prepare_audio(
                    audio, Path(self.out_var.get()) / "_work"
                )
                self.duration = duration
            texts = core.split_text(lyrics_path.read_text(encoding="utf-8-sig"))
            rows = core.align_rows(
                self.wav,
                texts,
                model=Path(self.model_var.get())
                if self.model_var.get()
                else core.DEFAULT_MODEL,
                duration=self.duration,
                refine=self.refine_var.get(),
                language=self._language(),
                progress=lambda message: self.queue.put(("log", message)),
            )
            self.queue.put(("rows", rows))
            self.queue.put(("log", f"对齐完成：{len(rows)} 行"))

        self._thread(work)

    def run_generate(self) -> None:
        if not self.rows:
            messagebox.showerror("没有歌词行", "请先载入并分割歌词")
            return
        out_dir = Path(self.out_var.get().strip() or "vmd_out")
        lyrics_path = Path(self.lyrics_var.get().strip())
        name = self.name_var.get().strip() or (lyrics_path.stem if lyrics_path.stem else "mouth")

        def work():
            language = self._language() or "zh-CN"
            words = core.rows_to_words(self.rows, language)
            self.queue.put(("log", f"{len(words)} 个词"))
            document = core.build_events(
                words, language=language, blend=self.blend_var.get()
            )
            curves = core.sample_curves(
                document.events,
                self.duration,
                fps=float(self.fps_var.get()),
                dense=self.dense_var.get(),
                style="NATURAL" if self.style_var.get() == "自然" else "CLASSIC",
                scale=self._number(self.scale_var, 1.0),
                limits=self._limits_for_generate(),
                variants=self._variants_for_generate(),
            )
            from vmd_writer import write_vmd

            out_dir.mkdir(parents=True, exist_ok=True)
            vmd_path = out_dir / f"{name}.vmd"
            keys = write_vmd(vmd_path, curves)
            core.write_lrc(self.rows, out_dir / f"{name}.lrc")
            core.write_align_json(self.rows, out_dir / f"{name}.align.json", self.duration, language)
            self.queue.put(("log", f"已写出 {vmd_path.name}（{keys} 个关键帧）"))

        self._thread(work)

    def apply_edit(self) -> None:
        index = self._selected()
        if index is None:
            messagebox.showinfo("未选中", "先在表格里选一行")
            return
        row = self.rows[index]
        try:
            start = float(self.start_var.get())
            end = float(self.end_var.get())
        except ValueError:
            messagebox.showerror("数值错误", "开始/结束必须是数字")
            return
        if end <= start:
            messagebox.showerror("时间错误", "结束必须大于开始")
            return
        row.start, row.end = start, end
        row.text = self.text_var.get().strip() or row.text
        row.edited = True
        row.spans = []
        self._refresh_rows()
        self._log(f"已更新第 {index + 1} 行")

    def nudge_end(self, delta: float) -> None:
        """Fine-tune the selected line's end by a small step."""

        index = self._selected()
        if index is None:
            messagebox.showinfo("未选中", "先在表格里选一行")
            return
        row = self.rows[index]
        row.end = max(row.start + 0.05, row.end + delta)
        row.edited = True
        row.spans = []
        self.end_var.set(f"{row.end:.2f}")
        self._refresh_rows()
        self._log(f"第 {index + 1} 行句尾 → {row.end:.2f}s")

    def delete_row(self) -> None:
        index = self._selected()
        if index is None:
            return
        del self.rows[index]
        self._refresh_rows()
        self._fill_text()
        if self.rows:
            target = min(index, len(self.rows) - 1)
            self.table.selection_set(str(target))
            self._on_select()

    def insert_row(self) -> None:
        """Insert an empty line right after the selection."""

        index = self._selected()
        if index is None:
            if not self.rows:
                return
            index = len(self.rows) - 1
        row = self.rows[index]
        start = row.end
        new = core.LyricRow(start, start + 1.0, "")
        self.rows.insert(index + 1, new)
        self._refresh_rows()
        self._fill_text()
        self.table.selection_set(str(index + 1))
        self._on_select()

    def merge_with_previous(self) -> None:
        index = self._selected()
        if index is None or index == 0:
            return
        previous, row = self.rows[index - 1], self.rows[index]
        previous.text = (previous.text + row.text).strip()
        previous.end = max(row.end, previous.start + 0.05)
        previous.edited = True
        previous.spans = []
        del self.rows[index]
        self._refresh_rows()
        self._fill_text()
        self.table.selection_set(str(index - 1))
        self._on_select()

    def split_row(self) -> None:
        """Split the selected line in half (text and time)."""

        index = self._selected()
        if index is None:
            return
        row = self.rows[index]
        text = row.text.strip()
        if len(text) < 2:
            messagebox.showinfo("无法拆分", "这一行太短了")
            return
        middle = (row.start + row.end) / 2
        cut = max(1, len(text) // 2)
        left = core.LyricRow(row.start, middle, text[:cut])
        right = core.LyricRow(middle, row.end, text[cut:])
        for piece in (left, right):
            piece.edited = True
            piece.spans = []
        self.rows[index : index + 1] = [left, right]
        self._refresh_rows()
        self._fill_text()
        self.table.selection_set(str(index))
        self._on_select()

    def seek_to_selected(self) -> None:
        index = self._selected()
        if index is None:
            return
        self.start_playback(self.rows[index].start)

    # ------------------------------------------------------- inline editing
    EDITABLE_COLUMNS = {"#2": "start", "#3": "end", "#5": "text"}

    def _on_double_click(self, event) -> None:
        if self.table.identify("region", event.x, event.y) != "cell":
            return
        row = self.table.identify_row(event.y)
        column = self.table.identify_column(event.x)
        if not row:
            return
        if column not in self.EDITABLE_COLUMNS:
            # Index/duration columns are read-only: double-click previews instead.
            self.play_selected()
            return
        self._edit_cell(int(row), column)

    def _edit_cell(self, row_index: int, column: str) -> None:
        bbox = self.table.bbox(str(row_index), column)
        if not bbox:
            return
        x, y, width, height = bbox
        current = self.table.set(str(row_index), column)
        entry = ttk.Entry(self.table)
        entry.insert(0, current)
        entry.select_range(0, "end")
        entry.place(x=x, y=y, width=width, height=height)
        entry.focus_set()

        def finish(_event=None) -> None:
            text = entry.get().strip()
            entry.destroy()
            self._commit_cell(row_index, column, text)

        def cancel(_event=None) -> None:
            entry.destroy()

        entry.bind("<Return>", finish)
        entry.bind("<KP_Enter>", finish)
        entry.bind("<Escape>", cancel)
        entry.bind("<FocusOut>", finish)

    def _commit_cell(self, row_index: int, column: str, text: str) -> None:
        if row_index >= len(self.rows):
            return
        row = self.rows[row_index]
        if column == "#5":
            row.text = text
        else:
            try:
                value = float(text)
            except ValueError:
                messagebox.showerror("数值错误", "时间必须是数字")
                return
            if column == "#2":
                if value >= row.end:
                    messagebox.showerror("时间错误", "开始必须早于结束")
                    return
                row.start = value
            else:
                if value <= row.start:
                    messagebox.showerror("时间错误", "结束必须晚于开始")
                    return
                row.end = value
        row.edited = True
        row.spans = []
        self._refresh_rows()
        self._fill_text()
        self.table.selection_set(str(row_index))
        self._on_select()

    def _show_context_menu(self, event) -> None:
        row = self.table.identify_row(event.y)
        if row:
            self.table.selection_set(row)
            self._on_select()
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="试听本行（空格）", command=self.play_selected)
        menu.add_command(label="跳到本行", command=self.seek_to_selected)
        menu.add_separator()
        menu.add_command(label="插入空行", command=self.insert_row)
        menu.add_command(label="拆分本行", command=self.split_row)
        menu.add_command(label="合并到上一行", command=self.merge_with_previous)
        menu.add_command(label="删除行（Del）", command=self.delete_row)
        menu.add_separator()
        menu.add_command(label="载入歌词并分割（Ctrl+O）", command=self.load_lyrics)
        menu.add_command(label="生成 VMD（Ctrl+Enter）", command=self.run_generate)
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def rebuild_from_text(self) -> None:
        texts = core.split_text(self.text_area.get("1.0", "end"))
        if not texts:
            return
        if self.rows and self.duration > 0.0:
            keeps = [row for row in self.rows if row.text in texts]
            if len(keeps) == len(texts):
                mapping = {row.text: row for row in keeps}
                new_rows = [mapping[text] for text in texts]
            else:
                new_rows = core.uniform_rows(texts, 0.0, self.duration)
        else:
            new_rows = core.uniform_rows(texts, 0.0, len(texts) * 3.0)
        self.rows = new_rows
        self._refresh_rows()
        self._log(f"按文本重建 {len(new_rows)} 行")

    def save_lrc(self) -> None:
        if not self.rows:
            return
        out_dir = Path(self.out_var.get().strip() or "vmd_out")
        lyrics_path = Path(self.lyrics_var.get().strip())
        name = self.name_var.get().strip() or lyrics_path.stem or "mouth"
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"{name}.lrc"
        core.write_lrc(self.rows, target)
        self._log(f"已保存 {target}")

    # ------------------------------------------------------------ morph limits
    @staticmethod
    def _number(variable, default: float) -> float:
        try:
            return float(str(variable.get()).strip())
        except (TypeError, ValueError):
            return default

    def _limits_for_generate(self) -> dict[str, tuple[float, float]]:
        low = self._number(self.min_var, 0.0)
        high = self._number(self.max_var, 1.0)
        if high < low:
            low, high = high, low
        limits = {channel: (low, high) for channel in core.CHANNELS}
        limits.update(self.channel_limits)
        return limits

    def _variants_for_generate(self) -> dict[str, int]:
        return {
            channel: core.normalize_variant(self.channel_variants.get(channel, 1))
            for channel in core.CHANNELS
        }

    def edit_channel_limits(self) -> None:
        top = tk.Toplevel(self.root)
        top.title("每通道：变体 / 上下限")
        top.transient(self.root)
        entries = {}
        ttk.Label(top, text="变体").grid(row=0, column=1)
        ttk.Label(top, text="下限").grid(row=0, column=2)
        ttk.Label(top, text="上限").grid(row=0, column=3)
        for row, channel in enumerate(core.CHANNELS, start=1):
            morph = core.MORPH_BY_CHANNEL[channel]
            low, high = self._limits_for_generate()[channel]
            variant = core.normalize_variant(self.channel_variants.get(channel, 1))
            ttk.Label(top, text=morph, width=4).grid(row=row, column=0, padx=6, pady=3)
            variant_var = tk.StringVar(value=str(variant))
            ttk.Combobox(
                top,
                textvariable=variant_var,
                values=("1", "2", "3"),
                width=4,
                state="readonly",
            ).grid(row=row, column=1, padx=2)
            low_var = tk.StringVar(value=f"{low:.2f}")
            high_var = tk.StringVar(value=f"{high:.2f}")
            ttk.Entry(top, textvariable=low_var, width=7).grid(row=row, column=2)
            ttk.Entry(top, textvariable=high_var, width=7).grid(row=row, column=3)
            entries[channel] = (variant_var, low_var, high_var)

        def apply() -> None:
            for channel, (variant_var, low_var, high_var) in entries.items():
                self.channel_variants[channel] = core.normalize_variant(variant_var.get())
                low = self._number(low_var, 0.0)
                high = self._number(high_var, 1.0)
                if high < low:
                    low, high = high, low
                self.channel_limits[channel] = (max(0.0, low), min(1.0, high))
            self._refresh_limit_hint()
            top.destroy()

        def clear() -> None:
            self.channel_limits.clear()
            self.channel_variants.clear()
            self._refresh_limit_hint()
            top.destroy()

        ttk.Button(top, text="应用", command=apply).grid(
            row=6, column=0, columnspan=4, pady=8, sticky="we"
        )
        ttk.Button(top, text="恢复默认（全 1 / 0-1）", command=clear).grid(
            row=7, column=0, columnspan=4, pady=(0, 8), sticky="we"
        )

    def _refresh_limit_hint(self) -> None:
        variants = self._variants_for_generate()
        active = [f"{core.MORPH_BY_CHANNEL[c]}{'' if v == 1 else v}" for c, v in variants.items()]
        self.limit_hint.set("变体：" + " ".join(active))

    # ------------------------------------------------------------- full playback
    @staticmethod
    def _fmt(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

    def toggle_playback(self) -> None:
        if self.playing:
            self.pause_playback()
        else:
            self.start_playback()

    def _sync_scale(self) -> None:
        """Keep the seek bar range in step with the real audio duration."""

        target = max(1.0, float(self.duration))
        try:
            if abs(float(self.seek.cget("to")) - target) > 0.01:
                self.seek.configure(to=target)
        except tk.TclError:
            pass
        self.time_var.set(f"{self._fmt(self.play_offset)} / {self._fmt(self.duration)}")

    def start_playback(self, offset: float | None = None) -> None:
        if winsound is None:
            messagebox.showerror("不支持播放", "当前环境没有 winsound")
            return
        if not self._ensure_audio():
            return
        self._sync_scale()
        position = self.play_offset if offset is None else float(offset)
        if position >= self.duration - 0.05:
            position = 0.0
        out_dir = Path(self.out_var.get().strip() or "vmd_out")
        self.play_counter += 1
        target = out_dir / "_work" / f"playback_{self.play_counter % 2}.wav"
        try:
            core.slice_wav(self.wav, position, self.duration, target)
            winsound.PlaySound(str(target), winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("播放失败", str(exc))
            return
        self.play_offset = position
        self.play_started = time.perf_counter()
        self.playing = True
        self.play_label.set("⏸ 暂停")
        self._log(f"播放中，从 {self._fmt(position)} 开始")

    def pause_playback(self) -> None:
        if not self.playing:
            return
        elapsed = time.perf_counter() - self.play_started
        self.play_offset = min(self.duration, self.play_offset + elapsed)
        if winsound is not None:
            winsound.PlaySound(None, winsound.SND_PURGE)
        self.playing = False
        self.play_label.set("▶ 继续播放")
        self._log(f"已暂停于 {self._fmt(self.play_offset)}")

    def stop_playback(self) -> None:
        if winsound is not None:
            winsound.PlaySound(None, winsound.SND_PURGE)
        self.playing = False
        self.play_offset = 0.0
        self.play_label.set("▶ 播放全曲")
        self._update_play_ui(0.0)
        self._highlight_line(None)

    def _begin_seek(self, _event=None) -> None:
        self.seeking = True

    def _on_seek_drag(self, value) -> None:
        try:
            position = float(value)
        except (TypeError, ValueError):
            return
        self.time_var.set(f"{self._fmt(position)} / {self._fmt(self.duration)}")

    def _commit_seek(self, _event=None) -> None:
        self.seeking = False
        try:
            position = float(self.seek.get())
        except (TypeError, ValueError):
            return
        if self.playing:
            if winsound is not None:
                winsound.PlaySound(None, winsound.SND_PURGE)
            self.start_playback(position)
        else:
            self.play_offset = position

    def _tick(self) -> None:
        if self.playing:
            elapsed = time.perf_counter() - self.play_started
            position = self.play_offset + elapsed
            if position >= self.duration - 0.02:
                self.stop_playback()
            else:
                self._update_play_ui(position)
        self.root.after(80, self._tick)

    def _update_play_ui(self, position: float) -> None:
        self._sync_scale()
        self.time_var.set(f"{self._fmt(position)} / {self._fmt(self.duration)}")
        if self.duration > 0.0 and not self.seeking:
            self.seek.set(min(position, self.duration))
        index = None
        for position_index, row in enumerate(self.rows):
            if row.start <= position < row.end:
                index = position_index
                break
            if position_index + 1 < len(self.rows) and row.end <= position < self.rows[position_index + 1].start:
                index = None
                break
        self._highlight_line(index)

    def _highlight_line(self, index: int | None) -> None:
        if index == self.current_line:
            return
        self.current_line = index
        for item in self.table.get_children():
            self.table.item(item, tags=())
        if index is None or index >= len(self.rows):
            return
        item = str(index)
        if self.table.exists(item):
            self.table.item(item, tags=("playing",))
            if self.follow_var.get():
                self.table.see(item)

    # ------------------------------------------------------------- previewing
    def _on_space(self, _event=None):
        self.play_selected()
        return "break"

    def _ensure_audio(self) -> bool:
        if self.wav is not None and Path(self.wav).is_file():
            return True
        audio = self.audio_var.get().strip()
        if not audio:
            messagebox.showerror("缺少音频", "请先选择音频文件")
            return False
        try:
            out_dir = Path(self.out_var.get().strip() or "vmd_out")
            self.wav, self.duration = core.prepare_audio(audio, out_dir / "_work")
        except Exception as exc:  # noqa: BLE001 - report to the user
            messagebox.showerror("音频读取失败", str(exc))
            return False
        self._log(f"音频已就绪：{self.duration:.2f}s")
        self.seek.configure(to=max(1.0, self.duration))
        return True

    def play_selected(self) -> None:
        index = self._selected()
        if index is None:
            self._log("先选一行再试听")
            return
        if winsound is None:
            messagebox.showerror("不支持试听", "当前环境没有 winsound")
            return
        if not self._ensure_audio():
            return
        row = self.rows[index]
        start = max(0.0, row.start - 0.2)
        tail = 0.2
        if index + 1 < len(self.rows):
            gap = self.rows[index + 1].start - row.end
            # Do not spill into the next line when the gap is small.
            tail = max(0.0, min(0.2, gap / 2.0))
        end = row.end + tail
        if self.duration > 0.0:
            end = min(end, self.duration)
        if end <= start:
            end = start + 0.4
        out_dir = Path(self.out_var.get().strip() or "vmd_out")
        self.preview_counter += 1
        target = out_dir / "_work" / f"preview_{self.preview_counter % 4}.wav"
        try:
            core.slice_wav(self.wav, start, end, target)
            winsound.PlaySound(str(target), winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("试听失败", str(exc))
            return
        self._log(f"试听第 {index + 1} 行：{start:.2f}–{end:.2f}s  {row.text}")

    def stop_preview(self) -> None:
        if winsound is not None:
            winsound.PlaySound(None, winsound.SND_PURGE)
        self._log("停止试听")

    # -------------------------------------------------------------- selection
    def _on_select(self, _event=None) -> None:
        index = self._selected()
        if index is None:
            self._update_row_status()
            return
        row = self.rows[index]
        self.start_var.set(f"{row.start:.2f}")
        self.end_var.set(f"{row.end:.2f}")
        self.text_var.set(row.text)
        self._update_row_status()
        if self.auto_preview.get():
            self.root.after(150, self.play_selected)

    # ------------------------------------------------------------ shortcuts
    def _bind_shortcuts(self) -> None:
        bindings = {
            "<Control-s>": lambda _event: self.save_lrc(),
            "<Control-S>": lambda _event: self.save_lrc(),
            "<Control-o>": lambda _event: self.load_lyrics(),
            "<Control-O>": lambda _event: self.load_lyrics(),
            "<Control-Return>": lambda _event: self.run_generate(),
            "<Control-KP_Enter>": lambda _event: self.run_generate(),
            "<Escape>": lambda _event: self.stop_preview(),
        }
        for sequence, handler in bindings.items():
            self.root.bind(sequence, handler)


    # ------------------------------------------------------------ model setup
    def ensure_model(self) -> None:
        """Make sure an aligner model exists, offering a first-run download."""

        try:
            import tool_support
        except ImportError:
            return  # running from an older checkout without the helper
        found = tool_support.model_path()
        if found is not None:
            self.model_var.set(str(found))
            core.DEFAULT_MODEL = found
            return
        target = tool_support.default_model_path()
        answer = messagebox.askyesno(
            "首次运行",
            "歌词对齐需要一个语音模型（约 1.2 GB，CC-BY-NC 许可）。\n\n"
            f"下载位置：\n{target}\n\n"
            "现在下载吗？（也可以稍后手动放入该文件夹）",
        )
        if answer:
            self._download_model(tool_support)

    def _download_model(self, tool_support) -> None:
        window = tk.Toplevel(self.root)
        window.title("下载语音模型")
        window.transient(self.root)
        window.resizable(False, False)
        ttk.Label(
            window,
            text="正在下载 mms_fa.onnx（约 1.2 GB），只需一次。",
        ).pack(padx=16, pady=(14, 6))
        bar = ttk.Progressbar(window, length=480, maximum=100.0)
        bar.pack(padx=16, pady=6)
        info = tk.StringVar(value="连接中…")
        ttk.Label(window, textvariable=info).pack(padx=16)
        state = {"stop": False, "last": 0.0}

        def progress(done: int, total: int, _url: str) -> None:
            now = time.time()
            if now - state["last"] < 0.25 and done < total:
                return
            state["last"] = now
            percent = 100.0 * done / max(total, 1)
            message = f"{done / 1048576:.1f} / {total / 1048576:.1f} MB（{percent:.1f}%）"
            self.root.after(
                0,
                lambda p=percent, m=message: (bar.configure(value=p), info.set(m)),
            )

        def work() -> None:
            try:
                path = tool_support.download_model(
                    progress=progress, cancel=lambda: state["stop"]
                )
            except Exception as error:  # noqa: BLE001 - shown in a dialog
                text = str(error)
                self.root.after(0, lambda t=text: self._model_failed(window, t))
                return
            self.root.after(0, lambda p=path: self._model_ready(window, p))

        ttk.Button(window, text="取消", command=lambda: state.update(stop=True)).pack(
            pady=(6, 12)
        )
        threading.Thread(target=work, daemon=True).start()

    def _model_ready(self, window, path) -> None:
        self.model_var.set(str(path))
        core.DEFAULT_MODEL = path
        window.destroy()
        self._log(f"模型已就绪：{path}")
        try:
            import tool_support

            settings = tool_support.load_settings()
            settings["model"] = str(path)
            tool_support.save_settings(settings)
        except Exception:  # noqa: BLE001
            pass
        messagebox.showinfo("完成", "模型下载完成，可以开始对齐了。")

    def _model_failed(self, window, message: str) -> None:
        window.destroy()
        messagebox.showerror("下载失败", message)


def main() -> int:
    root = tk.Tk()
    try:
        root.style = ttk.Style()
        root.style.theme_use("vista")
    except tk.TclError:
        pass
    root.geometry("1320x880")
    root.minsize(1040, 700)
    app = RhythmApp(root)
    app._bind_shortcuts()
    root.after(200, app.ensure_model)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
