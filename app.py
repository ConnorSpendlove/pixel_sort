# PixelSort Studio 
# Final Project CS 3000
# Connor Spendlove - Abbie Pitts - Ethan Thompson

import colorsys
import math
import os
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

import numpy as np
from PIL import Image, ImageOps, ImageTk


def next_power_of_two(value: int) -> int:
    # Bitonic sort works cleanly with list sizes that are powers of two.
    # If the row length is not a power of two, we pad to the next one.
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def pixel_key(rgb, mode: str) -> float:
    # Convert RGB pixel into HSV so we can sort by hue/saturation/value.
    # For brightness mode we use perceptual luminance instead.
    r, g, b = rgb
    rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(rn, gn, bn)

    if mode == "hue":
        return h
    if mode == "saturation":
        return s
    if mode == "value":
        return v

    # Perceived brightness (Rec. 709)
    return 0.2126 * rn + 0.7152 * gn + 0.0722 * bn


def bitonic_sort_pairs(pairs):
    # Bitonic sort (sorting network style) with compare-and-swap stages.
    n = len(pairs)
    m = next_power_of_two(n)
    sentinel = (float("inf"), (0, 0, 0))
    # Pad with "infinite" keys so extra values sink to the end.
    arr = list(pairs) + [sentinel] * (m - n)

    k = 2
    while k <= m:
        j = k // 2
        while j > 0:
            for i in range(m):
                ixj = i ^ j
                if ixj > i:
                    ascending = (i & k) == 0
                    if (arr[i][0] > arr[ixj][0]) == ascending:
                        arr[i], arr[ixj] = arr[ixj], arr[i]
            j //= 2
        k *= 2

    return arr[:n]


def sort_row_bitonic(row_pixels: np.ndarray, mode: str) -> np.ndarray:
    # Sort one image row left-to-right by chosen key.
    pixels = [tuple(px) for px in row_pixels.tolist()]
    keyed = [(pixel_key(px, mode), px) for px in pixels]
    sorted_pairs = bitonic_sort_pairs(keyed)
    sorted_pixels = [px for _, px in sorted_pairs]
    return np.array(sorted_pixels, dtype=np.uint8)


def sort_row_task(args):
    # Top-level helper required by ProcessPoolExecutor (picklable function).
    index, row, mode = args
    return index, sort_row_bitonic(row, mode)


class PixelSortApp:
    def __init__(self, root: tk.Tk):
        # Main app state
        self.root = root
        self.root.title("PixelSort Studio")
        self.root.geometry("1260x760")
        self.root.minsize(1100, 680)

        self.input_image = None
        self.preview_size = (340, 340)
        self.result_seq_image = None
        self.result_par_image = None

        self.worker_queue = queue.Queue()
        self.animating = False

        # Build UI + start polling queue messages from worker thread.
        self._build_style()
        self._build_layout()
        self.root.after(60, self._poll_worker_queue)

    def _build_style(self):
        # Centralized ttk styling so the GUI looks consistent/professional.
        style = ttk.Style(self.root)
        style.theme_use("clam")

        self.root.configure(bg="#111827")
        # Force default ttk frames to use the app's dark background.
        style.configure("TFrame", background="#111827")
        style.configure(
            "Card.TFrame",
            background="#1f2937",
            borderwidth=0,
        )
        style.configure(
            "Header.TLabel",
            background="#111827",
            foreground="#f9fafb",
            font=("Segoe UI", 20, "bold"),
        )
        style.configure(
            "Subtle.TLabel",
            background="#111827",
            foreground="#9ca3af",
            font=("Segoe UI", 10),
        )
        style.configure(
            "CardTitle.TLabel",
            background="#1f2937",
            foreground="#e5e7eb",
            font=("Segoe UI", 11, "bold"),
        )
        style.configure(
            "CardBody.TLabel",
            background="#1f2937",
            foreground="#d1d5db",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Action.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=(14, 8),
        )
        style.configure(
            "TProgressbar",
            troughcolor="#374151",
            background="#22c55e",
            bordercolor="#374151",
            lightcolor="#22c55e",
            darkcolor="#22c55e",
        )

    def _build_layout(self):
        # Layout has: controls (top), three image panels (middle), metrics (bottom).
        outer = ttk.Frame(self.root, style="TFrame")
        outer.pack(fill=tk.BOTH, expand=True, padx=20, pady=12)

        title = ttk.Label(outer, text="PixelSort Studio", style="Header.TLabel")
        title.pack(anchor="w")
        subtitle = ttk.Label(
            outer,
            text="Benchmarking sequential vs parallel pixel sorting with Bitonic Sort.",
            style="Subtle.TLabel",
        )
        subtitle.pack(anchor="w", pady=(0, 8))

        controls = ttk.Frame(outer, style="Card.TFrame")
        controls.pack(fill=tk.X, pady=(0, 10))

        controls_inner = ttk.Frame(controls, style="Card.TFrame")
        controls_inner.pack(fill=tk.X, padx=14, pady=12)

        ttk.Button(
            controls_inner,
            text="Upload Image",
            command=self.upload_image,
            style="Action.TButton",
        ).grid(row=0, column=0, padx=(0, 10), pady=6, sticky="w")

        ttk.Label(controls_inner, text="Sort By", style="CardBody.TLabel").grid(
            row=0, column=1, padx=(2, 8), pady=6, sticky="w"
        )
        self.sort_mode = tk.StringVar(value="hue")
        sort_mode_combo = ttk.Combobox(
            controls_inner,
            width=16,
            textvariable=self.sort_mode,
            state="readonly",
            values=("hue", "saturation", "value", "brightness"),
        )
        sort_mode_combo.grid(row=0, column=2, padx=(0, 12), pady=6, sticky="w")

        ttk.Label(controls_inner, text="Max Size", style="CardBody.TLabel").grid(
            row=0, column=3, padx=(2, 8), pady=6, sticky="w"
        )
        self.max_side = tk.IntVar(value=420)
        max_size_spin = ttk.Spinbox(
            controls_inner,
            from_=160,
            to=900,
            increment=20,
            width=8,
            textvariable=self.max_side,
        )
        max_size_spin.grid(row=0, column=4, padx=(0, 12), pady=6, sticky="w")

        self.run_button = ttk.Button(
            controls_inner,
            text="Run Sequential + Parallel Benchmark",
            command=self.start_sorting,
            style="Action.TButton",
        )
        self.run_button.grid(row=0, column=5, padx=(0, 0), pady=6, sticky="e")

        controls_inner.grid_columnconfigure(5, weight=1)

        self.status_var = tk.StringVar(
            value="Upload an image to begin. The app will animate results after processing."
        )
        status_label = ttk.Label(
            outer,
            textvariable=self.status_var,
            style="Subtle.TLabel",
        )
        status_label.pack(anchor="w", pady=(2, 8))

        self.progress = ttk.Progressbar(outer, orient="horizontal", mode="determinate")
        self.progress.pack(fill=tk.X, pady=(0, 14))

        panels = ttk.Frame(outer)
        panels.pack(fill=tk.BOTH, expand=True)
        panels.configure(style="Card.TFrame")

        self.original_panel = self._create_image_panel(panels, "Original")
        self.original_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.seq_panel = self._create_image_panel(panels, "Sequential (Bitonic)")
        self.seq_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8)

        self.par_panel = self._create_image_panel(panels, "Parallel (Bitonic)")
        self.par_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))

        metrics = ttk.Frame(outer, style="Card.TFrame")
        metrics.pack(fill=tk.X, pady=(14, 0))
        metrics_inner = ttk.Frame(metrics, style="Card.TFrame")
        metrics_inner.pack(fill=tk.X, padx=14, pady=12)

        self.seq_time_var = tk.StringVar(value="Sequential: --")
        self.par_time_var = tk.StringVar(value="Parallel: --")
        self.diff_var = tk.StringVar(value="Difference: --")
        self.speedup_var = tk.StringVar(value="Speedup: --")

        ttk.Label(metrics_inner, textvariable=self.seq_time_var, style="CardBody.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 26)
        )
        ttk.Label(metrics_inner, textvariable=self.par_time_var, style="CardBody.TLabel").grid(
            row=0, column=1, sticky="w", padx=(0, 26)
        )
        ttk.Label(metrics_inner, textvariable=self.diff_var, style="CardBody.TLabel").grid(
            row=0, column=2, sticky="w", padx=(0, 26)
        )
        ttk.Label(metrics_inner, textvariable=self.speedup_var, style="CardBody.TLabel").grid(
            row=0, column=3, sticky="w"
        )

    def _create_image_panel(self, parent, title_text):
        # Reusable card for original/sequential/parallel image views.
        panel = ttk.Frame(parent, style="Card.TFrame")
        ttk.Label(panel, text=title_text, style="CardTitle.TLabel").pack(anchor="w", padx=10, pady=(10, 8))

        image_label = tk.Label(
            panel,
            bg="#0f172a",
            fg="#9ca3af",
            text="No image",
            font=("Segoe UI", 10),
            relief=tk.FLAT,
        )
        image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        panel.image_label = image_label
        return panel

    def upload_image(self):
        # User chooses an image file from disk.
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Image Error", f"Could not open image:\n{exc}")
            return

        self.input_image = image
        self._set_image_on_panel(self.original_panel.image_label, image)
        self._set_image_on_panel(self.seq_panel.image_label, Image.new("RGB", image.size, color=(18, 24, 38)))
        self._set_image_on_panel(self.par_panel.image_label, Image.new("RGB", image.size, color=(18, 24, 38)))
        self.status_var.set("Image loaded. Choose a color metric and run the benchmark.")

    def _set_image_on_panel(self, label_widget, image: Image.Image):
        # Render to a fixed preview box without stretching aspect ratio.
        rendered = image.copy()
        rendered = ImageOps.contain(rendered, self.preview_size, method=Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(rendered)
        label_widget.configure(image=photo, text="")
        label_widget.image = photo

    def start_sorting(self):
        # Triggered by the benchmark button.
        # Starts background work so the UI does not freeze.
        if self.input_image is None:
            messagebox.showinfo("Missing Image", "Please upload an image first.")
            return
        if self.animating:
            return

        mode = self.sort_mode.get().strip().lower()
        max_side = max(160, min(900, int(self.max_side.get())))

        self.run_button.state(["disabled"])
        self.progress.configure(value=0, maximum=100)
        self.status_var.set("Preparing image and launching benchmark...")
        self.seq_time_var.set("Sequential: --")
        self.par_time_var.set("Parallel: --")
        self.diff_var.set("Difference: --")
        self.speedup_var.set("Speedup: --")

        worker = threading.Thread(
            target=self._benchmark_worker,
            args=(self.input_image.copy(), mode, max_side),
            daemon=True,
        )
        worker.start()

    def _benchmark_worker(self, pil_image: Image.Image, mode: str, max_side: int):
        try:
            # Preprocess: resize based on Max Size setting, preserve aspect ratio.
            prepared = ImageOps.contain(pil_image, (max_side, max_side), method=Image.Resampling.LANCZOS).convert("RGB")
            arr = np.array(prepared, dtype=np.uint8)
            h, _, _ = arr.shape

            # Split image into rows; each row is sorted independently.
            rows = [arr[i, :, :] for i in range(h)]

            # ----- Sequential benchmark -----
            self.worker_queue.put(("status", f"Sequential {mode} sorting in progress..."))
            seq_rows = []
            t0 = time.perf_counter()
            for idx, row in enumerate(rows):
                seq_rows.append(sort_row_bitonic(row, mode))
                if (idx + 1) % max(1, h // 40) == 0 or idx + 1 == h:
                    self.worker_queue.put(("progress", (idx + 1) / h * 50.0))
            seq_time = time.perf_counter() - t0
            seq_image = Image.fromarray(np.stack(seq_rows, axis=0), mode="RGB")

            # ----- Parallel benchmark -----
            # We cap workers to avoid overloading low-end machines.
            workers = max(2, min((os.cpu_count() or 2), 12))
            self.worker_queue.put(("status", f"Parallel sorting on {workers} worker processes..."))

            par_rows = [None] * h
            t1 = time.perf_counter()
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(sort_row_task, (idx, row, mode))
                    for idx, row in enumerate(rows)
                ]
                done_count = 0
                for future in as_completed(futures):
                    idx, sorted_row = future.result()
                    # Futures complete out of order, so we reinsert by index.
                    par_rows[idx] = sorted_row
                    done_count += 1
                    if done_count % max(1, h // 40) == 0 or done_count == h:
                        self.worker_queue.put(("progress", 50.0 + (done_count / h * 50.0)))
            par_time = time.perf_counter() - t1
            par_image = Image.fromarray(np.stack(par_rows, axis=0), mode="RGB")

            self.worker_queue.put(
                (
                    "done",
                    {
                        "prepared": prepared,
                        "seq_image": seq_image,
                        "par_image": par_image,
                        "seq_time": seq_time,
                        "par_time": par_time,
                        "mode": mode,
                        "workers": workers,
                    },
                )
            )
        except Exception as exc:
            self.worker_queue.put(("error", str(exc)))

    def _poll_worker_queue(self):
        # Pull messages from worker thread/processes and safely update UI.
        try:
            while True:
                item = self.worker_queue.get_nowait()
                kind = item[0]

                if kind == "status":
                    self.status_var.set(item[1])
                elif kind == "progress":
                    self.progress.configure(value=float(item[1]))
                elif kind == "error":
                    self.run_button.state(["!disabled"])
                    self.status_var.set("Processing failed.")
                    messagebox.showerror("Processing Error", item[1])
                elif kind == "done":
                    payload = item[1]
                    self._apply_results(payload)
        except queue.Empty:
            pass

        self.root.after(60, self._poll_worker_queue)

    def _apply_results(self, payload):
        # Show final timing metrics and launch reveal animation.
        prepared = payload["prepared"]
        seq_image = payload["seq_image"]
        par_image = payload["par_image"]
        seq_time = float(payload["seq_time"])
        par_time = float(payload["par_time"])
        mode = payload["mode"]
        workers = payload["workers"]

        self._set_image_on_panel(self.original_panel.image_label, prepared)
        self.result_seq_image = seq_image
        self.result_par_image = par_image

        diff = abs(seq_time - par_time)
        speedup = (seq_time / par_time) if par_time > 0 else 0.0

        self.seq_time_var.set(f"Sequential: {seq_time:.4f} s")
        self.par_time_var.set(f"Parallel: {par_time:.4f} s")
        self.diff_var.set(f"Difference: {diff:.4f} s")
        self.speedup_var.set(f"Speedup: {speedup:.2f}x")

        self.status_var.set(
            f"Completed {mode} pixel sorting. Animated reveal running (parallel workers: {workers})."
        )
        self.progress.configure(value=100.0)
        self._animate_reveal(prepared, seq_image, par_image)

    def _animate_reveal(self, original: Image.Image, seq_image: Image.Image, par_image: Image.Image):
        # Dynamic effect: reveal sorted rows from top to bottom.
        if self.animating:
            return
        self.animating = True

        base = np.array(original, dtype=np.uint8)
        seq = np.array(seq_image, dtype=np.uint8)
        par = np.array(par_image, dtype=np.uint8)

        h = base.shape[0]
        steps = min(120, h)
        rows_per_step = max(1, math.ceil(h / steps))
        current_row = 0

        def tick():
            nonlocal current_row
            if current_row > h:
                self._set_image_on_panel(self.seq_panel.image_label, seq_image)
                self._set_image_on_panel(self.par_panel.image_label, par_image)
                self.animating = False
                self.run_button.state(["!disabled"])
                self.status_var.set("Benchmark complete. Try a different metric or image.")
                return

            seq_frame = base.copy()
            par_frame = base.copy()
            seq_frame[:current_row, :, :] = seq[:current_row, :, :]
            par_frame[:current_row, :, :] = par[:current_row, :, :]

            self._set_image_on_panel(self.seq_panel.image_label, Image.fromarray(seq_frame, mode="RGB"))
            self._set_image_on_panel(self.par_panel.image_label, Image.fromarray(par_frame, mode="RGB"))

            current_row += rows_per_step
            self.root.after(24, tick)

        tick()


def main():
    # App entry point.
    root = tk.Tk()
    app = PixelSortApp(root)
    root.mainloop()


if __name__ == "__main__":
    # Needed for multiprocessing compatibility on Windows.
    freeze_support()
    main()
