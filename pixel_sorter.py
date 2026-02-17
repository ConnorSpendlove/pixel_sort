"""
Pixel Sorter - Sort image pixels by hue, luminance, etc.
Uses parallel and sequential algorithms with timing comparison and GUI.
"""

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np
from PIL import Image, ImageTk

# Sort key names and their index for the worker
SORT_KEYS = [
    "luminance",
    "hue",
    "saturation",
    "value",
    "red",
    "green",
    "blue",
    "chroma",
]


def rgb_to_luminance(r, g, b):
    """Relative luminance (perceived brightness)."""
    return 0.299 * r + 0.587 * g + 0.114 * b


def rgb_to_hsv_batch(r, g, b):
    """Vectorized RGB [0-255] to (h, s, v) in 0-1. r,g,b are 1D arrays."""
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    c = mx - mn
    v = mx
    s = np.where(mx > 1e-9, c / mx, 0.0)
    rc = np.where(np.abs(c) > 1e-9, (mx - r) / c, 0.0)
    gc = np.where(np.abs(c) > 1e-9, (mx - g) / c, 0.0)
    bc = np.where(np.abs(c) > 1e-9, (mx - b) / c, 0.0)
    h = np.zeros_like(r)
    mask_r = (mx == r) & (c > 0)
    mask_g = (mx == g) & (c > 0)
    mask_b = (mx == b) & (c > 0)
    h = np.where(mask_r, bc - gc, h)
    h = np.where(mask_g, 2.0 + rc - bc, h)
    h = np.where(mask_b, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    return h, s, v


def get_sort_key_vector(r, g, b, key_name):
    """Return 1D array of sort key for each pixel (same length as r)."""
    if key_name == "luminance":
        return rgb_to_luminance(r, g, b)
    if key_name in ("hue", "saturation", "value"):
        h, s, v = rgb_to_hsv_batch(r, g, b)
        return {"hue": h, "saturation": s, "value": v}[key_name]
    if key_name == "red":
        return r.astype(float)
    if key_name == "green":
        return g.astype(float)
    if key_name == "blue":
        return b.astype(float)
    if key_name == "chroma":
        return np.maximum(r, np.maximum(g, b)).astype(float) - np.minimum(
            r, np.minimum(g, b)
        ).astype(float)
    return rgb_to_luminance(r, g, b)


def sort_row_by_key(row_rgb, key_name):
    """Sort a single row (Hx3 array) by the given key. Returns sorted row."""
    r, g, b = row_rgb[:, 0], row_rgb[:, 1], row_rgb[:, 2]
    key = get_sort_key_vector(r, g, b, key_name)
    order = np.argsort(key)
    return row_rgb[order]


def bubble_sort_one_pass(row_rgb, key_name):
    """
    One pass of bubble sort on a row (in-place by sort key).
    Returns True if any swap was made (row may still be unsorted).
    """
    w = row_rgb.shape[0]
    r, g, b = row_rgb[:, 0], row_rgb[:, 1], row_rgb[:, 2]
    key = get_sort_key_vector(r, g, b, key_name)
    had_swap = False
    for i in range(w - 1):
        if key[i] > key[i + 1]:
            row_rgb[i], row_rgb[i + 1] = row_rgb[i + 1].copy(), row_rgb[i].copy()
            key[i], key[i + 1] = key[i + 1], key[i]
            had_swap = True
    return had_swap


def bubble_sort_passes(row_rgb, key_name, max_passes):
    """
    Run up to max_passes bubble-sort passes on row_rgb (in-place).
    Returns True if the row is now fully sorted.
    """
    for _ in range(max_passes):
        if not bubble_sort_one_pass(row_rgb, key_name):
            return True
    return False


def sort_row_worker(args):
    """Worker for parallel sort: (row_index, row_rgb, key_name) -> (row_index, sorted_row)."""
    row_index, row_rgb, key_name = args
    sorted_row = sort_row_by_key(np.array(row_rgb), key_name)
    return row_index, sorted_row


def sort_image_sequential(pixels, key_name, progress_callback=None):
    """
    Sort each row of the image by the given key. Sequential.
    pixels: (H, W, 3) uint8 array.
    """
    h, w, _ = pixels.shape
    out = np.empty_like(pixels)
    for y in range(h):
        out[y] = sort_row_by_key(pixels[y].copy(), key_name)
        if progress_callback and (y % max(1, h // 50)) == 0:
            progress_callback(y / h)
    if progress_callback:
        progress_callback(1.0)
    return out


def sort_image_parallel(pixels, key_name, progress_callback=None, max_workers=None):
    """
    Sort each row in parallel. pixels: (H, W, 3) uint8.
    On environments where process spawn fails (e.g. some Windows setups),
    falls back to sequential and returns (result, used_fallback=True) via exception.
    """
    if max_workers is None:
        max_workers = max(1, cpu_count() - 1)
    h, w, _ = pixels.shape
    tasks = [(y, pixels[y].tolist(), key_name) for y in range(h)]
    out = np.empty_like(pixels)
    completed = 0
    used_fallback = False
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(sort_row_worker, t): t[0] for t in tasks}
            for future in as_completed(futures):
                row_index, sorted_row = future.result()
                out[row_index] = sorted_row
                completed += 1
                if progress_callback and completed % max(1, h // 50) == 0:
                    progress_callback(completed / h)
    except (PermissionError, OSError, Exception):
        used_fallback = True
        out = sort_image_sequential(pixels, key_name, progress_callback)
    if progress_callback:
        progress_callback(1.0)
    return out, used_fallback


def generate_scrambled_image(width, height, seed=None):
    """Generate an image of random colors (scrambled pixels)."""
    if seed is not None:
        np.random.seed(seed)
    pixels = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return pixels


def pixels_to_pil(pixels):
    """(H,W,3) uint8 -> PIL Image RGB."""
    return Image.fromarray(pixels, mode="RGB")


def pil_to_pixels(im):
    """PIL Image -> (H,W,3) uint8."""
    return np.array(im.convert("RGB"))


# --- Timing and history for graph ---
def run_timed_sort(pixels, key_name, use_parallel, progress_callback=None):
    """Run sort and return (result_pixels, elapsed_seconds, parallel_fallback=False)."""
    start = time.perf_counter()
    parallel_fallback = False
    if use_parallel:
        result, parallel_fallback = sort_image_parallel(pixels, key_name, progress_callback)
    else:
        result = sort_image_sequential(pixels, key_name, progress_callback)
    elapsed = time.perf_counter() - start
    return result, elapsed, parallel_fallback


# --- GUI ---
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class PixelSorterApp:
    # Delay between frames (0 = as fast as possible)
    WATCH_DELAY_MS = 0
    # Rows per frame for sequential watch (instant sort per row)
    WATCH_SEQUENTIAL_ROWS_PER_FRAME = 8
    # Rows per frame for parallel watch
    WATCH_PARALLEL_ROWS_PER_FRAME = 16

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Pixel Sorter — Parallel vs Sequential")
        self.root.geometry("960x720")
        self.root.minsize(800, 580)

        # Data
        self.pixels_before = None  # (H,W,3)
        self.pixels_after = None
        self.time_history = []  # list of {"key": str, "sequential": float, "parallel": float}
        self.img_width = 320
        self.img_height = 240
        self.display_width = 200
        self.display_height = 150
        self.size_presets = [
            ("Small (320×240)", 320, 240),
            ("Medium (480×360)", 480, 360),
            ("Large (640×480)", 640, 480),
        ]
        self._watch_after_id = None  # cancel slow-mo when starting new one

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # Top: controls
        ctrl = ttk.Frame(main)
        ctrl.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(ctrl, text="Generate scrambled image", command=self._on_generate).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Label(ctrl, text="Size:").pack(side=tk.LEFT, padx=(8, 4))
        self.size_var = tk.StringVar(value=self.size_presets[0][0])
        self.size_combo = ttk.Combobox(
            ctrl, textvariable=self.size_var,
            values=[p[0] for p in self.size_presets], state="readonly", width=16
        )
        self.size_combo.pack(side=tk.LEFT, padx=4)
        ttk.Label(ctrl, text="Sort by:").pack(side=tk.LEFT, padx=(8, 4))
        self.sort_var = tk.StringVar(value="luminance")
        self.sort_combo = ttk.Combobox(
            ctrl, textvariable=self.sort_var, values=SORT_KEYS, state="readonly", width=12
        )
        self.sort_combo.pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrl, text="Sort (sequential + parallel, update graph)", command=self._on_sort).pack(
            side=tk.LEFT, padx=8
        )
        ttk.Button(ctrl, text="Watch sequential sort", command=self._on_watch_sequential).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(ctrl, text="Watch parallel sort", command=self._on_watch_parallel).pack(
            side=tk.LEFT, padx=4
        )

        # Progress
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress = ttk.Progressbar(main, variable=self.progress_var, maximum=1.0)
        self.progress.pack(fill=tk.X, pady=(0, 8))
        self.status_var = tk.StringVar(value="Generate a scrambled image, then choose sort and run.")
        ttk.Label(main, textvariable=self.status_var).pack(anchor=tk.W)
        tip = "Parallel = multiple CPU cores (this app uses CPU, not GPU). Parallel is often faster."
        ttk.Label(main, text=tip, foreground="gray", font=("", 8)).pack(anchor=tk.W)

        # Middle: before / after images (compact so graph gets space)
        img_frame = ttk.Frame(main)
        img_frame.pack(fill=tk.X, pady=6)

        before_frame = ttk.Frame(img_frame)
        before_frame.pack(side=tk.LEFT, padx=6)
        ttk.Label(before_frame, text="Before").pack()
        self.canvas_before = tk.Canvas(before_frame, width=self.display_width, height=self.display_height, bg="#222")
        self.canvas_before.pack(pady=2)

        after_frame = ttk.Frame(img_frame)
        after_frame.pack(side=tk.LEFT, padx=6)
        ttk.Label(after_frame, text="After").pack()
        self.canvas_after = tk.Canvas(after_frame, width=self.display_width, height=self.display_height, bg="#222")
        self.canvas_after.pack(pady=2)

        self.photo_before = None
        self.photo_after = None

        # Bottom: time comparison graph (main focus)
        ttk.Label(main, text="Time comparison (Sequential vs Parallel)", font=("", 10, "bold")).pack(
            anchor=tk.W, pady=(6, 2)
        )
        fig = Figure(figsize=(7, 3.2), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_ylabel("Time (seconds)")
        self.ax.set_xlabel("Run (sort key)")
        self.fig = fig
        self.canvas_graph = FigureCanvasTkAgg(fig, master=main)
        self.canvas_graph.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_empty_graph()

    def _draw_empty_graph(self):
        self.ax.clear()
        self.ax.set_ylabel("Time (seconds)")
        self.ax.set_xlabel("Run (sort key)")
        self.ax.set_ylim(0, 1)
        self.ax.set_xlim(-0.5, 0.5)
        self.ax.set_title("Run a sort to see timing comparison")
        self.canvas_graph.draw()

    def _update_graph(self):
        self.ax.clear()
        if not self.time_history:
            self._draw_empty_graph()
            return
        keys = [t["key"] for t in self.time_history]
        seq_times = [t["sequential"] for t in self.time_history]
        par_times = [t["parallel"] for t in self.time_history]
        x = np.arange(len(keys))
        w = 0.35
        self.ax.bar(x - w / 2, seq_times, w, label="Sequential", color="steelblue")
        self.ax.bar(x + w / 2, par_times, w, label="Parallel", color="coral")
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(keys, rotation=45, ha="right")
        self.ax.set_ylabel("Time (seconds)")
        self.ax.set_xlabel("Sort key")
        self.ax.legend()
        self.ax.set_title("Sequential vs Parallel sort time")
        self.canvas_graph.draw()

    def _show_pixels_before(self, pixels):
        if pixels is None:
            return
        im = pixels_to_pil(pixels)
        im = im.resize((self.display_width, self.display_height), Image.Resampling.NEAREST)
        self.photo_before = ImageTk.PhotoImage(im)
        self.canvas_before.delete("all")
        self.canvas_before.create_image(0, 0, anchor=tk.NW, image=self.photo_before)

    def _show_pixels_after(self, pixels):
        if pixels is None:
            return
        im = pixels_to_pil(pixels)
        im = im.resize((self.display_width, self.display_height), Image.Resampling.NEAREST)
        self.photo_after = ImageTk.PhotoImage(im)
        self.canvas_after.delete("all")
        self.canvas_after.create_image(0, 0, anchor=tk.NW, image=self.photo_after)

    def _get_size(self):
        for label, w, h in self.size_presets:
            if self.size_var.get() == label:
                return w, h
        return 320, 240

    def _on_generate(self):
        self.img_width, self.img_height = self._get_size()
        self.status_var.set("Generating scrambled image...")
        self.root.update_idletasks()
        self.pixels_before = generate_scrambled_image(self.img_width, self.img_height)
        self.pixels_after = None
        self._show_pixels_before(self.pixels_before)
        self._show_pixels_after(None)
        self.canvas_after.delete("all")
        self.canvas_after.create_text(
            self.display_width // 2, self.display_height // 2, text="(after sort)", fill="gray"
        )
        self.status_var.set("Scrambled image ready. Choose sort key and run Sort or Watch.")
        self.progress_var.set(0.0)

    def _on_sort(self):
        if self.pixels_before is None:
            messagebox.showinfo("Info", "Generate a scrambled image first.")
            return
        key_name = self.sort_var.get()
        if key_name not in SORT_KEYS:
            key_name = "luminance"

        self.status_var.set("Running sequential sort...")
        self.progress_var.set(0.0)
        self.root.update_idletasks()
        _, t_seq, _ = run_timed_sort(
            self.pixels_before.copy(),
            key_name,
            use_parallel=False,
            progress_callback=lambda p: self._safe_progress(p),
        )
        self._safe_progress(1.0)

        self.status_var.set("Running parallel sort...")
        self.progress_var.set(0.0)
        self.root.update_idletasks()
        result, t_par, par_fallback = run_timed_sort(
            self.pixels_before.copy(),
            key_name,
            use_parallel=True,
            progress_callback=lambda p: self._safe_progress(p),
        )
        self._safe_progress(1.0)

        self.pixels_after = result
        self._show_pixels_after(self.pixels_after)
        self.time_history.append({
            "key": key_name,
            "sequential": t_seq,
            "parallel": t_par,
        })
        self._update_graph()
        msg = f"Done. Sequential: {t_seq:.3f}s | Parallel: {t_par:.3f}s | Key: {key_name}"
        if par_fallback:
            msg += " [Parallel spawn failed — sequential used for parallel run]"
        self.status_var.set(msg)
        self.progress_var.set(0.0)

    def _safe_progress(self, p):
        self.progress_var.set(p)
        self.root.update_idletasks()

    def _watch_display_update(self, out):
        """Update the After canvas with current buffer and keep photo ref."""
        im = pixels_to_pil(out)
        im = im.resize((self.display_width, self.display_height), Image.Resampling.NEAREST)
        ph = ImageTk.PhotoImage(im)
        self.canvas_after.delete("all")
        self.canvas_after.create_image(0, 0, anchor=tk.NW, image=ph)
        self.photo_after = ph
        self.root.update_idletasks()

    def _watch_step(self):
        """One step: sort next chunk of rows and update display."""
        h = self._watch_pixels.shape[0]
        rows_per_frame = (
            self.WATCH_PARALLEL_ROWS_PER_FRAME if self._watch_use_parallel
            else self.WATCH_SEQUENTIAL_ROWS_PER_FRAME
        )
        end_y = min(self._watch_y + rows_per_frame, h)
        for y in range(self._watch_y, end_y):
            self._watch_out[y] = sort_row_by_key(self._watch_pixels[y].copy(), self._watch_key)
        self._watch_y = end_y
        self._safe_progress(self._watch_y / h)
        self._watch_display_update(self._watch_out)
        if self._watch_y >= h:
            self.pixels_after = self._watch_out.copy()
            self._watch_after_id = None
            elapsed = time.perf_counter() - self._watch_start
            mode = "parallel" if self._watch_use_parallel else "sequential"
            self.status_var.set(f"{mode.capitalize()} sort finished in {elapsed:.3f}s")
            self.progress_var.set(0.0)
            return
        self._watch_after_id = self.root.after(self.WATCH_DELAY_MS, self._watch_step)

    def _watch_sort(self, use_parallel):
        if self.pixels_before is None:
            messagebox.showinfo("Info", "Generate a scrambled image first.")
            return
        key_name = self.sort_var.get()
        if key_name not in SORT_KEYS:
            key_name = "luminance"
        if self._watch_after_id is not None:
            self.root.after_cancel(self._watch_after_id)
            self._watch_after_id = None
        self._watch_pixels = self.pixels_before.copy()
        self._watch_out = self._watch_pixels.copy()  # so unsorted rows show correctly
        self._watch_y = 0
        self._watch_key = key_name
        self._watch_use_parallel = use_parallel
        self._watch_start = time.perf_counter()
        mode = "parallel" if use_parallel else "sequential"
        self.status_var.set(f"Watching {mode} sort...")
        self._watch_step()

    def _on_watch_sequential(self):
        self._watch_sort(use_parallel=False)

    def _on_watch_parallel(self):
        self._watch_sort(use_parallel=True)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = PixelSorterApp()
    app.run()
