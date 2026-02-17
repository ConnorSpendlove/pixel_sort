# Pixel Sorter

Sort image pixels by **hue**, **luminance**, **saturation**, **value**, **red**, **green**, **blue**, or **chroma**. Compare **sequential** vs **parallel** sort times in a GUI with before/after views and a timing graph.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python pixel_sorter.py
```

## Usage

1. **Generate scrambled image** – Creates a random-color image at the selected size (Small / Medium / Large).
2. **Sort by** – Choose sort key: luminance, hue, saturation, value, red, green, blue, chroma.
3. **Sort (sequential + parallel, update graph)** – Runs both algorithms, shows the result, and adds a bar to the timing graph.
4. **Watch sequential sort** – Sorts row-by-row and updates the “After” view as it goes.
5. **Watch parallel sort** – Runs the parallel sort and shows the result (no row-by-row animation).

The **time comparison** graph at the bottom shows sequential vs parallel time for each run. On multi-core machines, parallel is usually faster for larger images.

## Why is parallel faster? (CPU, not GPU)

This app uses your **CPU only** (no GPU). **Parallel** here means splitting work across **multiple CPU cores** (via `ProcessPoolExecutor`): each core sorts a subset of rows at the same time, so total time can drop. It's faster because several rows are sorted simultaneously, not because a GPU is doing the work. GPUs would require different code (e.g. CuPy or a GPU kernel); this project is CPU-based so it runs everywhere without special hardware.

## Notes

- If parallel process spawn fails (e.g. in some restricted environments), the app falls back to sequential for the “parallel” run and reports it in the status bar.
- Use **Medium** or **Large** image size for clearer timing differences between sequential and parallel.
