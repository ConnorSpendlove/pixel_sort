# PixelSort Studio

PixelSort Studio is a Python GUI application that loads an image, sorts its pixels by a selected color property, and benchmarks **sequential vs parallel** execution times.

## Why this project exists

The purpose of this app is to show how algorithmic design and hardware parallelism affect runtime in real image-processing tasks.  

## Bitonic Sort

This project uses **Bitonic Sort** for each image row.

### What is Bitonic Sort?
Bitonic Sort is a comparison-based sorting network algorithm known from parallel systems and GPU-style sorting pipelines. It is less common in everyday application programming than algorithms like quicksort, mergesort, or timsort.

### Why it fits this assignment
- Its compare-and-swap stages are structured in a way that maps naturally to parallel execution models.
- It is deterministic and visually interesting for pixel-sorting art effects.

### How it is used in this app
1. The image is resized to a manageable working resolution.
2. Each row of pixels is sorted by a chosen key (`hue`, `saturation`, `value`, or `brightness`).
3. Sequential mode sorts rows one after another.
4. Parallel mode distributes row-sorting jobs across multiple processes (`ProcessPoolExecutor`).
5. The GUI reports timing difference and speedup.

## Features

- Professional desktop GUI (Tkinter + ttk styling)
- Upload image support (`png`, `jpg`, `jpeg`, `bmp`, `webp`)
- Sort by:
  - Hue
  - Saturation
  - Value
  - Brightness
- Sequential benchmark
- Parallel benchmark using multiple CPU workers
- Time comparison:
  - Sequential time
  - Parallel time
  - Absolute difference
  - Speedup ratio
- Animated reveal of sorted outputs

## Installation

1. Ensure Python 3.10+ is installed.
2. Install dependencies:

```bash
pip install pillow numpy
```

## Run

```bash
python app.py
```

## Notes about performance

- Parallel can be faster for larger workloads, but on small images process startup overhead may reduce the gain.
- For clearer differences, test with larger `Max Size` values.
- Timing can vary by CPU core count and background system load.