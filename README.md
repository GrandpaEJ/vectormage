# vectormage

High-quality PNG/JPG to SVG vectorizer CLI tool. Produces clean, detailed SVGs from raster images — optimized for high-poly quality with many colors and fine detail.

## Installation

```bash
cd image-tool
pip install -e .
```

## Usage

```bash
# Basic conversion
vectormage input.png -o output.svg

# Or use as a module
python -m vectormage input.png -o output.svg
```

### Quality Levels

| Level | Colors | Detail | Speed |
|-------|--------|--------|-------|
| `low` | 8 | Aggressive simplification | Fast |
| `medium` | 16 | Moderate (default) | Medium |
| `high` | 32 | Fine detail | Slower |
| `max` | 64+ | Minimal simplification | Very slow |

### Examples

```bash
# High quality with 64 colors
vectormage photo.jpg -o vector.svg --quality high --colors 64

# Posterized look for illustrations
vectormage illustration.png -o art.svg --posterize --colors 16

# Maximum detail with smoothing
vectormage logo.png -o logo.svg --quality max --smooth 8 --optimize

# Preprocessing
vectormage noisy.png -o clean.svg --denoise --sharpen --contrast 1.2

# Batch mode
vectormage *.png -o output_dir/ --quality high

# Preview (stdout)
vectormage input.png --preview
```

### Options

- `-q, --quality` — Quality level: low, medium, high, max
- `-c, --colors N` — Quantize to N colors (auto based on quality)
- `--posterize` — Flat color regions
- `--gradient` — Preserve gradients
- `--denoise` — Gaussian blur preprocessing
- `--sharpen` — Unsharp mask enhancement
- `--contrast N` — Contrast adjustment (1.0 = normal)
- `--threshold N` — Black/white threshold (0-255)
- `--optimize` — Simplify paths, reduce SVG size
- `--smooth N` — Bezier curve smoothing (0-10)
- `--stroke-width N` — Use stroked paths
- `--preview` — Print SVG to stdout

## Architecture

```
vectormage/
├── __init__.py      # Package init
├── __main__.py      # python -m entry point
├── cli.py           # Click-based CLI
├── processor.py     # Image preprocessing pipeline
├── palettes.py      # Color quantization (K-means)
├── tracer.py        # Core vectorization (contours → bezier → paths)
├── svg_writer.py    # SVG file generation
└── optimizer.py     # Path optimization & merging
```

## Tech Stack

- Pillow — image loading, preprocessing
- NumPy — pixel array operations
- scikit-learn — K-means color quantization
- OpenCV — edge detection, contour finding
- SciPy — bezier curve fitting
- Click — CLI framework
- Rich — progress bars, colored output
