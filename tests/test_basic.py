"""Basic sanity tests for vectormage."""

import os
import tempfile
import numpy as np
from PIL import Image


def create_test_image(path: str, size: int = 100):
    """Create a simple test PNG with colored shapes."""
    img = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    pixels = np.array(img)

    # Draw a red square
    pixels[20:80, 20:80] = [255, 0, 0, 255]
    # Draw a blue circle-ish shape
    for x in range(size):
        for y in range(size):
            if (x - 70) ** 2 + (y - 30) ** 2 < 20 ** 2:
                pixels[y, x] = [0, 0, 255, 255]

    img = Image.fromarray(pixels)
    img.save(path)
    return path


def test_preprocessing():
    """Test image preprocessing pipeline."""
    from vectormage.processor import load_image, preprocess

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        create_test_image(f.name)
        path = f.name

    try:
        img = load_image(path)
        assert img.shape[2] == 4, "Should be RGBA"
        assert img.shape[0] == 100, "Height should be 100"

        processed = preprocess(img, do_denoise=True, do_sharpen=True)
        assert processed.shape == img.shape
        print("PASS: preprocessing")
    finally:
        os.unlink(path)


def test_quantization():
    """Test color quantization."""
    from vectormage.processor import load_image
    from vectormage.palettes import quantize_colors

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        create_test_image(f.name)
        path = f.name

    try:
        img = load_image(path)
        colors, proportions = quantize_colors(img, n_colors=4, quality="low")
        assert len(colors) > 0, "Should find colors"
        assert len(colors) == len(proportions)
        print("PASS: quantization")
    finally:
        os.unlink(path)


def test_vectorization():
    """Test full vectorization pipeline."""
    from vectormage.processor import load_image, preprocess
    from vectormage.palettes import quantize_colors, build_color_layers
    from vectormage.tracer import trace_all_layers

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        create_test_image(f.name)
        path = f.name

    try:
        img = load_image(path)
        img = preprocess(img)
        colors, _ = quantize_colors(img, n_colors=4, quality="low")
        layers = build_color_layers(img, colors, posterize=True)
        assert len(layers) > 0, "Should produce layers"

        paths = trace_all_layers(layers, smooth=3)
        assert len(paths) > 0, "Should produce paths"
        assert all(vp.path_data for vp in paths), "Paths should have data"
        print("PASS: vectorization")
    finally:
        os.unlink(path)


def test_svg_generation():
    """Test SVG file generation."""
    from vectormage.processor import load_image, preprocess
    from vectormage.palettes import quantize_colors, build_color_layers
    from vectormage.tracer import trace_all_layers
    from vectormage.svg_writer import generate_svg, write_svg

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        create_test_image(f.name)
        path = f.name

    out_path = path.replace(".png", ".svg")
    try:
        img = load_image(path)
        h, w = img.shape[:2]
        img = preprocess(img)
        colors, _ = quantize_colors(img, n_colors=4, quality="low")
        layers = build_color_layers(img, colors, posterize=True)
        paths = trace_all_layers(layers, smooth=3)
        svg = generate_svg(paths, w, h)
        write_svg(svg, out_path)

        assert os.path.exists(out_path), "SVG file should exist"
        with open(out_path) as f:
            content = f.read()
        assert content.startswith("<?xml"), "Should be valid XML"
        assert "<svg" in content, "Should contain SVG element"
        print("PASS: svg_generation")
    finally:
        os.unlink(path)
        if os.path.exists(out_path):
            os.unlink(out_path)


def test_cli_help():
    """Test CLI help output."""
    import subprocess
    result = subprocess.run(
        ["python", "-m", "vectormage", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, "Help should exit 0"
    assert "vectormage" in result.stdout.lower(), "Help should mention vectormage"
    assert "--quality" in result.stdout, "Help should show --quality"
    assert "--colors" in result.stdout, "Help should show --colors"
    print("PASS: cli_help")


def test_full_pipeline():
    """Test end-to-end: create image -> vectorize -> verify SVG."""
    from vectormage.processor import load_image, preprocess
    from vectormage.palettes import quantize_colors, build_color_layers
    from vectormage.tracer import trace_all_layers
    from vectormage.svg_writer import generate_svg, write_svg
    from vectormage.optimizer import optimize_paths

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        create_test_image(f.name)
        path = f.name

    out_path = path.replace(".png", ".svg")
    try:
        img = load_image(path)
        h, w = img.shape[:2]
        img = preprocess(img, do_denoise=True)
        colors, _ = quantize_colors(img, n_colors=8, quality="medium")
        layers = build_color_layers(img, colors, posterize=True)
        paths = trace_all_layers(layers, smooth=5)
        paths = optimize_paths(paths)
        svg = generate_svg(paths, w, h, optimize=True)
        write_svg(svg, out_path)

        assert os.path.exists(out_path)
        size = os.path.getsize(out_path)
        assert size > 100, f"SVG should have content, got {size} bytes"
        print(f"PASS: full_pipeline ({size} bytes SVG)")
    finally:
        os.unlink(path)
        if os.path.exists(out_path):
            os.unlink(out_path)


if __name__ == "__main__":
    test_preprocessing()
    test_quantization()
    test_vectorization()
    test_svg_generation()
    test_cli_help()
    test_full_pipeline()
    print("\nAll tests passed!")
