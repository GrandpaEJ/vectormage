"""CLI interface with Click framework."""

import os
import glob as globmod
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .processor import load_image, preprocess
from .palettes import quantize_colors, build_color_layers
from .tracer import trace_all_layers
from .svg_writer import generate_svg, write_svg
from .optimizer import optimize_paths

console = Console()

QUALITY_HELP = "Quality level: low (8 colors), medium (16), high (32), max (64)"


def process_single(
    input_path: str,
    output_path: str | None,
    quality: str,
    colors: int | None,
    posterize: bool,
    gradient: bool,
    denoise: bool,
    sharpen: bool,
    contrast: float,
    threshold_val: int | None,
    optimize: bool,
    smooth: int,
    stroke_width: float,
    preview: bool,
) -> str | None:
    """Process a single image file."""
    try:
        img_array = load_image(input_path)
        h, w = img_array.shape[:2]

        # Preprocess
        img_array = preprocess(
            img_array,
            do_denoise=denoise,
            do_sharpen=sharpen,
            contrast=contrast,
            thresh=threshold_val,
        )

        # Quantize colors
        colors_list, proportions = quantize_colors(
            img_array, n_colors=colors, quality=quality
        )

        # Build layers
        layers = build_color_layers(img_array, colors_list, posterize=posterize)

        # Trace paths
        eps = 0.001 if quality in ("high", "max") else 0.002
        if quality == "low":
            eps = 0.005
        paths = trace_all_layers(layers, smooth=smooth, epsilon_factor=eps)

        # Optimize
        if optimize:
            paths = optimize_paths(paths)

        # Generate SVG
        svg_content = generate_svg(paths, w, h, optimize=optimize, stroke_width=stroke_width)

        if preview:
            console.print(svg_content)
            return None

        # Determine output path
        if output_path and os.path.isdir(output_path):
            base = Path(input_path).stem
            output_path = os.path.join(output_path, f"{base}.svg")
        elif not output_path:
            output_path = str(Path(input_path).with_suffix(".svg"))

        write_svg(svg_content, output_path)
        return output_path

    except Exception as e:
        console.print(f"[red]Error processing {input_path}: {e}[/red]")
        return None


@click.command()
@click.argument("inputs", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("-o", "--output", default=None, help="Output file or directory")
@click.option("-q", "--quality", default="medium", type=click.Choice(["low", "medium", "high", "max"]), help=QUALITY_HELP)
@click.option("-c", "--colors", default=None, type=int, help="Number of colors (default: auto based on quality)")
@click.option("--posterize", is_flag=True, help="Flat color regions (good for illustrations)")
@click.option("--gradient", is_flag=True, help="Preserve gradients with semi-transparent layers")
@click.option("--denoise", is_flag=True, help="Gaussian blur before vectorizing")
@click.option("--sharpen", is_flag=True, help="Unsharp mask enhancement")
@click.option("--contrast", default=1.0, type=float, help="Contrast adjustment (1.0 = normal)")
@click.option("--threshold", default=None, type=int, help="Black/white threshold (0-255)")
@click.option("--optimize", is_flag=True, help="Simplify paths (reduce SVG size)")
@click.option("--smooth", default=5, type=int, help="Bezier curve smoothing level (0-10)")
@click.option("--stroke-width", default=0.0, type=float, help="Use stroked paths instead of filled")
@click.option("--preview", is_flag=True, help="Print SVG to stdout instead of writing file")
def cli(
    inputs,
    output,
    quality,
    colors,
    posterize,
    gradient,
    denoise,
    sharpen,
    contrast,
    threshold,
    optimize,
    smooth,
    stroke_width,
    preview,
):
    """vectormage - High-quality PNG/JPG to SVG vectorizer.

    Convert raster images to clean vector SVGs with multiple quality levels.

    \b
    Examples:
      vectormage input.png -o output.svg
      vectormage photo.jpg -o vector.svg --quality high --colors 64
      vectormage illustration.png --posterize --colors 16
      vectormage *.png -o output_dir/ --quality high
    """
    smooth = max(0, min(10, smooth))

    # Expand glob patterns
    all_inputs = []
    for inp in inputs:
        expanded = globmod.glob(inp)
        if expanded:
            all_inputs.extend(expanded)
        else:
            all_inputs.append(inp)

    if not all_inputs:
        console.print("[red]No input files found.[/red]")
        raise SystemExit(1)

    console.print(f"[bold green]vectormage v0.1.0[/bold green]")
    console.print(f"  Quality: {quality} | Colors: {colors or 'auto'} | Smooth: {smooth}")

    # Create output directory if needed
    if output and not preview and not os.path.isdir(output) and len(all_inputs) > 1:
        os.makedirs(output, exist_ok=True)

    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing...", total=len(all_inputs))
        for input_path in all_inputs:
            progress.update(task, description=f"Processing {Path(input_path).name}...")
            result = process_single(
                input_path,
                output,
                quality,
                colors,
                posterize,
                gradient,
                denoise,
                sharpen,
                contrast,
                threshold,
                optimize,
                smooth,
                stroke_width,
                preview,
            )
            if result:
                results.append(result)
            progress.advance(task)

    if results:
        console.print(f"\n[bold green]Done![/bold green] Generated {len(results)} SVG(s):")
        for r in results:
            console.print(f"  -> {r}")
    elif not preview:
        console.print("[yellow]No SVG files were generated.[/yellow]")


if __name__ == "__main__":
    cli()
