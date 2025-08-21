#!/usr/bin/env python3
"""
Test prediction for a single GPX file using the saved model.
This shows exactly what the app would predict, helping debug discrepancies.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
import pandas as pd
from rich.console import Console
from rich.table import Table

from utils.persistence import load_pace_model_from_disk, fmt
from utils.prediction import run_prediction_simulation
from models import Course
import config

app = typer.Typer()
console = Console()


@app.command()
def predict(
        gpx_file: str = typer.Argument(..., help="Path to GPX file"),
        aid_stations: str = typer.Option("", "--aid-stations", "-a",
                                         help="Aid station distances (km), e.g. '10,25,40'"),
        feel: str = typer.Option("ok", "--feel", "-f",
                                 help="How you feel: good/ok/meh"),
        heat: str = typer.Option("moderate", "--heat", "-h",
                                 help="Weather: cool/moderate/hot"),
        verbose: bool = typer.Option(False, "--verbose", "-v",
                                     help="Show detailed debug information")
):
    """
    Test prediction for a single GPX file using your saved pace model.

    Example:
        python scripts/test_prediction.py my_race.gpx --aid-stations "10,25,40"
    """

    # Load the saved pace model
    console.print("[blue]Loading saved pace model...[/blue]")
    pace_model = load_pace_model_from_disk()

    if not pace_model:
        console.print("[red]❌ No saved pace model found. Build one in the app first.[/red]")
        raise typer.Exit(1)

    # Load GPX file
    gpx_path = Path(gpx_file)
    if not gpx_path.exists():
        console.print(f"[red]❌ GPX file not found: {gpx_file}[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Loading GPX: {gpx_path.name}[/blue]")
    with open(gpx_path, 'rb') as f:
        gpx_bytes = f.read()

    # Create course
    course = Course(gpx_bytes, aid_stations, "km")

    # Show course details
    console.print("\n[bold]📍 Course Details[/bold]")
    course_table = Table(show_header=False)
    course_table.add_column("Metric", style="cyan")
    course_table.add_column("Value", style="white")

    course_table.add_row("Distance", f"{course.total_km:.1f} km")
    course_table.add_row("Elevation Gain", f"{course.gain_m:.0f} m")
    course_table.add_row("Elevation Loss", f"{course.loss_m:.0f} m")
    course_table.add_row("Median Altitude", f"{course.median_altitude:.0f} m")
    course_table.add_row("Min/Max Elevation", f"{course.min_ele:.0f}m / {course.max_ele:.0f}m")

    console.print(course_table)

    # Run prediction
    console.print(f"\n[blue]Running prediction (feel={feel}, heat={heat})...[/blue]")
    results = run_prediction_simulation(course, pace_model, feel, heat)

    p10 = results["p10"]
    p50 = results["p50"]
    p90 = results["p90"]
    meta = results["metadata"]

    # Show prediction results
    console.print("\n[bold]⏱️ Predicted Finish Times[/bold]")

    results_table = Table(show_header=True, header_style="bold magenta")
    results_table.add_column("Percentile", style="cyan")
    results_table.add_column("Time", style="green")
    results_table.add_column("Pace (min/km)", style="yellow")

    finish_p10 = p10[-1] if len(p10) > 0 else 0
    finish_p50 = p50[-1] if len(p50) > 0 else 0
    finish_p90 = p90[-1] if len(p90) > 0 else 0

    results_table.add_row(
        "P10 (Optimistic)",
        fmt(finish_p10),
        f"{(finish_p10 / 60) / course.total_km:.2f}"
    )
    results_table.add_row(
        "P50 (Best Guess)",
        fmt(finish_p50),
        f"{(finish_p50 / 60) / course.total_km:.2f}"
    )
    results_table.add_row(
        "P90 (Pessimistic)",
        fmt(finish_p90),
        f"{(finish_p90 / 60) / course.total_km:.2f}"
    )

    console.print(results_table)

    # Show model details
    if verbose:
        console.print("\n[bold]🔧 Model Details[/bold]")

        details_table = Table(show_header=False)
        details_table.add_column("Parameter", style="cyan")
        details_table.add_column("Value", style="white")

        details_table.add_row("Altitude Factor", f"{meta['alt_speed_factor']:.3f}")
        details_table.add_row("Riegel k", f"{meta['riegel_k']:.3f}")
        details_table.add_row("Riegel Applied", str(meta['riegel_applied']))

        if meta['riegel_applied']:
            details_table.add_row("Riegel Scale Factor", f"{meta['riegel_scale_factor']:.3f}")
            if meta.get('ref_distance_km'):
                details_table.add_row("Reference Distance", f"{meta['ref_distance_km']:.1f} km")
                details_table.add_row("Reference Time", fmt(meta.get('ref_time_s', 0)))

        ultra_applied = meta.get('slow_factor_finish', 1.0) > 1.0
        details_table.add_row("Ultra Adjustments", str(ultra_applied))

        if ultra_applied:
            details_table.add_row("Ultra Slow Factor", f"{meta.get('slow_factor_finish', 1):.3f}")
            details_table.add_row("Rest Added", fmt(meta.get('rest_added_finish_s', 0)))

        console.print(details_table)

        # Show checkpoint predictions if aid stations were provided
        if aid_stations and len(p50) > 1:
            console.print("\n[bold]📊 Checkpoint Predictions[/bold]")

            cp_table = Table(show_header=True, header_style="bold yellow")
            cp_table.add_column("Checkpoint", style="cyan")
            cp_table.add_column("Distance", style="white")
            cp_table.add_column("P50 Time", style="green")
            cp_table.add_column("Pace", style="yellow")

            for i, km in enumerate(course.leg_end_km):
                if i < len(course.leg_end_km) - 1:
                    name = f"AS{i + 1}"
                else:
                    name = "Finish"

                time_s = p50[i] if i < len(p50) else 0
                pace = (time_s / 60) / km if km > 0 else 0

                cp_table.add_row(
                    name,
                    f"{km:.1f} km",
                    fmt(time_s),
                    f"{pace:.2f} min/km"
                )

            console.print(cp_table)

    # Show comparison with actual time if provided
    actual_time = typer.prompt(
        "\nEnter actual finish time (HH:MM:SS) for comparison, or press Enter to skip",
        default="",
        show_default=False
    )

    if actual_time:
        try:
            parts = actual_time.split(":")
            actual_s = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

            error_s = finish_p50 - actual_s
            error_pct = (error_s / actual_s) * 100

            console.print("\n[bold]📈 Comparison with Actual[/bold]")

            comp_table = Table(show_header=False)
            comp_table.add_column("Metric", style="cyan")
            comp_table.add_column("Value", style="white")

            comp_table.add_row("Actual Time", fmt(actual_s))
            comp_table.add_row("Predicted (P50)", fmt(finish_p50))
            comp_table.add_row("Error", f"{fmt(abs(error_s))} ({error_pct:+.1f}%)")
            comp_table.add_row("Within P10-P90?",
                               "✅ Yes" if finish_p10 <= actual_s <= finish_p90 else "❌ No")

            console.print(comp_table)

            if abs(error_pct) < 5:
                console.print("[green]Excellent prediction accuracy![/green]")
            elif abs(error_pct) < 10:
                console.print("[yellow]Good prediction accuracy[/yellow]")
            else:
                console.print("[red]Large prediction error - investigate model[/red]")

        except Exception as e:
            console.print(f"[red]Could not parse time: {e}[/red]")


if __name__ == "__main__":
    app()