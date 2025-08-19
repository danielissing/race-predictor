#!/usr/bin/env python3
"""
Validate race predictions against historical performance.

This script backtests the prediction model by:
1. For each race, building a model using only races that occurred before it
2. Predicting that race's finish time
3. Comparing predictions to actual times
4. Generating metrics and visualizations

Usage:
    python scripts/validate_races.py --help
    python scripts/validate_races.py --max-races 20 --output-dir data/validation
"""

import os
import json
import hashlib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from typing import Dict, List, Optional, Tuple

import typer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.progress import track
from rich.table import Table

# Import from newly refactored modules
from utils.strava import ensure_token, list_activities, get_activity_streams, is_run, is_race
from utils.pace_builder import build_pace_curves_from_races
from utils.gpx_parsing import parse_gpx
from utils.elevation import resample_with_grade, segment_stats
from utils.course_analysis import legs_from_aid_stations, distance_by_grade_bins
from utils.performance import altitude_impairment_multiplicative, apply_ultra_adjustments_progressive
from utils.simulation import simulate_etas
from utils.prediction import run_prediction_simulation
from utils.persistence import load_saved_app_creds, fmt
from models import Course, PaceModel
import config

app = typer.Typer()
console = Console()


def construct_gpx_from_streams(streams: Dict) -> Optional[bytes]:
    """
    Construct GPX file content from Strava streams.

    Args:
        streams: Strava activity streams containing latlng and altitude data

    Returns:
        GPX file content as bytes, or None if insufficient data
    """
    latlng_data = streams.get("latlng", {}).get("data")
    if not latlng_data:
        return None

    altitude_data = streams.get("altitude", {}).get("data", [])
    distance_data = streams.get("distance", {}).get("data", [])

    # Build GPX with elevation if available
    gpx_content = """<?xml version="1.0" encoding="UTF-8"?>
<gpx creator="RacePredictorValidator" version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
  <trk>
    <name>Validation Race</name>
    <trkseg>
"""

    for i, (lat, lon) in enumerate(latlng_data):
        ele_tag = ""
        if i < len(altitude_data):
            ele_tag = f"<ele>{altitude_data[i]}</ele>"

        gpx_content += f"      <trkpt lat=\"{lat}\" lon=\"{lon}\">{ele_tag}</trkpt>\n"

    gpx_content += """    </trkseg>
  </trk>
</gpx>
"""
    return gpx_content.encode('utf-8')


def create_uniform_checkpoints(total_km: float, checkpoint_interval_km: float) -> List[float]:
    """
    Create uniform checkpoint distances.

    Args:
        total_km: Total race distance in kilometers
        checkpoint_interval_km: Interval between checkpoints (e.g., 10km)

    Returns:
        List of cumulative distances to checkpoints (including finish)
    """
    checkpoints = []
    current_km = checkpoint_interval_km

    while current_km < total_km - 0.001:
        checkpoints.append(current_km)
        current_km += checkpoint_interval_km

    # Always include finish
    checkpoints.append(total_km)
    return checkpoints


def extract_actual_splits(streams: Dict, checkpoint_km: List[float]) -> np.ndarray:
    """
    Extract actual race times at checkpoint distances.

    Args:
        streams: Strava streams with distance and time data
        checkpoint_km: List of checkpoint distances in km

    Returns:
        Array of actual times in seconds at each checkpoint
    """
    distance_data = streams.get("distance", {}).get("data", [])
    time_data = streams.get("time", {}).get("data", [])

    if not distance_data or not time_data:
        return np.array([])

    # Convert to numpy arrays
    distances_m = np.array(distance_data, dtype=float)
    times_s = np.array(time_data, dtype=float)

    # Ensure monotonic for interpolation
    distances_m = np.maximum.accumulate(distances_m)

    # Interpolate times at checkpoint distances
    checkpoint_m = np.array(checkpoint_km) * 1000.0
    actual_times = np.interp(checkpoint_m, distances_m, times_s)

    return actual_times


def validate_single_race(
        race: Dict,
        historical_races: List[Dict],
        access_token: str,
        checkpoint_interval_km: float = 10.0,
        recency_mode: str = "off"
) -> Optional[Dict]:
    """
    Validate prediction for a single race using historical data.

    Args:
        race: Race to predict
        historical_races: Races that occurred before this one
        access_token: Strava API token
        checkpoint_interval_km: Distance between validation checkpoints
        recency_mode: Recency weighting mode for building pace curves

    Returns:
        Dictionary with validation metrics, or None if validation fails
    """
    if not historical_races:
        return None

    try:
        # 1. Build pace model from historical races only
        pace_df, used_df, meta = build_pace_curves_from_races(
            access_token,
            historical_races,
            config.GRADE_BINS,
            max_activities=config.MAX_ACTIVITIES,
            recency_mode=recency_mode
        )

        if pace_df.empty:
            return None

        pace_model = PaceModel(pace_df, used_df, meta)

        # 2. Get race course data
        streams = get_activity_streams(access_token, race['id'])
        if not streams:
            return None

        gpx_bytes = construct_gpx_from_streams(streams)
        if not gpx_bytes:
            return None

        # 3. Create course with uniform checkpoints
        race_distance_km = race.get('distance', 0) / 1000.0
        checkpoint_km = create_uniform_checkpoints(race_distance_km, checkpoint_interval_km)

        # Convert to aid station string for Course constructor
        aid_km_text = ", ".join(str(km) for km in checkpoint_km[:-1])  # Exclude finish

        course = Course(gpx_bytes, aid_km_text, "km")

        # 4. Run prediction (matching app.py logic exactly)
        prediction_results = run_prediction_simulation(
            course, pace_model,
            feel="ok", heat="moderate"  # Neutral conditions for validation
        )

        p10 = prediction_results["p10"]
        p50 = prediction_results["p50"]
        p90 = prediction_results["p90"]
        metadata = prediction_results["metadata"]

        # 5. Get actual race times at checkpoints
        actual_times = extract_actual_splits(streams, checkpoint_km)

        if len(actual_times) == 0:
            return None

        # 6. Calculate validation metrics
        finish_actual_s = race['elapsed_time']
        finish_pred_s = float(p50[-1])
        finish_error_s = finish_pred_s - finish_actual_s
        finish_error_pct = (finish_error_s / finish_actual_s) * 100

        # Calculate errors at all checkpoints
        checkpoint_errors_pct = []
        for i, (pred, actual) in enumerate(zip(p50, actual_times)):
            if actual > 0:
                error_pct = abs((pred - actual) / actual) * 100
                checkpoint_errors_pct.append(error_pct)

        median_checkpoint_error = np.median(checkpoint_errors_pct) if checkpoint_errors_pct else 0

        # Check if actual finish was within P10-P90 range
        within_confidence = (p10[-1] <= finish_actual_s <= p90[-1])

        return {
            "race_id": race['id'],
            "race_name": race['name'],
            "race_date": race['start_date'][:10],
            "distance_km": round(race_distance_km, 1),
            "actual_time_s": finish_actual_s,
            "predicted_p10_s": float(p10[-1]),
            "predicted_p50_s": finish_pred_s,
            "predicted_p90_s": float(p90[-1]),
            "error_s": finish_error_s,
            "error_pct": finish_error_pct,
            "abs_error_pct": abs(finish_error_pct),
            "median_checkpoint_error_pct": median_checkpoint_error,
            "within_confidence": within_confidence,
            "n_historical_races": len(historical_races),
            "altitude_factor": metadata.get("alt_speed_factor", 1.0),
            "riegel_scale": metadata.get("riegel_scale_factor", 1.0),
            "ultra_adjustments_applied": metadata.get("slow_factor_finish", 1.0) > 1.0
        }

    except Exception as e:
        console.print(f"[yellow]Warning: Failed to validate {race['name']}: {e}[/yellow]")
        return None


def plot_validation_results(results_df: pd.DataFrame, output_dir: Path):
    """Create and save validation plots."""

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Race Prediction Validation Results', fontsize=16)

    # 1. Predicted vs Actual scatter plot
    ax1 = axes[0, 0]
    ax1.scatter(results_df['actual_time_s'] / 3600, results_df['predicted_p50_s'] / 3600, alpha=0.6)

    # Add confidence intervals
    ax1.errorbar(
        results_df['actual_time_s'] / 3600,
        results_df['predicted_p50_s'] / 3600,
        yerr=[
            (results_df['predicted_p50_s'] - results_df['predicted_p10_s']) / 3600,
            (results_df['predicted_p90_s'] - results_df['predicted_p50_s']) / 3600
        ],
        fmt='none', alpha=0.3, ecolor='gray'
    )

    # Perfect prediction line
    max_time = max(results_df['actual_time_s'].max(), results_df['predicted_p50_s'].max()) / 3600
    ax1.plot([0, max_time], [0, max_time], 'r--', label='Perfect Prediction', alpha=0.5)

    ax1.set_xlabel('Actual Finish Time (hours)')
    ax1.set_ylabel('Predicted Finish Time (hours)')
    ax1.set_title('Predicted vs Actual Race Times')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Error distribution histogram
    ax2 = axes[0, 1]
    ax2.hist(results_df['error_pct'], bins=20, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='r', linestyle='--', label='Perfect Prediction')
    ax2.set_xlabel('Prediction Error (%)')
    ax2.set_ylabel('Number of Races')
    ax2.set_title('Distribution of Prediction Errors')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Error vs Race Distance
    ax3 = axes[1, 0]
    ax3.scatter(results_df['distance_km'], results_df['abs_error_pct'], alpha=0.6)

    # Add trend line
    z = np.polyfit(results_df['distance_km'], results_df['abs_error_pct'], 1)
    p = np.poly1d(z)
    ax3.plot(results_df['distance_km'].sort_values(),
             p(results_df['distance_km'].sort_values()),
             "r--", alpha=0.5, label='Trend')

    ax3.set_xlabel('Race Distance (km)')
    ax3.set_ylabel('Absolute Error (%)')
    ax3.set_title('Prediction Error vs Race Distance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Error over time (learning curve)
    ax4 = axes[1, 1]
    ax4.plot(range(len(results_df)), results_df['abs_error_pct'], marker='o', alpha=0.6)

    # Add rolling average
    window = min(5, len(results_df) // 3)
    if window > 1:
        rolling_mean = results_df['abs_error_pct'].rolling(window=window, center=True).mean()
        ax4.plot(range(len(results_df)), rolling_mean, 'r-', linewidth=2,
                 label=f'{window}-race moving average')

    ax4.set_xlabel('Race Number (chronological)')
    ax4.set_ylabel('Absolute Error (%)')
    ax4.set_title('Model Performance Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / 'validation_plots.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    console.print(f"[green]Saved plots to {plot_path}[/green]")
    plt.show()


@app.command()
def validate(
        max_races: int = typer.Option(20, "--max-races", "-n",
                                      help="Maximum number of recent races to validate"),
        checkpoint_km: float = typer.Option(10.0, "--checkpoint-km", "-c",
                                            help="Distance between validation checkpoints (km)"),
        output_dir: str = typer.Option("data/validation", "--output-dir", "-o",
                                       help="Directory for output files"),
        recency_mode: str = typer.Option("off", "--recency", "-r",
                                         help="Recency weighting: off/mild/medium"),
        min_historical: int = typer.Option(3, "--min-historical", "-m",
                                           help="Minimum historical races needed for prediction"),
        plot: bool = typer.Option(True, "--plot/--no-plot",
                                  help="Generate visualization plots")
):
    """
    Backtest the race prediction model against your Strava race history.

    This performs leave-one-out style validation where each race is predicted
    using only the races that occurred before it, simulating real-world usage.
    """
    console.print("[bold blue]🏃 Race Prediction Validator[/bold blue]")
    console.print("=" * 50)

    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load credentials and connect to Strava
    console.print("📡 Connecting to Strava...")
    creds = load_saved_app_creds(config.APP_CREDS_PATH)
    if not creds.get("client_id") or not creds.get("client_secret"):
        console.print("[red]❌ No Strava credentials found. Please run the main app first.[/red]")
        raise typer.Exit(1)

    tokens = ensure_token(creds["client_id"], creds["client_secret"])
    if not tokens:
        console.print("[red]❌ Could not authenticate with Strava.[/red]")
        raise typer.Exit(1)

    # Fetch all activities
    console.print("📥 Fetching race history...")
    all_activities = list_activities(tokens["access_token"], per_page=200)

    # Filter to races only
    races = [a for a in all_activities if is_run(a) and is_race(a)]
    races.sort(key=lambda x: x.get("start_date", ""))  # Sort chronologically

    if len(races) < min_historical + 1:
        console.print(f"[red]❌ Need at least {min_historical + 1} races for validation.[/red]")
        raise typer.Exit(1)

    # Limit to max_races most recent
    if len(races) > max_races:
        races = races[-max_races:]

    console.print(f"Found {len(races)} races to validate")

    # Run validation for each race
    results = []
    skipped = 0

    for i in track(range(min_historical, len(races)),
                   description="Validating races...",
                   console=console):

        race_to_predict = races[i]
        historical_races = races[:i]  # Only races before this one

        result = validate_single_race(
            race_to_predict,
            historical_races,
            tokens["access_token"],
            checkpoint_interval_km=checkpoint_km,
            recency_mode=recency_mode
        )

        if result:
            results.append(result)
        else:
            skipped += 1

    if not results:
        console.print("[red]❌ No races could be validated.[/red]")
        raise typer.Exit(1)

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('race_date')

    # Save detailed results
    csv_path = output_path / 'validation_results.csv'
    results_df.to_csv(csv_path, index=False)
    console.print(f"[green]✅ Saved detailed results to {csv_path}[/green]")

    # Calculate summary statistics
    mae_seconds = results_df['error_s'].abs().mean()
    mape = results_df['abs_error_pct'].mean()
    bias = results_df['error_pct'].mean()
    median_checkpoint_error = results_df['median_checkpoint_error_pct'].median()
    confidence_hit_rate = results_df['within_confidence'].mean() * 100

    # Display summary table
    console.print("\n[bold]📊 Validation Summary[/bold]")
    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Races Validated", str(len(results_df)))
    summary_table.add_row("Races Skipped", str(skipped))
    summary_table.add_row("Mean Absolute Error", fmt(mae_seconds))
    summary_table.add_row("Mean Absolute % Error", f"{mape:.1f}%")
    summary_table.add_row("Mean Bias", f"{bias:+.1f}%")
    summary_table.add_row("Median Checkpoint Error", f"{median_checkpoint_error:.1f}%")
    summary_table.add_row("P10-P90 Hit Rate", f"{confidence_hit_rate:.0f}%")

    console.print(summary_table)

    # Display worst predictions
    console.print("\n[bold]🎯 Best and Worst Predictions[/bold]")

    best_worst_table = Table(show_header=True, header_style="bold yellow")
    best_worst_table.add_column("Type", style="cyan")
    best_worst_table.add_column("Race", style="white")
    best_worst_table.add_column("Date", style="white")
    best_worst_table.add_column("Error", style="green")

    best = results_df.nsmallest(3, 'abs_error_pct')
    for _, row in best.iterrows():
        best_worst_table.add_row(
            "✅ Best",
            row['race_name'][:30],
            row['race_date'],
            f"{row['error_pct']:+.1f}%"
        )

    worst = results_df.nlargest(3, 'abs_error_pct')
    for _, row in worst.iterrows():
        best_worst_table.add_row(
            "❌ Worst",
            row['race_name'][:30],
            row['race_date'],
            f"{row['error_pct']:+.1f}%"
        )

    console.print(best_worst_table)

    # Save summary to JSON
    summary = {
        "n_races_validated": len(results_df),
        "n_races_skipped": skipped,
        "mean_absolute_error_s": float(mae_seconds),
        "mean_absolute_pct_error": float(mape),
        "mean_bias_pct": float(bias),
        "median_checkpoint_error_pct": float(median_checkpoint_error),
        "p10_p90_hit_rate": float(confidence_hit_rate),
        "checkpoint_interval_km": checkpoint_km,
        "recency_mode": recency_mode,
        "min_historical_races": min_historical
    }

    json_path = output_path / 'validation_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    console.print(f"[green]✅ Saved summary to {json_path}[/green]")

    # Generate plots
    if plot and len(results_df) > 1:
        console.print("\n📈 Generating plots...")
        plot_validation_results(results_df, output_path)

    # Provide interpretation
    console.print("\n[bold]💡 Interpretation Guide[/bold]")
    if mape < 5:
        console.print("[green]Excellent accuracy! Model is very reliable.[/green]")
    elif mape < 10:
        console.print("[yellow]Good accuracy. Model is generally reliable.[/yellow]")
    elif mape < 15:
        console.print("[yellow]Moderate accuracy. Consider more training data.[/yellow]")
    else:
        console.print("[red]Lower accuracy. Model may need improvement.[/red]")

    if abs(bias) > 5:
        if bias > 0:
            console.print("[yellow]Model tends to overestimate times (conservative).[/yellow]")
        else:
            console.print("[yellow]Model tends to underestimate times (optimistic).[/yellow]")

    console.print(f"\n[dim]Validation complete. Results saved to {output_path}/[/dim]")


@app.command()
def quick_test():
    """
    Quick test with just the last 5 races for debugging.
    """
    validate(max_races=5, checkpoint_km=10.0, plot=False)


if __name__ == "__main__":
    app()