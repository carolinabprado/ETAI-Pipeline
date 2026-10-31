"""
Saving each run's results to disk.

Printing to the terminal is fine while you're watching it happen, but it's gone the moment you scroll past it or close the window. This module writes the full report (every metric, confusion matrix, classification report, fairness table) to a timestamped file in `results/` instead, so you can open it again later. From week 7 every run also adds one row to the scoreboard (`results/runs.csv`), so runs can be compared side by side.
"""
import os

import pandas as pd


def save_scores(path: str, row: dict) -> pd.DataFrame:
    """Appends one run to the CSV at `path` (created if needed) and returns all runs."""
    runs = pd.read_csv(path, dtype={"run": str}) if os.path.exists(path) else pd.DataFrame()
    runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    runs.to_csv(path, index=False)
    return runs


def save_run(results_dir: str, run: str, config: dict, report_text: str) -> str:
    """
    Writes one run's full report to `results_dir/run_<run>.txt` (the folder is created if
    needed) and returns the path. `run` is the run id from main.py, the same one that goes
    on the scoreboard, so a scoreboard row and its report file can be matched.
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"run_{run}.txt")

    tuning = config.get("tuning", {})
    tuning_line = (f"Tuning: n_trials={tuning['n_trials']}  n_splits={tuning['n_splits']}  "
                   f"random_state={tuning['random_state']}\n") if tuning.get("enabled") else "Tuning: disabled\n"

    cv = config["cv"]
    header = (
        f"Run: {run}\n"
        f"Model: {config['model']['type']}  params={config['model'].get('params')}\n"
        f"Preprocessing: encoder={config['preprocessing']['encoder']}  "
        f"scaler={config['preprocessing']['scaler']}\n"
        f"CV: {cv['n_splits']} stratified folds  shuffle={cv.get('shuffle', True)}  "
        f"random_state={cv.get('random_state')}\n"
        f"Metrics: {', '.join(cv['metrics'])}  (selection metric: {cv['selection_metric']})\n"
        f"Locked test set: size={config['test_set']['size']}  "
        f"random_state={config['test_set']['random_state']}  (not evaluated)\n"
        f"{tuning_line}"
        + "=" * 60 + "\n\n"
    )

    with open(path, "w") as f:
        f.write(header + report_text)

    return path

