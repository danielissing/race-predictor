"""
Data persistence functions for saving and loading models/credentials
"""

import os
import json
import pandas as pd
from pathlib import Path
import streamlit as st
import config

def save_pace_model_to_disk(pace_model):
    """
    Persist pace model to CSV/JSON files.

    Args:
        pace_model: PaceModel object to save
    """
    try:
        Path(config.DATA_DIR).mkdir(exist_ok=True)
        pace_model.pace_df.to_csv(config.PACE_CURVES_PATH, index=False)
        pace_model.used_races.to_csv(config.USED_RACES_PATH, index=False)
        with open(config.MODEL_META_PATH, "w") as f:
            json.dump(pace_model.meta, f)
    except Exception as e:
        st.warning(f"Could not save model to disk: {e}")


def load_pace_model_from_disk():
    """
    Load pace model from disk if available.

    Returns:
        PaceModel object if successfully loaded, None otherwise
    """
    if not all([
        os.path.exists(config.PACE_CURVES_PATH),
        os.path.exists(config.USED_RACES_PATH),
        os.path.exists(config.MODEL_META_PATH)
    ]):
        return None

    try:
        from models import PaceModel
        pace_df = pd.read_csv(config.PACE_CURVES_PATH)
        used_races = pd.read_csv(config.USED_RACES_PATH)
        with open(config.MODEL_META_PATH, "r") as f:
            meta = json.load(f)
        return PaceModel(pace_df, used_races, meta)
    except Exception as e:
        print(f"Error loading pace model from disk: {e}")
        return None

def load_saved_app_creds( app_credits_path: str):
    try:
        with open(app_credits_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_app_creds(client_id: str, client_secret: str, data_dir: str, app_credits_path: str):
    os.makedirs(data_dir, exist_ok=True)
    with open(app_credits_path, "w") as f:
        json.dump({"client_id": str(client_id), "client_secret": str(client_secret)}, f)

def forget_app_creds(app_credits_path: str):
    try:
        os.remove(app_credits_path)
    except FileNotFoundError:
        pass


def load_excluded_race_ids() -> set:
    """Load set of race IDs the user has chosen to exclude."""
    path = config.EXCLUDED_RACES_PATH
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path)
        return set(df["id"].astype(str))
    except Exception:
        return set()


def save_excluded_race_ids(excluded_ids: set):
    """Persist the set of excluded race IDs to CSV."""
    Path(config.DATA_DIR).mkdir(exist_ok=True)
    df = pd.DataFrame({"id": sorted(excluded_ids)})
    df.to_csv(config.EXCLUDED_RACES_PATH, index=False)


def load_streams(race_id) -> dict | None:
    """Load cached streams JSON. Returns None if missing or empty."""
    path = os.path.join(config.CACHE_DIR, f"streams_{race_id}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if not data:
            return None
        return data
    except (json.JSONDecodeError, IOError):
        return None