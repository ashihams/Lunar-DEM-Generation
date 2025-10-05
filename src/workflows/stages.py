"""
Workflow Stage Helpers

This module exposes three stage functions used by Airflow:
- stage1_collect_raw: Query ODE and download top 5 thumbnails for a crater; upload to S3 raw.
- stage2_preprocess: Run lightweight preprocess using existing pipeline on the raw dir; upload to S3 staging.
- stage3_psfs: Placeholder for deep-learning P-SFS step; currently copies preprocessed to results and uploads to processed.

These functions are small wrappers around existing modules `api_client.py`, `pipeline.py`, and `s3_manager.py`.
"""

import os
import json
import uuid
import logging
from pathlib import Path
from typing import Dict

from src.api_client import ODEApiClient
from src.s3_manager import DataLakePipeline


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ensure_dirs(*dirs: str):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def _resolve_local_root(local_root: str) -> str:
    """Resolve a local root under the mounted project directory when running in Airflow.

    Preference order:
    1) APP_ROOT env if set
    2) /opt/airflow/app if it exists (compose mount)
    3) CWD
    """
    if os.path.isabs(local_root):
        return local_root
    app_root = os.environ.get("APP_ROOT")
    if app_root and os.path.isdir(app_root):
        return str(Path(app_root) / local_root)
    airflow_app = "/opt/airflow/app"
    if os.path.isdir(airflow_app):
        return str(Path(airflow_app) / local_root)
    return str(Path.cwd() / local_root)


def stage1_collect_raw(crater_name: str, run_id: str, local_root: str = "./data") -> Dict:
    """Stage 1: Collect raw data and push to S3 raw layer.

    Returns metadata dict written to S3.
    """
    local_root = _resolve_local_root(local_root)
    local_raw = os.path.join(local_root, "raw", crater_name, run_id)
    _ensure_dirs(local_raw)

    api = ODEApiClient()
    result = api.collect_crater_data(crater_name, output_dir=local_raw)

    # upload to S3 raw
    lake = DataLakePipeline()
    meta = lake.ingest_raw_data(local_raw, crater_name, run_id)
    meta["api_result"] = result
    return meta


def stage2_preprocess(crater_name: str, run_id: str, local_root: str = "./data") -> Dict:
    """Stage 2: Run preprocessing pipeline and promote to staging layer."""
    local_root = _resolve_local_root(local_root)
    local_raw = os.path.join(local_root, "raw", crater_name, run_id)
    local_pre = os.path.join(local_root, "staging", crater_name, run_id)
    _ensure_dirs(local_pre)

    # Lazy import heavy modules to avoid Airflow DAG import failures (no torch installed)
    from src.pipeline import LunarDLTPipeline, create_default_config

    if not os.path.exists("config.json"):
        create_default_config()

    pipeline = LunarDLTPipeline("config.json")
    # Process entire raw directory; save outputs to local_pre
    pipeline.process_dataset(local_raw, local_pre)

    lake = DataLakePipeline()
    return lake.promote_to_staging(crater_name, run_id, local_pre)


def stage3_psfs(crater_name: str, run_id: str, local_root: str = "./data") -> Dict:
    """Stage 3: P‑SFS deep learning step followed by publish to processed.

    If environment variable `PSFS_MODEL_PATH` is set and exists, runs the
    model on the preprocessed images to generate `<name>_elevation.tif` and
    `<name>_confidence.png`. Otherwise, falls back to copying preprocessed
    outputs to the results folder.
    """
    local_root = _resolve_local_root(local_root)
    local_pre = os.path.join(local_root, "staging", crater_name, run_id)
    local_out = os.path.join(local_root, "processed", crater_name, run_id)
    _ensure_dirs(local_out)

    model_path = os.environ.get("PSFS_MODEL_PATH")
    processed = 0
    if model_path and os.path.exists(model_path):
        try:
            # Use the full DEMGenerationPipeline when torch is available
            from src.sfs.pipeline import DEMGenerationPipeline
            logger.info(f"Running SFS→DEM pipeline with model: {model_path}")
            # For simplicity, process each preprocessed image separately
            pipeline = DEMGenerationPipeline()
            for p in Path(local_pre).glob("**/*.png"):
                img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                metadata = {"resolution": img.shape[:2], "product_id": p.name}
                out_dir = Path(local_out) / p.stem
                pipeline.run(img, metadata, str(out_dir))
                processed += 1
        except Exception as e:
            logger.warning(f"SFS pipeline unavailable ({e}); copying outputs instead")
            model_path = None
    else:
        logger.warning("PSFS_MODEL_PATH not set; copying preprocessed outputs as placeholder")
        for p in Path(local_pre).glob("**/*"):
            if p.is_file():
                rel = p.relative_to(local_pre)
                dst = Path(local_out) / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(p.read_bytes())

    lake = DataLakePipeline()
    meta = lake.publish_results(crater_name, run_id, local_out)
    meta["psfs_processed"] = processed
    return meta


