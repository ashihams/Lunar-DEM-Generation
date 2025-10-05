"""
Airflow DAG to orchestrate Lunar DEM pipeline:
  Stage 1: Data Collection (ODE → local → S3 raw)
  Stage 2: Preprocessing   (local raw → preprocess → S3 staging)
  Stage 3: P-SFS DL        (placeholder) and publish to S3 processed

This DAG is designed to run inside an Airflow environment (Composer, MWAA, or DIY).
Set environment variables for crater name and schedule if needed.
"""

from datetime import datetime, timedelta
import os

from airflow import DAG
from airflow.operators.python import PythonOperator

# Robust import whether project is on PYTHONPATH as package or mounted directory
try:
    from src.workflows.stages import stage1_collect_raw, stage2_preprocess, stage3_psfs
except Exception:
    import sys
    import pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    from src.workflows.stages import stage1_collect_raw, stage2_preprocess, stage3_psfs


CRATER_NAME = os.environ.get("CRATER_NAME", "Tycho")
RUN_ID = os.environ.get("RUN_ID", datetime.utcnow().strftime("%Y%m%dT%H%M%S"))


default_args = {
    "owner": "lunar",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="lunar_dem_pipeline",
    default_args=default_args,
    description="Lunar DEM data lake pipeline",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    def _s1(**context):
        return stage1_collect_raw(CRATER_NAME, RUN_ID)

    def _s2(**context):
        return stage2_preprocess(CRATER_NAME, RUN_ID)

    def _s3(**context):
        return stage3_psfs(CRATER_NAME, RUN_ID)

    t1 = PythonOperator(task_id="stage1_collect_raw", python_callable=_s1)
    t2 = PythonOperator(task_id="stage2_preprocess", python_callable=_s2)
    t3 = PythonOperator(task_id="stage3_psfs", python_callable=_s3)

    t1 >> t2 >> t3


