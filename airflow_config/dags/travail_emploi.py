from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from notifier.notifications_template import (
    get_failure_notifier,
    get_start_notifier,
    get_success_notifier,
)

from config import HF_TOKEN
from database import create_all_tables, export_table_to_parquet
from download_and_processing import download_and_optionally_process_files
from utils.hugging_face import upload_dataset_task

default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 8, 1),
    "retries": 3,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    "TRAVAIL_EMPLOI",
    default_args=default_args,
    schedule=None,
    catchup=False,
    max_active_runs=1,
    description="TRAVAIL-EMPLOI.GOUV.FR data processing pipeline",
    tags=["mediatech", "travail_emploi"],
    params={
        "table_name": "travail_emploi",
        "model": "BAAI/bge-m3",
        "private": False,
        "repository": "AgentPublic",
    },
) as dag:
    create_tables = PythonOperator(
        task_id="create_tables",
        python_callable=create_all_tables,
        op_kwargs={"delete_existing": False, "model": "{{ params.model }}"},
        on_execute_callback=get_start_notifier(),
        on_success_callback=get_success_notifier(),
        on_failure_callback=get_failure_notifier(),
    )

    download_and_process_files = PythonOperator(
        task_id="download_and_process_files",
        python_callable=download_and_optionally_process_files,
        op_kwargs={
            "table_name": "{{ params.table_name }}",
            "process": True,
            "model": "{{ params.model }}",
        },
        on_execute_callback=get_start_notifier(),
        on_success_callback=get_success_notifier(),
        on_failure_callback=get_failure_notifier(),
    )

    export_table = PythonOperator(
        task_id="export_table",
        python_callable=export_table_to_parquet,
        op_kwargs={"table_name": "{{ params.table_name }}"},
        on_execute_callback=get_start_notifier(),
        on_success_callback=get_success_notifier(),
        on_failure_callback=get_failure_notifier(),
    )

    upload_dataset = PythonOperator(
        task_id="upload_dataset",
        python_callable=upload_dataset_task,
        op_kwargs={
            "dataset_name": "travail-emploi",
            "token": HF_TOKEN,
            "repository": "{{ params.repository }}",
            "private": "{{ params.private }}",
        },
        on_execute_callback=get_start_notifier(),
        on_success_callback=get_success_notifier(),
        on_failure_callback=get_failure_notifier(),
    )

    create_tables >> download_and_process_files >> export_table >> upload_dataset
