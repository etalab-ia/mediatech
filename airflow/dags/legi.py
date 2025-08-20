from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 8, 1),
    "retries": 3,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    "LEGI",
    default_args=default_args,
    schedule=None,
    catchup=False,
    max_active_runs=1,
    description="LEGI data processing pipeline",
    tags=["mediatech", "legi"],
) as dag:
    wait_for_dole = ExternalTaskSensor(
        task_id="wait_for_dole",
        external_dag_id="DOLE",
        external_task_id="upload_dataset",
        allowed_states=["success"],
        mode="reschedule",  # Reschedule mode to avoid blocking the scheduler
        timeout=7 * 24 * 60 * 60,  # Wait up to 7 days, after which the task will fail
        poke_interval=120,  # Check every 2 minutes if the task has completed
    )
    create_tables = BashOperator(
        task_id="create_tables",
        bash_command="mediatech create_tables --model BAAI/bge-m3",
    )

    download_and_process_files = BashOperator(
        task_id="download_and_process_files",
        bash_command="mediatech download_and_process_files --source legi --model BAAI/bge-m3",
    )

    export_tables = BashOperator(
        task_id="export_tables", bash_command="mediatech export_tables"
    )

    upload_dataset = BashOperator(
        task_id="upload_dataset",
        bash_command="mediatech upload_dataset --dataset-name legi",
    )

    (
        wait_for_dole
        >> create_tables
        >> download_and_process_files
        >> export_tables
        >> upload_dataset
    )
