from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 8, 1),
    "retries": 3,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    "CNIL",
    default_args=default_args,
    schedule=None,
    catchup=False,
    max_active_runs=1,
    description="CNIL data processing pipeline",
    tags=["mediatech", "cnil"],
) as dag:
    create_tables = BashOperator(
        task_id="create_tables",
        bash_command="mediatech create_tables --model BAAI/bge-m3",
    )

    download_and_process_files = BashOperator(
        task_id="download_and_process_files",
        bash_command="mediatech download_and_process_files --source cnil --model BAAI/bge-m3",
    )

    export_table = BashOperator(
        task_id="export_table", bash_command="mediatech export_table --table cnil"
    )

    upload_dataset = BashOperator(
        task_id="upload_dataset",
        bash_command="mediatech upload_dataset --dataset-name cnil",
    )

    create_tables >> download_and_process_files >> export_table >> upload_dataset
