from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.external_task import ExternalTaskSensor

default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 8, 1),
    "retries": 3,
    "retry_delay": timedelta(minutes=10),
}

# Schedule based on Airflow's Variable, can be None for manual runs
PIPELINE_SCHEDULE = Variable.get("full_pipeline_schedule", default_var=None)


def create_trigger(dag_id: str, execution_date: str) -> TriggerDagRunOperator:
    """Create a TriggerDagRunOperator for a given DAG."""
    return TriggerDagRunOperator(
        task_id=f"trigger_{dag_id.lower()}",
        trigger_dag_id=dag_id,
        logical_date=execution_date,
        wait_for_completion=False,
        reset_dag_run=True,
    )


def create_wait_sensor(dag_id: str) -> ExternalTaskSensor:
    """Create an ExternalTaskSensor to wait for a DAG to complete."""
    return ExternalTaskSensor(
        task_id=f"wait_for_{dag_id.lower()}",
        external_dag_id=dag_id,
        external_task_id="upload_dataset",  # Last task of each DAG
        mode="reschedule",  # Reschedule mode to avoid blocking the scheduler
        timeout=21 * 24 * 60 * 60,  # Wait up to 21 days, after which the task will fail
        poke_interval=120,  # Check every 2 minutes if the task has completed
        allowed_states=["success"],
        failed_states=["failed", "skipped", "upstream_failed"],
    )


with DAG(
    "FULL_PIPELINE",
    default_args=default_args,
    schedule=PIPELINE_SCHEDULE,
    catchup=False,
    max_active_runs=1,
    description="MediaTech full data processing pipeline",
    tags=["mediatech", "full_pipeline"],
) as dag:
    shared_execution_date = "{{ ts }}"

    # DAGs execution order
    dag_order = [
        "CNIL",
        "CONSTIT",
        "DOLE",
        "STATE_ADMINISTRATIONS_DIRECTORY",
        "LOCAL_ADMINISTRATIONS_DIRECTORY",
        "SERVICE_PUBLIC",
        "TRAVAIL_EMPLOI",
        "DATA_GOUV_DATASETS_CATALOG",
        "LEGI",
    ]

    # Create triggers and wait sensors
    tasks = []
    for dag_id in dag_order:
        trigger = create_trigger(dag_id=dag_id, execution_date=shared_execution_date)
        wait = create_wait_sensor(dag_id=dag_id)
        tasks.extend([trigger, wait])

    # Chain sequentially: trigger_X >> wait_X >> trigger_Y >> wait_Y >> ...
    for i in range(len(tasks) - 1):
        tasks[i] >> tasks[i + 1]
