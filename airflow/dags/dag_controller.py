from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 8, 1),
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "DAG_CONTROLLER",
    schedule= "0 20 * * 5",  # Every Friday at 20:00
    start_date=datetime(2025, 8, 20),
    catchup=False,
    max_active_runs=1,
    description="DAG Controller for Mediatech Pipelines",
    tags=["mediatech", "dag_controller"],
) as dag:
    # Fixing the same execution date for all triggered DAGs
    shared_execution_date = "{{ ts }}"

    trigger_cnil = TriggerDagRunOperator(
        task_id="trigger_cnil",
        trigger_dag_id="CNIL",
        logical_date=shared_execution_date,
    )

    trigger_constit = TriggerDagRunOperator(
        task_id="trigger_constit",
        trigger_dag_id="CONSTIT",
        logical_date=shared_execution_date,
    )

    trigger_dole = TriggerDagRunOperator(
        task_id="trigger_dole",
        trigger_dag_id="DOLE",
        logical_date=shared_execution_date,
    )

    trigger_legi = TriggerDagRunOperator(
        task_id="trigger_legi",
        trigger_dag_id="LEGI",
        logical_date=shared_execution_date,
    )

    trigger_state_administrations_directory = TriggerDagRunOperator(
        task_id="trigger_state_administrations_directory",
        trigger_dag_id="STATE_ADMINISTRATIONS_DIRECTORY",
        logical_date=shared_execution_date,
    )

    trigger_local_administrations_directory = TriggerDagRunOperator(
        task_id="trigger_local_administrations_directory",
        trigger_dag_id="LOCAL_ADMINISTRATIONS_DIRECTORY",
        logical_date=shared_execution_date,
    )

    trigger_service_public = TriggerDagRunOperator(
        task_id="trigger_service_public",
        trigger_dag_id="SERVICE_PUBLIC",
        logical_date=shared_execution_date,
    )

    trigger_travail_emploi = TriggerDagRunOperator(
        task_id="trigger_travail_emploi",
        trigger_dag_id="TRAVAIL_EMPLOI",
        logical_date=shared_execution_date,
    )

    trigger_data_gouv_datasets_catalog = TriggerDagRunOperator(
        task_id="trigger_data_gouv_datasets_catalog",
        trigger_dag_id="DATA_GOUV_DATASETS_CATALOG",
        logical_date=shared_execution_date,
    )

    (
        trigger_cnil
        >> trigger_constit
        >> trigger_dole
        >> trigger_legi
        >> trigger_state_administrations_directory
        >> trigger_local_administrations_directory
        >> trigger_service_public
        >> trigger_travail_emploi
        >> trigger_data_gouv_datasets_catalog
    )
