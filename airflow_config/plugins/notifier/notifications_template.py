from airflow.providers.apprise.notifications.apprise import AppriseNotifier


# Notifications template
def get_start_notifier():
    return AppriseNotifier(
        title="🚀𖣘 Airflow - Starting task...",
        body="""- **DAG** : {{ dag.dag_id }}\n
- **Task** : {{ task.task_id }}\n
- **Description** : {{ dag.description | default("No description") }}\n
- **Date** : {{ ds }}\n
- **Run ID** : {{ run_id }}\n
""",
        body_format="markdown",
        notify_type="info",
        apprise_conn_id="TchapNotifier",
        interpret_escapes=True,
        tag="alerts",
    )


def get_success_notifier():
    return AppriseNotifier(
        title="✅𖣘 Airflow - Task success",
        body="""- **DAG** : {{ dag.dag_id }}\n
- **Task** : {{ task.task_id }}\n
""",
        body_format="markdown",
        notify_type="success",
        apprise_conn_id="TchapNotifier",
        interpret_escapes=True,
        tag="alerts",
    )


def get_failure_notifier():
    return AppriseNotifier(
        title="❌𖣘 Airflow - Failure",
        body="""- **DAG** : {{ dag.dag_id }}\n
- **Task** : {{ task.task_id }}\n
### 🚨 **ERROR :**\n
```
{{ exception | default('Error not specified') }}
```
""",
        body_format="markdown",
        notify_type="failure",
        apprise_conn_id="TchapNotifier",
        interpret_escapes=True,
        tag="alerts",
    )
