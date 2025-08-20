#!/bin/bash

# Load environment variables from .env file
export $(grep -v '^#' .env | xargs)

JWT_TOKEN=$JWT_TOKEN
AIRFLOW_API="http://127.0.0.1:8080/api/v2"

# Check if at least a DAG is running (1) or not (0)
RUNNING_FOUND=0

# Listing all dags
DAGS=$(curl -s -X GET "$AIRFLOW_API/dags" -H "Authorization: Bearer $JWT_TOKEN" | jq -r '.dags[].dag_id')

if [ -n "$DAGS" ]; then
    for dag_id in $DAGS; do
        RUNNING=$(curl -s -X GET "$AIRFLOW_API/dags/$dag_id/dagRuns" \
            -H "Authorization: Bearer $JWT_TOKEN" \
            -G \
            --data-urlencode "state=running" | jq -r '.total_entries')

        if [ "$RUNNING" -gt 0 ]; then
            echo "DAG $dag_id is RUNNING ($RUNNING runs)"
            RUNNING_FOUND=1
        fi
    done
fi

if [ "$RUNNING_FOUND" -eq 0 ]; then
  echo "No DAGs are currently running. Deploying..."
  exit 0
else
  echo "Some DAGs are still running. Deployment aborted."
  exit 1
fi
