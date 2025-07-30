#!/bin/bash

# Load environment variables from .env file
export $(grep -v '^#' .env | xargs)

# Wait for PostgreSQL to be ready
until pg_isready -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER; do
  echo "Waiting for PostgreSQL..."
  sleep 2
done

# Run the Python scripts
mediatech create_tables --model BAAI/bge-m3
mediatech download_and_process_files --all --model BAAI/bge-m3
mediatech export_tables
mediatech upload_dataset --all 
