# MEDIATECH

[![License](https://img.shields.io/github/license/etalab-ia/mediatech?label=licence&color=red)](https://github.com/etalab-ia/mediatech/blob/main/LICENSE)
[![French version](https://img.shields.io/badge/🇫🇷-French%20version-blue)](./docs/README_fr.md)
[![Hugging Face collection](https://img.shields.io/badge/🤗-Hugging%20Face%20collection-yellow)](https://huggingface.co/collections/AgentPublic/mediatech-68309e15729011f49ef505e8)


## 📝 Description

This project processes public data made available by various administrations in order to facilitate access to vectorized and ready-to-use public data for AI applications in the public sector.
It includes scripts for downloading, processing, embedding, and inserting this data into a PostgreSQL database, and facilitates its export via various means.

## 💡 Tutorial

### 𖣘 Method 1 : Airflow

#### Installing and configuring dependencies

1. Run the initial deployment script:
   ```bash
   sudo chmod +x ./scripts/initial_deployment.sh
   ./scripts/initial_deployment.sh
   ```
  
2. Set up the environment variables in a [`.env`](.env) file based on the example in [`.env.example`](.env.example).
   > The `AIRFLOW_UID` variable must be obtained by executing:
    ```bash
    echo $(id -u)
    ```
   > The `JWT_TOKEN` variable will be obtained later by using the Airflow API. Just leave it for now.

#### Initialize Airflow and PostgreSQL (PgVector) containers

1. Run the [`containers_deployment`](./scripts/containers_deployment) script :
   ```bash
   sudo chmod +x ./scripts/containers_deployment.sh
   ./scripts/containers_deployment.sh
   ```

2. Export [`.env`](.env) variables :
   ```bash
   export $(grep -v '^#' .env | xargs)
   ```

3. Make sure to remove the PostgreSQL (PgVector) volume:
   ```bash
   docker compose down -v
   ```
   > ⚠️ This operation will delete all volumes ! 

4. Use the Airflow API to obtain the `JWT_TOKEN` variable:
   ```bash
   curl -X 'POST' \
   'http://localhost:8080/auth/token' \
   -H 'Content-Type: application/json' \
   -d "{\"username\": \"${_AIRFLOW_WWW_USER_USERNAME}\", \"password\": \"${_AIRFLOW_WWW_USER_PASSWORD}\"}"
   ```

5. Define the `JWT_TOKEN` variable in the [`.env`](.env) file with the obtained `access_token`.

#### Downloading, Processing and Uploading Data

You are now ready to use Airflow and execute DAGs that are available.
Each dataset has its own DAG and a [`DAG_Controller`](./airflow/dags/dag_controller.py) is defined to manage all datasets DAGs and their execution order.

### </> Method 2 : Use local CLI

#### Installing Dependencies

1. Install the required apt dependencies:
   ```bash
   sudo apt-get update
   sudo apt-get install -y $(cat config/requirements-apt-container.txt)
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv  # Create the virtual environment
   source .venv/bin/activate  # Activate the virtual environment
   ```

3. Install the required python dependencies:
   ```bash
   pip install -e .
   ```

> Installing in development mode (`-e`) allows you to use the `mediatech` command and modify the code without reinstalling.

> **Note:** Make sure your environment is properly configured before continuing.

#### PostgreSQL (PgVector) Database Configuration

1. Set up the environment variables in a [`.env`](.env) file based on the example in [`.env.example`](.env.example).

2. Start the PostgreSQL container with Docker:
   ```bash
   docker compose up -d postgres
   ```

3. Check that the `pgvector_container` container is running:
   ```bash
   docker ps
   ```

#### Downloading, Processing and Uploading Data

##### Using the `mediatech` Command

After installation, the `mediatech` command is available globally and replaces `python main.py`:

> If you encounter issues with the `mediatech` command, you can still use `python main.py` instead.

The [`main.py`](main.py) file is the main entry point of the project and provides a command-line interface (CLI) to run each step of the pipeline separately.  
You can use it as follows:

```bash
mediatech <command> [options]
```
or 

```bash
python main.py <command> [options]
```

Command examples:
- View help:
  ```bash
  mediatech --help
  ```
- Create PostgreSQL tables:  
  ```bash
  mediatech create_tables --model BAAI/bge-m3
  ```
- Download all files listed in [`data_config.json`](config/data_config.json):  
  ```bash
  mediatech download_files --all
  ```
- Download files from the `service_public` source:  
  ```bash
  mediatech download_files --source service_public
  ```
- Download and process all files listed in [`data_config.json`](config/data_config.json):  
  ```bash
  mediatech download_and_process_files --all --model BAAI/bge-m3
  ```
- Process all data:  
  ```bash
  mediatech process_files --all --model BAAI/bge-m3
  ```
- Split a table into subtables based on different criteria (see [`main.py`](main.py)):  
  ```bash
  mediatech split_table --source legi
  ```
- Export PostgreSQL tables to parquet files:  
  ```bash
  mediatech export_tables --output data/parquet
  ```
- Upload parquet datasets to the Hugging Face repository:
  ```bash
  mediatech upload_dataset --input data/parquet/service_public.parquet --dataset-name service-public
  ```


Run `mediatech --help` in your terminal to see all available options, or check the code directly in [`main.py`](main.py).


##### Alternative Usage with `python main.py`

If you prefer to use the Python script directly, you can always use:

```bash
python main.py <command> [options]
```

Examples:
```bash
python main.py download_files
python main.py create_tables --model BAAI/bge-m3
python main.py process_files --all --model BAAI/bge-m3
```
##### Using the [`update.sh`](update.sh) Script

The [`update.sh`](update.sh) script allows you to run the entire data processing pipeline: downloading, table creation, vectorization, and export.  
To run it, execute the following command from the project root:

```bash
./scripts/update.sh
```

This script will:
- Wait for the PostgreSQL database to be available,
- Create or update the necessary tables in the PostgreSQL database,
- Download public files listed in [`data_config.json`](config/data_config.json),
- Process and vectorize the data,
- Export the tables in Parquet format,
- Upload the Parquet files to [Hugging Face](https://huggingface.co/AgentPublic).

## 🗂️ Project Structure

- **[`main.py`](main.py)**: Main entry point to run the complete pipeline via CLI.
- **[`pyproject.toml`](pyproject.toml)**: Python project and dependency configuration.
- **[`Dockerfile`](Dockerfile)**: Defines the instructions to build the custom Docker image for Airflow, installing system dependencies, Python packages, and setting up the project environment.
- **[`docker-compose.yml`](docker-compose.yml)**: Orchestrates the multi-container setup, defining Airflow services and the PostgreSQL (PgVector) database.
- **[`.github/`](.github/)**: Contains GitHub Actions workflows for Continuous Integration and Continuous Deployment (CI/CD), automating testing and deployment processes.
- **[`download_and_processing/`](download_and_processing/)**: Contains scripts to download and extract files.
- **[`database/`](database/)**: Contains scripts to manage the database (table creation, data insertion).
- **[`utils/`](utils/)**: Contains utility functions shared across modules.
- **[`config/`](config/)**: Contains project configuration scripts.
- **[`logs/`](logs/)**: Contains log files to track [scripts](scripts/) execution.
- **[`scripts/`](scripts/)**: Contains all shell scripts, executed either automatically or manually in some cases.
  - **[`scripts/update.sh`](scripts/update.sh)**: Shell script to run the entire data processing pipeline.
  - **[`scripts/periodic_update.sh`](scripts/periodic_update.sh)**: Shell script to automate the pipeline on the virtual machine. This script is executed periodically by [`cron_config.txt`](cron_config.txt).
  - **[`scripts/backup.sh`](scripts/backup.sh)**: Shell script to back up the Pgvector (PostgreSQL) volume and some configuration files. This script is executed periodically by [`cron_config.txt`](cron_config.txt).
  - **[`scripts/restore.sh`](scripts/restore.sh)**: Shell script to restore the Pgvector (PostgreSQL) volume and configuration files if needed.
  - **[`scripts/initial_deployment.sh`](scripts/initial_deployment.sh)**: Sets up a new server environment by installing Docker, Docker Compose, and other system dependencies.
  - **[`scripts/containers_deployment.sh`](scripts/containers_deployment.sh)**:  Manages the application's lifecycle by building, initializing, and deploying the Docker containers as defined in [docker-compose.yml](docker-compose.yml). It must be executed after each update of the Mediatech CLI or other script not shared with the Airflow container, as defined in [docker-compose.yml](docker-compose.yml).
  - **[`scripts/check_running_dags.sh`](scripts/check_running_dags.sh)**: Checks the Airflow API to see if any data pipelines (DAGs) are currently running, used to safely lock the deployment process.
  - **[`scripts/delete_old_logs.sh`](scripts/delete_old_logs.sh)**: Shell script to automatically delete old log files from the [`logs/`](logs/) folder. It keeps logs from the last X days and deletes older ones. This script can be run manually or scheduled via cron to keep the logs folder clean.
- **[`airflow`](airflow/)**: Contains all files related to Apache Airflow, including DAG definitions (`dags/`), configuration (`config/`), logs (`logs/`), and plugins (`plugins/`). This is where the data orchestration pipelines are defined and managed.

## ⚖️ License

This project is licensed under the [MIT License](./LICENSE).