# ALBERT Bibliothèque

## Description

This project processes public data made available by various administrations in order to facilitate access to vectorized and ready-to-use public data for AI applications in the public sector.
It includes scripts for downloading, processing, embedding, and inserting this data into a PostgreSQL database, and facilitates its export via various means.

## Instructions

### Installing Dependencies

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv  # Create the virtual environment
   source .venv/bin/activate  # Activate the virtual environment
   ```

2. Install the required dependencies:
   ```bash
   pip install -e .
   ```

> Installing in development mode (`-e`) allows you to use the `bibliotheque` command and modify the code without reinstalling.

> **Note:** Make sure your environment is properly configured before continuing.

### PostgreSQL Database Configuration

1. Start the PostgreSQL container with Docker:
   ```bash
   docker compose up -d
   ```

2. Check that the container is running:
   ```bash
   docker ps
   ```

3. Set up the environment variables in a [`.env`](.env) file based on the example in [`.env.example`](.env.example).

### Downloading and Processing Data

#### Using the `bibliotheque` Command

After installation, the `bibliotheque` command is available globally and replaces `python main.py`:

> If you encounter issues with the `bibliotheque` command, you can still use `python main.py` instead.

The [`main.py`](main.py) file is the main entry point of the project and provides a command-line interface (CLI) to run each step of the pipeline separately.  
You can use it as follows:

```bash
bibliotheque <command> [options]
```
or 

```bash
python main.py <command> [options]
```

Command examples:
- View help:
  ```bash
  bibliotheque --help
  ```
- Create PostgreSQL tables:  
  ```bash
  bibliotheque create_tables --model BAAI/bge-m3
  ```
- Download all files listed in [`data_config.json`](config/data_config.json):  
  ```bash
  bibliotheque download_files --all
  ```
- Download files from the `service_public` source:  
  ```bash
  bibliotheque download_files --source service_public
  ```
- Download and process all files listed in [`data_config.json`](config/data_config.json):  
  ```bash
  bibliotheque download_and_process_files --all --model BAAI/bge-m3
  ```
- Process all data:  
  ```bash
  bibliotheque process_files --all --model BAAI/bge-m3
  ```
- Split a table into subtables based on different criteria (see [`main.py`](main.py)):  
  ```bash
  bibliotheque split_table --source legi
  ```
- Export PostgreSQL tables to parquet files:  
  ```bash
  bibliotheque export_tables --output data/parquet
  ```
- Upload parquet datasets to the Hugging Face repository:
  ```bash
  bibliotheque upload_dataset --input data/parquet/service_public.parquet --dataset-name service-public
  ```


Run `bibliotheque --help` in your terminal to see all available options, or check the code directly in [`main.py`](main.py).


#### Alternative Usage with `python main.py`

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
#### Using the [`update.sh`](update.sh) Script

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

### Project Structure

- **[`main.py`](main.py)**: Main entry point to run the complete pipeline via CLI.
- **[`pyproject.toml`](pyproject.toml)**: Python project and dependency configuration.
- **[`download_and_processing/`](download_and_processing/)**: Contains scripts to download and extract files.
- **[`database/`](database/)**: Contains scripts to manage the database (table creation, data insertion).
- **[`utils/`](utils/)**: Contains utility functions shared across modules.
- **[`config/`](config/)**: Contains project configuration scripts.
- **[`logs/`](logs/)**: Contains log files to track script execution.
- **[`scripts/`](scripts/)**: Contains all shell scripts, executed either periodically or manually in some cases.
  - **[`scripts/update.sh`](scripts/update.sh)**: Shell script to run the entire data processing pipeline.
  - **[`scripts/periodic_update.sh`](scripts/periodic_update.sh)**: Shell script to automate the pipeline on the virtual machine. This script is executed periodically by [`cron_config.txt`](cron_config.txt).
  - **[`scripts/backup.sh`](scripts/backup.sh)**: Shell script to back up the Pgvector (PostgreSQL) volume and some configuration files. This script is executed periodically by [`cron_config.txt`](cron_config.txt).
  - **[`scripts/restore.sh`](scripts/restore.sh)**: Shell script to restore the Pgvector (PostgreSQL) volume and configuration files if needed.
  - **[`scripts/deployment_packages.sh`](scripts/deployment_packages.sh)**: Shell script to automatically install the required system packages (via apt) and configure Docker permissions. It reads the list of packages to install from [`config/requirements-apt.txt`](config/requirements-apt.txt), installs missing ones, and runs admin commands if needed. Run after cloning the project or when updating the system environment.
  - **[`scripts/delete_old_logs.sh`](scripts/delete_old_logs.sh)**: Shell script to automatically delete old log files from the [`logs/`](logs/) folder. It keeps logs from the last X days and deletes older ones. This script can be run manually or scheduled via cron to keep the logs folder clean.
