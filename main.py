#!/usr/bin/env python3

"""Mediatech CLI.

Usage:
    main.py download_files (--all | --source=<source>) [--debug]
    main.py download_and_process_files (--all | --source=<source>) [--model=<model_name>] [--debug]
    main.py create_tables [--model=<model_name>] [--delete-existing] [--debug]
    main.py process_files (--all | --source=<source>) [--folder=<path>] [--model=<model_name>] [--debug]
    main.py split_table (--source=<source>) [--debug]
    main.py export_table (--table=<name> | --all) [--output=<path>] [--debug]
    main.py upload_dataset (--all | --dataset-name=<name>) [--input=<path>] [--repository=<name>] [--private] [--debug]
    main.py -h | --help

Commands:
    download_files              Download files from sources
    download_and_process_files  Download and process files from sources
    create_tables               Create database tables (with option to delete existing ones)
    process_files               Process data from specific source or all sources and insert into database
    split_table                 Split a table into multiple smaller tables based on source and criteria
    export_table                Export table to Parquet files
    upload_dataset              Upload dataset to Hugging Face

Options:
    --config-file=<path>    Path to the config file
    --history-file=<path>   Path to the data history file
    --delete-existing       Delete existing tables before creating new ones
    --all                   Select all data sources from the data configuration file
    --model=<model_name>    Embedding model name [default: BAAI/bge-m3]. It is mandatory to specify the same model for all commands.
    --source=<source>       Source to process (service_public, travail_emploi, legi, cnil,
                            state_administrations_directory, local_administrations_directory, constit, dole)
    --folder=<path>         Folder containing unprocessed data
    --input=<path>          Input path of the dataset to upload
    --dataset-name=<name>   Name of the dataset to upload to Hugging Face
    --repository=<name>     Hugging Face repository name [default: AgentPublic]
    --output=<path>         Output folder for Parquet files
    --private               Upload dataset as private on Hugging Face
    --debug                 Enable debug logging
    -h --help               Show this help message

Examples:
    main.py create_tables --model BAAI/bge-m3 --delete-existing
    main.py download_files --all
    main.py download_and_process_files --source service_public --model BAAI/bge-m3 --debug
    main.py download_and_process_files --all --model BAAI/bge-m3
    main.py process_files --source service_public --model BAAI/bge-m3
    main.py process_files --all --folder data/unprocessed --model BAAI/bge-m3
    main.py split_table --source legi
    main.py export_table --table legi
    main.py export_table --all --output data/parquet
    main.py upload_dataset --input data/parquet/service_public.parquet --dataset-name service-public --repository AgentPublic --private
    main.py upload_dataset --all --repository AgentPublic
"""

from docopt import docopt
import sys
import os
from config import (
    setup_logging,
    get_logger,
    CNIL_DATA_FOLDER,
    LEGI_DATA_FOLDER,
    CONSTIT_DATA_FOLDER,
    DATA_GOUV_DATASETS_CATALOG_DATA_FOLDER,
    LOCAL_ADMINISTRATIONS_DIRECTORY_FOLDER,
    STATE_ADMINISTRATIONS_DIRECTORY_FOLDER,
    TRAVAIL_EMPLOI_DATA_FOLDER,
    SERVICE_PUBLIC_PRO_DATA_FOLDER,
    SERVICE_PUBLIC_PART_DATA_FOLDER,
    HF_TOKEN,
    DOLE_DATA_FOLDER,
    BASE_PATH,
    parquet_files_folder,
    config_file_path,
    data_history_path,
)
from database import create_all_tables, split_legi_table
from download_and_processing import (
    process_data,
    process_all_data,
    download_and_optionally_process_files,
    download_and_optionally_process_all_files,
)
from utils import export_table_to_parquet


def main():
    try:
        args = docopt(__doc__)

        # Setup logging
        debug_mode = args.get("--debug", False)
        setup_logging(debug=debug_mode)
        logger = get_logger(__name__)

        # Download files
        if args["download_files"]:
            if args["--all"]:
                logger.info(
                    f"Downloading all files using config: {config_file_path} and history: {data_history_path}"
                )
                download_and_optionally_process_all_files(
                    config_file_path=config_file_path,
                    data_history_path=data_history_path,
                    process=False,
                    model=args["--model"] if args["--model"] else "BAAI/bge-m3",
                )
            else:
                source = args["--source"]
                source_map = {
                    "service_public": [
                        "service_public_part",
                        "service_public_pro",
                    ],
                    "travail_emploi": ["travail_emploi"],
                    "legi": ["legi"],
                    "cnil": ["cnil"],
                    "state_administrations_directory": [
                        "state_administrations_directory"
                    ],
                    "local_administrations_directory": [
                        "local_administrations_directory"
                    ],
                    "constit": ["constit"],
                    "dole": ["dole"],
                    "data_gouv_datasets_catalog": ["data_gouv_datasets_catalog"],
                }

                if source not in source_map:
                    logger.error(f"Unknown source: {source}")
                    return 1
                else:
                    logger.info(
                        f"Downloading and processing {source} files using config: {config_file_path} and history: {data_history_path}"
                    )

                    for data_name in source_map[source]:
                        download_and_optionally_process_files(
                            data_name=data_name,
                            config_file_path=config_file_path,
                            data_history_path=data_history_path,
                            process=False,
                            model=args["--model"] if args["--model"] else "BAAI/bge-m3",
                        )

        # Download and process files
        # This method as a better storage optimization compared to download_files + process_files)
        elif args["download_and_process_files"]:
            if args["--all"]:
                logger.info(
                    f"Downloading and processing all files using config: {config_file_path} and history: {data_history_path}"
                )
                download_and_optionally_process_all_files(
                    config_file_path=config_file_path,
                    data_history_path=data_history_path,
                    process=True,
                    model=args["--model"] if args["--model"] else "BAAI/bge-m3",
                )
            else:
                source = args["--source"]
                source_map = {
                    "service_public": [
                        "service_public_part",
                        "service_public_pro",
                    ],
                    "travail_emploi": ["travail_emploi"],
                    "legi": ["legi"],
                    "cnil": ["cnil"],
                    "state_administrations_directory": [
                        "state_administrations_directory"
                    ],
                    "local_administrations_directory": [
                        "local_administrations_directory"
                    ],
                    "constit": ["constit"],
                    "dole": ["dole"],
                    "data_gouv_datasets_catalog": ["data_gouv_datasets_catalog"],
                }

                if source not in source_map:
                    logger.error(f"Unknown source: {source}")
                    return 1
                else:
                    for data_name in source_map[source]:
                        logger.info(
                            f"Downloading and processing {data_name} files using config: {config_file_path} and history: {data_history_path}"
                        )
                        download_and_optionally_process_files(
                            data_name=data_name,
                            config_file_path=config_file_path,
                            data_history_path=data_history_path,
                            process=True,
                            model=args["--model"] if args["--model"] else "BAAI/bge-m3",
                        )

        # Create tables
        elif args["create_tables"]:
            delete_existing = True if args["--delete-existing"] else False
            model = args["--model"] if args["--model"] else "BAAI/bge-m3"
            logger.info(
                f"Creating tables with model {model} (delete_existing={delete_existing})"
            )
            create_all_tables(delete_existing=delete_existing, model=model)

        # Process data
        elif args["process_files"]:
            model = args["--model"] if args["--model"] else "BAAI/bge-m3"
            if args["--all"]:
                folder = args["--folder"] or os.path.join(BASE_PATH, "data/unprocessed")
                logger.info(f"Processing all unprocessed data from folder: {folder}")
                process_all_data(unprocessed_data_folder=folder, model=model)
            else:
                source = args["--source"]
                source_map = {
                    "service_public": [
                        SERVICE_PUBLIC_PRO_DATA_FOLDER,
                        SERVICE_PUBLIC_PART_DATA_FOLDER,
                    ],
                    "travail_emploi": [TRAVAIL_EMPLOI_DATA_FOLDER],
                    "legi": [LEGI_DATA_FOLDER],
                    "cnil": [CNIL_DATA_FOLDER],
                    "state_administrations_directory": [
                        STATE_ADMINISTRATIONS_DIRECTORY_FOLDER
                    ],
                    "local_administrations_directory": [
                        LOCAL_ADMINISTRATIONS_DIRECTORY_FOLDER
                    ],
                    "constit": [CONSTIT_DATA_FOLDER],
                    "dole": [DOLE_DATA_FOLDER],
                    "data_gouv_datasets_catalog": [
                        DATA_GOUV_DATASETS_CATALOG_DATA_FOLDER
                    ],
                }

                if source not in source_map:
                    logger.error(f"Unknown source: {source}")
                    return 1
                else:
                    logger.info(f"Processing data from source: {source}")
                    for data_folder in source_map[source]:
                        process_data(base_folder=data_folder, model=model)

        # Split table into smaller tables based on several criteria
        elif args["split_table"]:
            source = args["--source"] if args["--source"] else "unknown"
            if source == "legi":
                logger.info("Splitting LEGI table into smaller tables")
                split_legi_table()
            else:
                logger.error(f"Splitting is not implemented for the {source} source.")
                return 1

        # Export tables to parquet
        elif args["export_table"]:
            output = args["--output"] or parquet_files_folder
            if args["--all"]:
                logger.info(
                    f"Exporting all PgVector tables to Parquet in folder: {output}"
                )
                export_table_to_parquet(table_name="all", output_folder=output)
            else:
                logger.info(
                    f"Exporting {args['--table']} table to Parquet in folder: {output}"
                )
                export_table_to_parquet(
                    table_name=args["--table"], output_folder=output
                )

        # Upload dataset to Hugging Face
        elif args["upload_dataset"]:
            from utils import HuggingFace

            if args["--all"]:
                logger.info("Uploading all datasets to Hugging Face")
                private = True if args["--private"] else False
                repository = (
                    args["--repository"] if args["--repository"] else "AgentPublic"
                )
                hf = HuggingFace(hugging_face_repo=repository, token=HF_TOKEN)
                hf.upload_all_datasets(
                    config_file_path=config_file_path, private=private
                )
            else:
                dataset_name = args[
                    "--dataset-name"
                ]  # The name of the dataset to upload (e.g., service-public, travail-emploi, etc.)
                input_path = (
                    args["--input"]
                    if args["--input"]
                    else os.path.join(
                        parquet_files_folder,
                        f"{dataset_name.lower().replace('-', '_')}.parquet",
                    )  # Default path for the dataset (e.g., ./data/parquet/service_public.parquet)
                )
                repository = (
                    args["--repository"] if args["--repository"] else "AgentPublic"
                )
                private = True if args["--private"] else False

                logger.info(
                    f"Uploading dataset {dataset_name} from {input_path} to Hugging Face (private={private})"
                )
                hf = HuggingFace(hugging_face_repo=repository, token=HF_TOKEN)
                hf.upload_dataset(
                    dataset_name=dataset_name, file_path=input_path, private=private
                )

        return 0

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
