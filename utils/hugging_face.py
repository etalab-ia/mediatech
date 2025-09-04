import json
import os

from . import load_data_history, file_sha256, remove_folder, remove_file
from config import (
    get_logger,
    data_history_path,
    config_file_path,
    BASE_PATH,
    SERVICE_PUBLIC_PRO_DATA_FOLDER,
    SERVICE_PUBLIC_PART_DATA_FOLDER,
)
import datetime as dt
from huggingface_hub import (
    HfApi,
    HfFolder,
    dataset_info,
    CommitOperationCopy,
    CommitOperationDelete,
)
from huggingface_hub.utils import HfHubHTTPError


logger = get_logger(__name__)


class HuggingFace:
    def __init__(self, hugging_face_repo, token=None):
        """
        Initialize the HuggingFace class.

        Args:
            hugging_face_repo (str): The Hugging Face repository name. (e.g. : "AgentPublic")
            token (str, optional): Hugging Face API token. If not provided, it will be retrieved from HfFolder.
        """
        self.hugging_face_repo = hugging_face_repo
        self.token = token if token else HfFolder.get_token()
        self.api = HfApi()

    def is_dataset_up_to_date(self, dataset_name: str, local_folder_path: str) -> bool:
        """
        Checks if the remote files on Hugging Face is identical to the local files
        by comparing their SHA256 hashes without downloading the remote files.

        Args:
            dataset_name: Name of the HF dataset to check.
            local_folder_path: Local dataset folder path to compare.

        Returns:
            bool: True if local file hash matches the remote file hash, False otherwise.
        """
        repo_id = f"{self.hugging_face_repo}/{dataset_name}"
        path_in_repo = f"data/{dataset_name}-latest/"

        try:
            # Get the repository metadata from the Hub API
            info = self.api.dataset_info(
                repo_id=repo_id, files_metadata=True, token=self.token
            )

            # List all local parquet files in the specified folder
            local_files = [
                f for f in os.listdir(local_folder_path) if f.endswith(".parquet")
            ]
            local_files.sort()  # Sort to ensure consistent order

            if not local_files:
                logger.error(
                    f"No local files found in folder '{local_folder_path}' for dataset '{dataset_name}'."
                )
                return False

            # List all remote parquet files in the specified folder
            remote_files = [
                f.rfilename
                for f in info.siblings
                if f.rfilename.startswith(path_in_repo)
                and f.rfilename.endswith(".parquet")
            ]
            remote_files.sort()  # Sort to ensure consistent order

            if not remote_files:
                logger.warning(
                    f"No remote files found in repo '{repo_id}' for dataset '{dataset_name}'. Assuming not up to date."
                )
                return False

            if len(local_files) != len(remote_files):
                logger.warning(
                    f"Local files count ({len(local_files)}) does not match remote files count ({len(remote_files)}) for dataset '{dataset_name}'. Assuming not up to date."
                )
                return False

            for k, local_file in enumerate(local_files):
                remote_file = remote_files[k] if k < len(remote_files) else None
                remote_file_info = next(
                    (f for f in info.siblings if f.rfilename == remote_file), None
                )

                if not remote_file.endswith(local_file):
                    logger.warning(
                        f"Local file '{local_file}' does not match remote file '{remote_file}'. Assuming not up to date."
                    )
                    return False

                # Get the SHA256 hash from the LFS metadata
                remote_hash = (
                    remote_file_info.lfs.get("sha256") if remote_file_info.lfs else None
                )
                if not remote_hash:
                    logger.error(
                        f"Could not retrieve LFS hash for remote file '{path_in_repo}'."
                    )
                    return False

                # Calculate the SHA256 hash of the local file
                local_hash = file_sha256(os.path.join(local_folder_path, local_file))
                logger.debug(
                    f"Comparing local SHA256 ({local_hash}) with remote SHA256 ({remote_hash})"
                )
                if local_hash != remote_hash:
                    logger.warning(
                        f"Local file '{local_file}' is not up to date with remote file '{remote_file}'."
                    )
                    return False

            logger.info(
                f"No anomalies found for dataset '{dataset_name}' and for local files. Assuming up to date."
            )
            return True

        except HfHubHTTPError as e:
            # If the repo or file does not exist (404), it's not up to date.
            if e.response.status_code == 404:
                logger.info(
                    f"Dataset '{repo_id}' or file does not exist. Assuming not up to date."
                )
            else:
                logger.error(f"HTTP Error checking dataset status for '{repo_id}': {e}")
            return False
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while checking dataset '{repo_id}': {e}"
            )
            return False

    def get_file_upload_date(self, dataset_name: str, hf_file_path: str) -> str:
        """
        Get the upload date of a file from Hugging Face repository.

        Tries three methods in order:
        1. Extract from HF LFS metadata
        2. Get from data history file
        3. Find from repository commit history

        Args:
            dataset_name: Name of the dataset
            hf_file_path: Path to the file in HF repository

        Returns:
            Upload date in YYYYMMDD format, or "01012999" if not found
        """
        repo_id = f"{self.hugging_face_repo}/{dataset_name}"
        info = dataset_info(repo_id, token=self.token)
        for sibling in info.siblings:
            if sibling.rfilename == hf_file_path:
                # If the file is found in HF, try to get the upload date from LFS metadatas
                try:
                    upload_date = sibling.lfs.get("upload_date") or sibling.lfs.get(
                        "last_modified"
                    )
                    logger.info(
                        f"Method 1/3 : Renaming the file based on the upload date from Hugging Face LFS metadata: {upload_date}"
                    )
                    return (
                        upload_date.strftime("%Y%m%d")
                        if hasattr(upload_date, "strftime")
                        else str(upload_date)
                    )
                except AttributeError as e:
                    # If the upload/last modification date is not available, try to get the last upload date from the data history file.

                    logger.info(
                        f"Method 1/3 fail : Unable to retrieve the upload date for {hf_file_path} with the LFS datas from Hugging Face: {e}"
                    )
                    try:
                        # Define a mapping for dataset names to their corresponding file names in the data history file
                        source_map = {
                            "service-public": [
                                "service_public_part",
                                "service_public_pro",
                            ],
                            "travail-emploi": ["travail_emploi"],
                            "legi": ["legi"],
                            "cnil": ["cnil"],
                            "state-administrations-directory": [
                                "state_administrations_directory"
                            ],
                            "local-administrations-directory": [
                                "local_administrations_directory"
                            ],
                            "constit": ["constit"],
                            "dole": ["dole"],
                            "data-gouv-datasets-catalog": [
                                "data_gouv_datasets_catalog"
                            ],
                        }

                        log = load_data_history(data_history_path=data_history_path)
                        file_name = source_map[dataset_name.lower()][
                            0
                        ]  # Get only the first file name from the source map
                        attributes = log.get(file_name, {})
                        last_hf_upload_date = attributes.get("last_hf_upload_date", "")
                        if last_hf_upload_date:
                            logger.info(
                                f"Method 2/3 : Renaming the file based on the last Hugging Face upload date from the data history file : {last_hf_upload_date}"
                            )
                            last_hf_upload_date = dt.datetime.strptime(
                                last_hf_upload_date, "%Y-%m-%d %H:%M:%S"
                            )
                            return last_hf_upload_date.strftime("%Y%m%d")
                        else:
                            raise Exception(
                                f"Last Hugging Face upload date not found in the data history file for the dataset : {dataset_name}"
                            )
                        raise Exception(
                            f"Dataset {dataset_name} not found in the data history file."
                        )
                    except Exception as e:
                        # If the last Hugging Face upload date is not available in the data history file, try to get the last commit date from the Hugging Face repo. BUT MANUAL CHECK WILL BE NEEDED IN HF!
                        logger.info(
                            f"Method 2/3 fail : Unable to retrieve the last Hugging Face upload date from the data history file: {e}"
                        )
                        commits = self.api.list_repo_commits(
                            repo_id=repo_id, repo_type="dataset"
                        )
                        for commit in commits:
                            if dataset_name in commit.title:
                                last_commit_date = (
                                    commit.created_at.strftime("%Y%m%d")
                                    if hasattr(commit.created_at, "strftime")
                                    else str(commit.created_at)
                                )
                                logger.info(
                                    f"Method 3/3 : Renaming the file based on the last file commit date : {last_commit_date}. Manual check is needed in the Hugging Face repository to verify if the date corresponds to the upload date !"
                                )
                                return last_commit_date
                except Exception as e:
                    # If all attempts fail, return a default date and log the error. MANUAL CHECK WILL BE NEEDED IN HF!
                    logger.warning(
                        f"Error : {e}\nRenaming the file based on the default error date : 01012999"
                    )
                    return "01012999"  # Default date if no upload date is found

        logger.error(
            f"File {hf_file_path} not found in the Hugging Face repository: {repo_id}."
        )
        return None  # If file is not found

    def rename_old_latest_folder(self, dataset_name: str):
        """
        Rename the old latest data folder in the Hugging Face dataset repo.

        Args:
            repo_id (str): The Hugging Face dataset repo id (e.g., "user/dataset-name").
            file_name (str): The base name for the file to be renamed.
        """
        repo_id = f"{self.hugging_face_repo}/{dataset_name}"
        representative_file_path = f"data/{dataset_name}-latest/{dataset_name}_part_0.parquet"  # Chosing only the first part to get its upload date
        old_file_date = self.get_file_upload_date(
            dataset_name=dataset_name, hf_file_path=representative_file_path
        )

        old_folder_path = f"data/{dataset_name}-latest"
        new_folder_path = f"data/{dataset_name}-{old_file_date}"

        if old_file_date is None:
            logger.warning(
                f"Failed to retrieve the upload date for {representative_file_path}. Cannot rename the old latest file."
            )
            return

        # List all files in the source directory on HF
        repo_info = self.api.dataset_info(
            repo_id=repo_id, files_metadata=True, token=self.token
        )
        files_to_copy = [
            f
            for f in repo_info.siblings
            if f.rfilename.startswith(old_folder_path + "/")
        ]

        if not files_to_copy:
            logger.warning(
                f"Source folder '{old_folder_path}' is empty or does not exist. Skipping rename."
            )
            return

        # Build the list of operations: copy each file and delete the folder
        operations = []
        for file_info in files_to_copy:
            # Construct the new path for each file inside the new dated folder
            new_file_path = file_info.rfilename.replace(
                old_folder_path, new_folder_path, 1
            )
            operations.append(
                CommitOperationCopy(
                    src_path_in_repo=file_info.rfilename,
                    path_in_repo=new_file_path,
                )
            )

        operations.append(
            CommitOperationDelete(path_in_repo=old_folder_path, is_folder=True)
        )

        # Execute all operations in a single commit
        self.api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            token=self.token,
            commit_message=f"Renamed {old_folder_path} to {new_folder_path}",
        )
        logger.info(
            f"Old folder {old_folder_path} successfuly renamed to {new_folder_path} in repository: {repo_id}."
        )

    def upload_dataset(
        self, dataset_name: str, local_folder_path: str, private: bool = False
    ):
        """
        Upload a parquet file to the Hugging Face dataset repo, creating the repo if needed.

        Args:
            dataset_name (str): The Hugging Face dataset name (e.g. : "service-public").
            local_folder_path (str): Local folder path to the parquet files.
            private (bool): Whether the repo should be private if created.
        """
        # Check if the local folder exists
        if not os.path.exists(local_folder_path):
            logger.error(f"Folder {local_folder_path} does not exist.")
            raise FileNotFoundError(f"Folder {local_folder_path} does not exist.")

        # Get a list of all parquet files in the local folder
        parquet_files = [
            f for f in os.listdir(local_folder_path) if f.endswith(".parquet")
        ]

        if not parquet_files:
            logger.warning(f"No parquet files found in {local_folder_path}.")
            return

        # Check if the dataset is already up to date
        if self.is_dataset_up_to_date(dataset_name, local_folder_path):
            logger.info(
                f"The dataset {dataset_name} is already up to date in the Hugging Face repository. No need to upload it again."
            )
            # Remove the local parquet file folder
            logger.debug(f"Removing local files located in {local_folder_path}.")
            remove_folder(folder_path=local_folder_path)
            return

        # Create the repo if it does not exist
        repo_id = f"{self.hugging_face_repo}/{dataset_name}"
        if not self.api.repo_exists(repo_id, repo_type="dataset"):
            self.api.create_repo(
                repo_id, repo_type="dataset", token=self.token, private=private
            )
            logger.info(f"Hugging Face repository: {repo_id} successfuly created.")

        path_in_repo = f"data/{dataset_name}-latest"

        # Uploading all files in the HF repo
        self.rename_old_latest_folder(dataset_name=dataset_name)

        try:
            for file in parquet_files:
                self.api.upload_file(
                    path_or_fileobj=os.path.join(local_folder_path, file),
                    path_in_repo=f"{path_in_repo}/{file}",
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=self.token,
                    commit_message=f"Updating {dataset_name} dataset file",
                )
                logger.debug(
                    f"File {file} successfuly uploaded to Hugging Face repository: {repo_id}."
                )
                logger.debug(
                    f"Removing local file {os.path.join(local_folder_path, file)}."
                )
                remove_file(file_path=os.path.join(local_folder_path, file))

            # Remove the local parquet files folder after upload
            logger.debug(f"Removing local files located in {local_folder_path}.")
            remove_folder(folder_path=local_folder_path)

        except Exception as e:
            logger.error(
                f"Error while uploading file {file} to the Hugging Face repository {repo_id}: {e}"
            )
            raise e

        # Update the data history file with the last Hugging Face upload date

        # Define a mapping for dataset names to their corresponding file names in the data history file
        source_map = {
            "service-public": ["service_public_part", "service_public_pro"],
            "travail-emploi": ["travail_emploi"],
            "legi": ["legi"],
            "cnil": ["cnil"],
            "state-administrations-directory": ["state_administrations_directory"],
            "local-administrations-directory": ["local_administrations_directory"],
            "constit": ["constit"],
            "dole": ["dole"],
            "data-gouv-datasets-catalog": ["data_gouv_datasets_catalog"],
        }

        try:
            log = load_data_history(data_history_path=data_history_path)
            date = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for file_name in source_map[dataset_name.lower()]:
                if not log.get(file_name):
                    log[file_name] = {}
                log[file_name]["last_hf_upload_date"] = date

            with open(data_history_path, "w") as file:
                json.dump(log, file, indent=4)
            logger.info(
                f"Log data history file successfully updated to {data_history_path}"
            )
        except Exception as e:
            logger.error(f"Error while updating log data history file: {e}")

    def upload_all_datasets(
        self, config_file_path: str = config_file_path, private: bool = False
    ):
        """
        Upload all datasets defined in the config file to Hugging Face.

        Args:
            config_file_path (str): Path to the configuration file containing dataset names and paths.
        """
        try:
            with open(config_file_path, "r") as file:
                config = json.load(file)

            for dataset_name, attributes in config.items():
                if os.path.join(BASE_PATH, attributes.get("download_folder")) in [
                    SERVICE_PUBLIC_PRO_DATA_FOLDER,
                    SERVICE_PUBLIC_PART_DATA_FOLDER,
                ]:
                    dataset_name = "service_public"
                local_folder_path = f"{BASE_PATH}/data/parquet/{dataset_name.lower()}"
                if os.path.exists(local_folder_path):
                    self.upload_dataset(
                        dataset_name=dataset_name.lower().replace("_", "-"),
                        local_folder_path=local_folder_path,
                        private=private,
                    )
                else:
                    logger.warning(
                        f"Folder {local_folder_path} does not exist. Skipping upload."
                    )
        except Exception as e:
            logger.error(f"Error while uploading datasets: {e}")
            raise e
