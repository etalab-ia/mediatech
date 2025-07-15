import json
import os
from . import load_data_history, file_md5, remove_file
from config import (
    get_logger,
    data_history_path,
    config_file_path,
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

    def is_dataset_up_to_date(self, dataset_name: str, local_file_path: str) -> bool:
        """
        Checks if remote version on Hugging Face is already up to date compared to the local dataset file.
        This method compares the local dataset file's MD5 hash with the remote file's hash.
        If the hashes match, it means the local file is up to date with the remote version.

        Args:
            dataset_name: Name of the HF dataset to check
            file_path: Local dataset file path to compare

        Returns:
            bool: True if local file matches remote file, False otherwise
        """
        repo_id = f"{self.hugging_face_repo}/{dataset_name}"
        path_in_repo = f"data/{dataset_name}-latest.parquet"
        try:
            # Download the remote file
            hf_local_path = self.api.hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=path_in_repo,
                token=self.token,
            )
            remote_hash = file_md5(hf_local_path)
            local_hash = file_md5(local_file_path)
            logger.info(
                f"Comparing local file hash {local_hash} with remote file hash {remote_hash}"
            )

            # Remove the downloaded file
            remove_file(file_path=hf_local_path)

            return local_hash == remote_hash
        except Exception as e:
            logger.error(f"Error checking if dataset {dataset_name} is up to date: {e}")
            # If the dataset does not exist or any other error occurs, we assume it's not up to date
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
                        log = load_data_history(data_history_path=data_history_path)
                        for file_name, attributes in log.items():
                            if file_name == dataset_name:
                                last_hf_upload_date = attributes.get(
                                    "last_hf_upload_date", ""
                                )
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
                            else:
                                pass
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

    def rename_old_latest_file(self, dataset_name: str):
        """
        Rename the old latest data file in the Hugging Face dataset repo.

        Args:
            repo_id (str): The Hugging Face dataset repo id (e.g., "user/dataset-name").
            file_name (str): The base name for the file to be renamed.
        """
        repo_id = f"{self.hugging_face_repo}/{dataset_name}"
        old_file_path = f"data/{dataset_name}-latest.parquet"
        old_file_date = self.get_file_upload_date(
            dataset_name=dataset_name, hf_file_path=old_file_path
        )
        if old_file_date is None:
            logger.warning(
                f"Failed to retrieve the upload date for {old_file_path}. Cannot rename the old latest file."
            )
            return
        new_file_path = f"data/{dataset_name}-{old_file_date}.parquet"

        operations = [
            CommitOperationCopy(
                src_path_in_repo=old_file_path, path_in_repo=new_file_path
            ),
            CommitOperationDelete(path_in_repo=old_file_path, is_folder=False),
        ]

        self.api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            token=self.token,
            commit_message=f"Renamed {old_file_path} to {new_file_path}",
        )
        logger.info(
            f"Old file {old_file_path} successfuly renamed to {new_file_path} in repository: {repo_id}."
        )

    def upload_dataset(self, dataset_name: str, file_path: str, private: bool = False):
        """
        Upload a parquet file to the Hugging Face dataset repo, creating the repo if needed.

        Args:
            dataset_name (str): The Hugging Face dataset name (e.g. : "service-public").
            file_path (str): Local path to the parquet file.
            private (bool): Whether the repo should be private if created.
        """

        # Check if the local file exists
        if not os.path.exists(file_path):
            logger.error(f"File {file_path} does not exist.")
            raise FileNotFoundError(f"File {file_path} does not exist.")

        # Check if the local file is a Parquet file
        if not file_path.endswith(".parquet"):
            logger.error(f"File {file_path} is not a Parquet file.")
            raise ValueError(f"File {file_path} is not a Parquet file.")

        # Check if the dataset is already up to date
        if self.is_dataset_up_to_date(dataset_name, file_path):
            logger.info(
                f"The dataset {dataset_name} is already up to date in the Hugging Face repository. No need to upload it again."
            )
            return

        # Create the repo if it does not exist
        repo_id = f"{self.hugging_face_repo}/{dataset_name}"
        if not self.api.repo_exists(repo_id, repo_type="dataset"):
            self.api.create_repo(
                repo_id, repo_type="dataset", token=self.token, private=private
            )
            logger.info(f"Hugging Face repository: {repo_id} successfuly created.")

        path_in_repo = f"data/{dataset_name}-latest.parquet"

        # Upload the file to the Hugging Face dataset repo
        try:
            self.rename_old_latest_file(dataset_name=dataset_name)

            self.api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                token=self.token,
                commit_message=f"Updating {dataset_name} dataset file",
            )
            logger.info(
                f"File {file_path.split('/')[-1]} successfuly uploaded to Hugging Face repository: {repo_id}."
            )

            # Update the data history file with the last Hugging Face upload date
            log = load_data_history(data_history_path=data_history_path)
            date = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for dataset, attributes in log.items():
                if dataset.startswith(dataset_name):
                    log[dataset]["last_hf_upload_date"] = date

            with open(data_history_path, "w") as file:
                json.dump(log, file, indent=4)
            logger.info(f"Log config file successfully updated to {data_history_path}")
        except Exception as e:
            logger.error(
                f"Error while uploading file {file_path} to the Hugging Face repository {repo_id}: {e}"
            )
            raise e

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
                if attributes.get("download_folder") in [
                    SERVICE_PUBLIC_PRO_DATA_FOLDER,
                    SERVICE_PUBLIC_PART_DATA_FOLDER,
                ]:
                    dataset_name = "service_public"
                file_path = f"data/parquet/{dataset_name.lower()}.parquet"
                if os.path.exists(file_path):
                    self.upload_dataset(
                        dataset_name=dataset_name.lower().replace("_", "-"),
                        file_path=file_path,
                        private=private,
                    )
                else:
                    logger.warning(f"File {file_path} does not exist. Skipping upload.")
        except Exception as e:
            logger.error(f"Error while uploading datasets: {e}")
            raise e
