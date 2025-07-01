from datetime import datetime
from urllib.request import urlopen
from bs4 import BeautifulSoup
from urllib.error import HTTPError
from utils import extract_and_remove_tar_files, load_data_history, load_config
from config import get_logger
import os
import requests
import json
import shutil
import wget

logger = get_logger(__name__)


def download_files(config_file_path: str, data_history_path: str):
    """
    Downloads and processes files according to a configuration file and maintains a history in a history file.

    This function reads a configuration file listing files or folders to download.
    It handles different types of downloads and updates a history log after each operation.

    Args:
        config_file_path (str): Path to the JSON configuration file.
        data_history_path (str): Path to the JSON history file.

    Raises:
        Exception: Errors are logged.
    """

    config = load_config(config_file_path=config_file_path)
    log = load_data_history(data_history_path=data_history_path)
    for file_name, attributes in config.items():
        if attributes.get("type") == "dila_folder":
            url = attributes.get("download_url", "")
            download_folder = attributes.get("download_folder", "")
            last_downloaded_file = log.get(file_name).get("last_downloaded_file", "")

            try:
                response = requests.get(url)
                response.raise_for_status()

                # Parse the HTML content using BeautifulSoup
                soup = BeautifulSoup(response.text, "html.parser")

                # Find all links that end with ".tar.gz"
                links = soup.find_all("a", href=True)

                tar_gz_files = sorted(
                    [link["href"] for link in links if link["href"].endswith(".tar.gz")]
                )
                # Placing the freemium file at the beginning
                try:
                    freemium_file = next(
                        (
                            file
                            for file in tar_gz_files
                            if file.lower().startswith("freemium")
                        ),
                        None,
                    )
                    tar_gz_files.remove(freemium_file)
                    tar_gz_files.insert(0, freemium_file)
                except ValueError:
                    logger.info(f"There is no freemium file in {url}")

                logger.info(
                    f"{len(tar_gz_files)} tar.gz files found in {url}: {tar_gz_files}"
                )

                # Ensure the download folder exists
                os.makedirs(download_folder, exist_ok=True)

                if last_downloaded_file in tar_gz_files:
                    last_file_index = tar_gz_files.index(last_downloaded_file)
                    logger.info(
                        f"Last downloaded file is {last_downloaded_file} according to the data history"
                    )
                else:
                    last_file_index = -1

                if last_file_index == len(tar_gz_files) - 1:
                    logger.info("No new files to download")
                else:
                    for filename in tar_gz_files[
                        last_file_index + 1 :
                    ]:  # As we already downloaded the last file, we start from the next file
                        file_url = url + filename
                        download_path = os.path.join(download_folder, filename)
                        logger.info(f"Downloading {file_url} to {download_folder}")

                        file_response = requests.get(file_url)
                        file_response.raise_for_status()
                        with open(download_path, "wb") as file:
                            file.write(file_response.content)
                        logger.info(
                            f"Successfully downloaded {filename} to {download_folder}"
                        )

                    # Update the last download file and date in the log
                    log[file_name] = {
                        "last_downloaded_file": filename,
                        "last_download_date": datetime.now().strftime(
                            "%d-%m-%Y %H:%M:%S"
                        ),
                    }

                    with open(data_history_path, "w") as file:
                        json.dump(log, file, indent=4)
                    logger.info(
                        f"Log config file successfully updated to {data_history_path}"
                    )

                extract_and_remove_tar_files(download_folder)
            except Exception as e:
                logger.error(f"Error downloading files: {e}")

        elif attributes.get("type") == "directory":
            storage_dir = attributes.get("download_folder", "")
            os.makedirs(storage_dir, exist_ok=True)
            old_files = os.listdir(storage_dir)

            logger.info(f"downloading {file_name} archive...")
            url = requests.head(attributes["download_url"], allow_redirects=True).url
            info = urlopen(url).info()
            file = info.get_filename() if info.get_filename() else os.path.basename(url)

            try:
                wget.download(
                    attributes["download_url"], os.path.join(storage_dir, file)
                )
            except Exception as e:
                logger.error(f"Error downloading files: {e}")

            logger.info(f"unpacking {file_name} archive...")
            shutil.unpack_archive(os.path.join(storage_dir, file), storage_dir)

            logger.info(f"deleting {file_name} archive...")
            os.remove((os.path.join(storage_dir, file)))

            new_files = [x for x in os.listdir(storage_dir) if x not in old_files]
            logger.debug(f"new files: {new_files}")

            for downloaded_file in new_files:
                if not downloaded_file.endswith(".json"):
                    logger.debug(f"deleting {downloaded_file}...")
                    os.remove(os.path.join(storage_dir, downloaded_file))

                else:
                    logger.debug(f"renaming {downloaded_file} to {file_name}.json...")
                    os.rename(
                        os.path.join(storage_dir, downloaded_file),
                        os.path.join(storage_dir, f"{file_name}.json"),
                    )

                    # Update the last download file and date in the log
                    log[file_name] = {
                        "last_downloaded_file": downloaded_file,
                        "last_download_date": datetime.now().strftime(
                            "%d-%m-%Y %H:%M:%S"
                        ),
                    }

                    with open(data_history_path, "w") as file:
                        json.dump(log, file, indent=4)
                    logger.info(
                        f"Log config file successfully updated to {data_history_path}"
                    )

            logger.info(f"{file_name} successfully downloaded")

        elif attributes.get("type") == "sheets":
            # Script based on the pyalbert.corpus.download_rag_sources function)
            storage_dir = attributes.get("download_folder", "")

            # create the storage path if it does not exist
            os.makedirs(storage_dir, exist_ok=True)
            target = f"{storage_dir}/{file_name}"
            filename_tmp = f"{storage_dir}/temp_{file_name}"

            logger.info(f"Downloading '{file_name}'...")
            if os.path.exists(filename_tmp):
                os.remove(filename_tmp)
            try:
                old_name = wget.download(
                    attributes.get("download_url"), out=storage_dir
                )
                shutil.move(old_name, filename_tmp)  # Renaming the file to temp
            except HTTPError as err:
                logger.error(f"Error: {err}")
                logger.error(
                    f"Failed to fetch source {file_name} from {file_name['url']}"
                )
                continue

            url = requests.head(attributes["download_url"], allow_redirects=True).url
            info = urlopen(url).info()
            downloaded_file_name = (
                info.get_filename() if info.get_filename() else os.path.basename(url)
            )
            content_type = info.get_content_type().split("/")[-1]
            if content_type in ["zip"]:  # List can be extended with other formats
                if os.path.exists(target):
                    shutil.rmtree(target)
                shutil.unpack_archive(
                    filename_tmp, extract_dir=target, format=content_type
                )
            else:
                target = f"{target}.{old_name.split('.')[-1]}"
                shutil.move(
                    filename_tmp, target
                )  # Renaming the file with the correct extension

            # Update the last download file and date in the log
            log[file_name] = {
                "last_downloaded_file": f"{downloaded_file_name}",
                "last_download_date": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            }

            with open(data_history_path, "w") as file:
                json.dump(log, file, indent=4)
            logger.info(f"Log config file successfully updated to {data_history_path}")

            logger.info(f"Corpus files {file_name} successfuly downloaded")
        elif attributes.get("type") == "data_gouv":
            logger.info(f"Downloading '{file_name}'...")
            url = attributes.get("download_url", "")
            storage_dir = attributes.get("download_folder", "")
            try:
                last_downloaded_file = log.get(file_name).get(
                    "last_downloaded_file", ""
                )
            except Exception as e:
                last_downloaded_file = ""

            os.makedirs(storage_dir, exist_ok=True)

            if file_name == "data_gouv_datasets_catalog":
                try:
                    response = requests.get(url)
                    resources = response.json().get("resources")
                    datasets = []
                    for resource in resources:
                        if resource.get("title").startswith(
                            "export-dataset"
                        ) and resource.get("title").endswith(".csv"):
                            # Filter out datasets that are not CSV files
                            datasets.append(
                                {
                                    "title": resource.get("title"),
                                    "url": resource.get("url"),
                                }
                            )
                        else:
                            continue
                    if not datasets:
                        logger.error(
                            f"No datasets found in {url} that match the criteria."
                        )
                        continue
                    datasets = sorted(datasets, key=lambda x: x["title"].lower())
                    download_url = datasets[0].get("url")
                    info = urlopen(download_url).info()
                    downloaded_file_name = (
                        info.get_filename()
                        if info.get_filename()
                        else os.path.basename(download_url)
                    )
                    if last_downloaded_file == file:
                        logger.info(
                            f"Last downloaded file is {last_downloaded_file} according to the data history. No new files to download."
                        )
                        continue
                    else:
                        try:
                            wget.download(
                                download_url,
                                os.path.join(storage_dir, f"{file_name}.csv"),
                            )

                            logger.info(
                                f"Successfully downloaded {downloaded_file_name} to {storage_dir} as {file_name}.csv"
                            )

                            # Update the last download file and date in the log
                            log[file_name] = {
                                "last_downloaded_file": downloaded_file_name,
                                "last_download_date": datetime.now().strftime(
                                    "%d-%m-%Y %H:%M:%S"
                                ),
                            }

                            with open(data_history_path, "w") as file:
                                json.dump(log, file, indent=4)
                            logger.info(
                                f"Log config file successfully updated to {data_history_path}"
                            )

                        except Exception as e:
                            logger.error(f"Error downloading files: {e}")
                except Exception as e:
                    logger.error(f"Error downloading data_gouv datasets: {e}")
                    continue
            else:
                logger.error(
                    f"File : {file_name} is not a supported file, skipping download."
                )
                continue

        else:
            logger.error(f"Unknown type {attributes.get('type')} for {file_name}")
