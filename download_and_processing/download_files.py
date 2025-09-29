from datetime import datetime
from urllib.request import urlopen
from bs4 import BeautifulSoup
from urllib.error import HTTPError
from utils import (
    load_data_history,
    load_config,
    download_file,
    extract_and_remove_tar_file,
)
from .files_processing import process_data
from config import get_logger, BASE_PATH
import os
import requests
import json
import shutil

logger = get_logger(__name__)


def download_and_optionally_process_files(
    data_name: str,
    config_file_path: str,
    data_history_path: str,
    process: bool = False,
    streaming: bool = False,
    model: str = "BAAI/bge-m3",
):
    """
    Download and optionally process files based on the data configuration type.
    Downloads files, extracts archives if needed, optionally processes data using specified model,
    and updates download history.
    Args:
        data_name: Name of the data source to process
        config_file_path: Path to configuration file containing download settings
        data_history_path: Path to JSON file tracking download history
        process: Flag to indicate whether to process the data after download (default: False)
        streaming: Flag to indicate whether to stream extraction of tar files, for DILA files only (default: True)
        model: Model name for data processing (default: "BAAI/bge-m3")
    """

    config = load_config(config_file_path=config_file_path)
    log = load_data_history(data_history_path=data_history_path)
    try:
        attributes = config.get(data_name.lower(), {})
        if attributes.get("type") == "dila_folder":
            url = attributes.get("download_url", "")
            download_folder = os.path.join(
                BASE_PATH, attributes.get("download_folder", "")
            )
            # Ensure the download folder exists
            os.makedirs(download_folder, exist_ok=True)
            try:
                last_downloaded_file = log.get(data_name).get(
                    "last_downloaded_file", ""
                )
            except Exception:
                last_downloaded_file = ""

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
                    logger.warning(f"There is no freemium file in {url}")

                logger.debug(
                    f"{len(tar_gz_files)} tar.gz files found in {url}: {tar_gz_files}"
                )

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
                        file_url = os.path.join(url, filename)
                        download_path = os.path.join(download_folder, filename)

                        download_file(url=file_url, destination_path=download_path)

                        if not streaming:
                            extract_and_remove_tar_file(
                                file_path=download_path, extract_path=download_folder
                            )
                        if process:
                            # Process the downloaded file and remove the folder after processing
                            process_data(
                                base_folder=download_folder,
                                streaming=streaming,
                                model=model,
                            )

                            logger.info(
                                f"Successfully downloaded and processed {filename}"
                            )
                        else:
                            logger.info(f"Successfully downloaded {filename}")

                        # Update the last download file and date in the log
                        log[data_name] = {
                            "last_downloaded_file": filename,
                            "last_download_date": datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                        }

                        with open(data_history_path, "w") as file:
                            json.dump(log, file, indent=4)
                        logger.info(
                            f"Log config file successfully updated to {data_history_path}"
                        )

            except Exception as e:
                logger.error(f"Error downloading files: {e}")
                raise e

        elif attributes.get("type") == "directory":
            download_folder = os.path.join(
                BASE_PATH, attributes.get("download_folder", "")
            )
            os.makedirs(download_folder, exist_ok=True)

            try:
                last_download_date = log.get(data_name).get("last_download_date", "")
            except Exception:
                last_download_date = ""
            try:
                url = requests.head(
                    attributes["download_url"], allow_redirects=True
                ).url
                info = urlopen(url).info()
                file = (
                    info.get_filename()
                    if info.get_filename()
                    else os.path.basename(url)
                )
                last_modified = info.get("Last-Modified")
                last_modified = datetime.strptime(
                    last_modified, "%a, %d %b %Y %H:%M:%S GMT"
                ).strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                logger.debug(f"Error fetching metadata for {data_name}: {e}")
                # If the last modified date is not available, set it to a far future date
                last_modified = datetime.strptime(
                    "9999-12-31 23:59:59", "%Y-%m-%d %H:%M:%S"
                ).strftime("%Y-%m-%d %H:%M:%S")

            old_files = os.listdir(download_folder)
            logger.info(f"downloading {data_name} archive...")
            # Checking if the file was already downloaded
            if last_download_date > last_modified:
                logger.info(
                    f"Last downloaded date is {last_download_date} according to the data history and {data_name} files have been lastly updated the {last_modified}. No new files to download."
                )
                return
            else:
                logger.info(
                    f"Last downloaded date is {last_download_date} according to the data history. {data_name} files have been updated the {last_modified}. Downloading new files."
                )

                try:
                    downloaded_file_path = os.path.join(download_folder, file)
                    download_file(
                        url=attributes["download_url"],
                        destination_path=downloaded_file_path,
                    )
                except Exception as e:
                    logger.error(f"Error downloading files: {e}")
                    raise e

                logger.debug(f"unpacking {data_name} archive...")
                shutil.unpack_archive(
                    os.path.join(download_folder, file), download_folder
                )

                shutil.unpack_archive(downloaded_file_path, download_folder)
                os.remove(downloaded_file_path)

                new_files = [
                    x for x in os.listdir(download_folder) if x not in old_files
                ]
                logger.debug(f"new files: {new_files}")

                for downloaded_file in new_files:
                    if not downloaded_file.endswith(".json"):
                        logger.debug(f"deleting {downloaded_file}...")
                        os.remove(os.path.join(download_folder, downloaded_file))

                    else:
                        logger.debug(
                            f"renaming {downloaded_file} to {data_name}.json..."
                        )
                        os.rename(
                            os.path.join(download_folder, downloaded_file),
                            os.path.join(download_folder, f"{data_name}.json"),
                        )

                        logger.debug(
                            f"Successfully downloaded {downloaded_file} to {download_folder}"
                        )

                        if process:
                            # Process the downloaded file and remove the folder after processing
                            process_data(base_folder=download_folder, model=model)

                            logger.info(
                                f"Successfully downloaded and processed {data_name}"
                            )
                            break
                        else:
                            logger.info(f"Successfully downloaded {data_name}")

                        # Update the last download file and date in the log
                        log[data_name] = {
                            "last_downloaded_file": downloaded_file,
                            "last_download_date": datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                        }

                        with open(data_history_path, "w") as file:
                            json.dump(log, file, indent=4)
                        logger.info(
                            f"Log config file successfully updated to {data_history_path}"
                        )

        elif attributes.get("type") == "sheets":
            # Script based on the pyalbert.corpus.download_rag_sources function
            download_folder = os.path.join(
                BASE_PATH, attributes.get("download_folder", "")
            )

            try:
                last_download_date = log.get(data_name).get("last_download_date", "")
            except Exception:
                last_download_date = ""

            # create the storage path if it does not exist
            os.makedirs(download_folder, exist_ok=True)
            target = f"{download_folder}/{data_name}"
            filename_tmp = f"{download_folder}/temp_{data_name}"

            try:
                url = requests.head(
                    attributes["download_url"], allow_redirects=True
                ).url
                info = urlopen(url).info()
                last_modified = info.get("Last-Modified")
                last_modified = datetime.strptime(
                    last_modified, "%a, %d %b %Y %H:%M:%S GMT"
                ).strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                logger.debug(f"Error fetching metadata for {data_name}: {e}")
                # If the last modified date is not available, set it to a far future date
                last_modified = datetime.strptime(
                    "9999-12-31 23:59:59", "%Y-%m-%d %H:%M:%S"
                ).strftime("%Y-%m-%d %H:%M:%S")

            logger.info(f"Downloading '{data_name}'...")

            # Checking if the file was already downloaded
            if last_download_date > last_modified:
                logger.info(
                    f"Last downloaded date is {last_download_date} according to the data history and {data_name} files have been lastly updated the {last_modified}. No new files to download."
                )
                return
            else:
                logger.info(
                    f"Last downloaded date is {last_download_date} according to the data history. {data_name} files have been updated the {last_modified}. Downloading new files."
                )

                if os.path.exists(filename_tmp):
                    os.remove(filename_tmp)
                try:
                    download_file(
                        url=attributes.get("download_url"),
                        destination_path=filename_tmp,
                    )

                except HTTPError as err:
                    logger.error(f"Error: {err}")
                    logger.error(
                        f"Failed to fetch source {data_name} from {data_name['url']}"
                    )

                url = requests.head(
                    attributes["download_url"], allow_redirects=True
                ).url
                info = urlopen(url).info()
                downloaded_file_name = (
                    info.get_filename()
                    if info.get_filename()
                    else os.path.basename(url)
                )
                content_type = info.get_content_type().split("/")[-1]
                if content_type in ["zip"]:  # List can be extended with other formats
                    if os.path.exists(target):
                        shutil.rmtree(target)
                    shutil.unpack_archive(
                        filename_tmp, extract_dir=target, format=content_type
                    )
                else:
                    target = f"{target}.{downloaded_file_name.split('.')[-1]}"
                    shutil.move(
                        filename_tmp, target
                    )  # Renaming the file with the correct extension

                if process:
                    # Process the downloaded file and remove the folder after processing
                    process_data(base_folder=download_folder, model=model)

                    logger.info(f"Successfully downloaded and processed {data_name}")
                else:
                    logger.info(f"Successfully downloaded {data_name}")

                # Update the last download file and date in the log
                log[data_name] = {
                    "last_downloaded_file": f"{downloaded_file_name}",
                    "last_download_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

                with open(data_history_path, "w") as file:
                    json.dump(log, file, indent=4)
                logger.info(
                    f"Log config file successfully updated to {data_history_path}"
                )

        elif attributes.get("type") == "data_gouv":
            logger.info(f"Downloading '{data_name}'...")
            url = attributes.get("download_url", "")
            download_folder = os.path.join(
                BASE_PATH, attributes.get("download_folder", "")
            )
            try:
                last_downloaded_file = log.get(data_name).get(
                    "last_downloaded_file", ""
                )
            except Exception:
                last_downloaded_file = ""

            os.makedirs(download_folder, exist_ok=True)

            if data_name == "data_gouv_datasets_catalog":
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
                    datasets = sorted(datasets, key=lambda x: x["title"].lower())
                    download_url = datasets[0].get("url")
                    info = urlopen(download_url).info()
                    downloaded_file_name = (
                        info.get_filename()
                        if info.get_filename()
                        else os.path.basename(download_url)
                    )
                    if last_downloaded_file == downloaded_file_name:
                        logger.info(
                            f"Last downloaded file is {last_downloaded_file} according to the data history. No new files to download."
                        )
                    else:
                        try:
                            logger.info(
                                f"Downloading {downloaded_file_name} from {download_url}..."
                            )
                            # Download the file
                            download_file(
                                url=download_url,
                                destination_path=os.path.join(
                                    download_folder, f"{data_name}.csv"
                                ),
                            )

                            logger.info(
                                f"Successfully downloaded {downloaded_file_name} to {download_folder} as {data_name}.csv"
                            )

                            if process:
                                # Process the downloaded file and remove the folder after processing
                                process_data(base_folder=download_folder, model=model)

                                logger.info(
                                    f"Successfully downloaded and processed {data_name}"
                                )
                            else:
                                logger.info(f"Successfully downloaded {data_name}")

                            # Update the last download file and date in the log
                            log[data_name] = {
                                "last_downloaded_file": downloaded_file_name,
                                "last_download_date": datetime.now().strftime(
                                    "%Y-%m-%d %H:%M:%S"
                                ),
                            }

                            with open(data_history_path, "w") as file:
                                json.dump(log, file, indent=4)
                            logger.info(
                                f"Log config file successfully updated to {data_history_path}"
                            )

                        except Exception as e:
                            logger.error(f"Error downloading files: {e}")
                            raise e
                except Exception as e:
                    logger.error(f"Error downloading data_gouv datasets: {e}")
                    raise e
            else:
                logger.error(
                    f"File : {data_name} is not a supported file, skipping download."
                )

        else:
            logger.error(f"Unknown type {attributes.get('type')} for {data_name}")

    except Exception as e:
        logger.error(f"Error processing {data_name}: {e}")
        return


def download_and_optionally_process_all_files(
    config_file_path: str,
    data_history_path: str,
    process: bool = False,
    model: str = "BAAI/bge-m3",
):
    """
    Downloads and optionally processes all files listed in the configuration file.
    This function iterates through each data source defined in the configuration,
    downloads the files, optionally processes them, and updates the history log by using `download_and_process_files()`.

    Args:
        config_file_path (str): Path to the JSON configuration file.
        data_history_path (str): Path to the JSON history file.
        process (bool): Flag to indicate whether to process the data after download (default: False).
        model (str): Model name for data processing (default: "BAAI/bge-m3").
    """
    config = load_config(config_file_path=config_file_path)

    for data_name in config.keys():
        if process:
            logger.info(f"Downloading and processing {data_name}...")
        else:
            logger.info(f"Downloading {data_name}...")
        download_and_optionally_process_files(
            data_name=data_name,
            config_file_path=config_file_path,
            data_history_path=data_history_path,
            process=process,
            model=model,
        )
