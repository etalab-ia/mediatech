import hashlib
import json
import os
import shutil
import tarfile
from datetime import datetime

import psycopg2
import requests
from tqdm import tqdm
from unidecode import unidecode

from config import (
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
    get_logger,
)

from .sheets_parser import RagSource

logger = get_logger(__name__)


def load_config(config_file_path: str):
    """
    Load and parse a JSON configuration file.

    Args:
        config_file_path (str): Path to the JSON configuration file to be loaded.

    Returns:
        dict: The parsed JSON content as a Python dictionary.

    Raises:
        FileNotFoundError: If the specified configuration file doesn't exist.
        json.JSONDecodeError: If the configuration file contains invalid JSON.
    """
    with open(config_file_path, "r") as file:
        return json.load(file)


def load_data_history(data_history_path: str):
    """
    Load the downloaded data history from a JSON file.
    This function attempts to read and parse a JSON file at the specified path.

    Parameters:
        data_history_path (str): Path to the JSON file containing the data history.

    Returns:
        dict: The data history as a dictionary. Returns an empty dictionary if the file
              does not exist, is empty, or contains invalid JSON.
    """
    if os.path.exists(data_history_path):
        with open(data_history_path, "r") as file:
            try:
                data = json.load(file)
                return data if data else {}
            except json.JSONDecodeError:
                logger.warning(f"File {data_history_path} is empty or invalid.")
                return {}
    else:
        with open(data_history_path, "w") as file:
            json.dump({}, file)
        return {}


def download_file(url: str, destination_path: str):
    """
    Downloads a file from an URL to a destination with a real-time progress bar.
    Uses streaming to handle large files efficiently.
    Args:
        url (str): The URL of the file to download.
        destination_path (str): The path (including filename) where the file will be saved.
    """
    logger.info(f"Downloading from {url}")
    try:
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            block_size = 1024 * 1024  # 1 MB

            logger.debug(f"Saving to {destination_path}")
            with tqdm(
                total=total_size,
                unit="iB",
                unit_scale=True,
                desc=os.path.basename(destination_path),
                leave=True,
            ) as progress_bar:
                with open(destination_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=block_size):
                        if chunk:
                            progress_bar.update(len(chunk))
                            f.write(chunk)

            # Check file size after download to verify completeness
            file_size = os.path.getsize(destination_path)
            if total_size > 0 and file_size != total_size:
                logger.warning(
                    f"Downloaded file size {file_size} bytes does not match expected {total_size} bytes. File may be incomplete."
                )
            elif total_size == 0:
                logger.info(
                    f"Downloaded file size: {file_size} bytes (content-length not provided by server)."
                )

        logger.info(f"Successfully downloaded {os.path.basename(destination_path)}")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        raise e


def extract_and_remove_tar_file(file_path: str, extract_path: str):
    """
    Extracts a .tar.gz file to a specified directory and then removes the archive.

    Args:
        file_path (str): The path to the .tar.gz file to be extracted.
        extract_path (str): The directory where the contents will be extracted.
    """
    if not os.path.exists(file_path):
        logger.debug(f"File {file_path} does not exist")
        return
    if not os.path.isdir(extract_path):
        logger.debug(f"Directory {extract_path} does not exist")
        return
    if file_path.endswith(".tar.gz"):
        logger.debug(f"Found {file_path}")
        logger.debug(f"Extracting {file_path}")
        try:
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(path=extract_path, members=tar.getmembers())
            os.remove(file_path)
            logger.debug(f"Removed {file_path}")
        except Exception as e:
            logger.error(f"Error extracting or removing {file_path}: {e}")


def extract_and_remove_tar_files(download_folder: str):
    """
    Extracts and removes all `.tar.gz` files in the specified folder.

    This function checks if the given folder exists and is not empty. It then iterates
    through all files in the folder, identifies files with a `.tar.gz` extension,
    extracts their contents into the same folder, and removes the original `.tar.gz` files.

    Args:
        download_folder (str): The path to the folder containing `.tar.gz` files.

    Logs:
        - A debug if the folder does not exist or is empty.
        - Information about each `.tar.gz` file found, extracted, and removed.
        - An error if there is an issue during extraction or removal.

    Raises:
        None: Any exceptions during extraction or removal are logged but not raised.
    """
    if not os.path.exists(download_folder):
        logger.debug(f"Folder {download_folder} does not exist")
        return
    if not os.listdir(download_folder):
        logger.debug(f"Folder {download_folder} is empty")
        return
    for file_name in os.listdir(download_folder):
        if file_name.endswith(".tar.gz"):
            logger.debug(f"Found {file_name}")
            file_path = os.path.join(download_folder, file_name)
            logger.debug(f"Extracting {file_path}")
            try:
                with tarfile.open(file_path, "r:gz") as tar:
                    tar.extractall(path=download_folder)
                os.remove(file_path)
                logger.debug(f"Removed {file_path}")
            except Exception as e:
                logger.error(f"Error extracting or removing {file_path}: {e}")


def format_subtitles(subtitles: str) -> str:
    """
    Extract concepts from subtitles by removing prefixes before colons.

    Args:
        subtitles (str): String containing subtitles separated by " - "

    Returns:
        str: Extracted concepts joined by " - "
    """
    if not subtitles:
        return ""

    parts = subtitles.split(" - ")
    concepts = []

    for part in parts:
        part = part.strip()
        if ":" in part:
            concept = part.split(":", 1)[1].strip()
            if len(concept) > 3:  # Avoid very short concepts
                concepts.append(concept)
        else:
            # Keep only parts that are longer than 3 characters
            if len(part) > 3:
                concepts.append(part)

    return " - ".join(concepts)


def format_time(time: str) -> str:
    """
    Formats a time string from "HH:MM:SS" to "HhMM" format.

    Args:
        time (str): A time string in the format "HH:MM:SS".

    Returns:
        str: The formatted time string in "HhMM" format (e.g., "9h30" for "09:30:00").
             Returns an empty string if the input is empty.
             Returns the original input if parsing fails.
    """
    if not time:
        return ""
    try:
        return datetime.strptime(time, "%H:%M:%S").strftime("%-Hh%M")
    except Exception:
        return time


def format_model_name(model_name: str) -> str:
    return model_name.partition("/")[2]


def file_sha256(file_path: str) -> str:
    """Computes the SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def _make_schedule(plages: list) -> str:
    """
    Generates a formatted schedule string from a list of time slot dictionaries.
    Used for the metadata 'plage_ouverture' from the directories.

    Each dictionary in the input list should represent a time slot (plage) with the following keys:
        - "nom_jour_debut": Name of the starting day.
        - "nom_jour_fin": Name of the ending day (optional, can be the same as start).
        - "valeur_heure_debut_1": Start time for the first period (optional).
        - "valeur_heure_fin_1": End time for the first period (optional).
        - "valeur_heure_debut_2": Start time for the second period (optional).
        - "valeur_heure_fin_2": End time for the second period (optional).
        - "commentaire": Additional comment for the time slot (optional).

    Returns:
        str: A formatted multi-line string representing the schedule, where each line corresponds to a time slot.
             If the input list is empty, returns the input as is.
    """
    lines = []
    if plages:
        for plage in plages:
            jour = plage["nom_jour_debut"]
            if (
                plage["nom_jour_debut"] != plage["nom_jour_fin"]
                and plage["nom_jour_fin"]
            ):
                jour += f" à {plage['nom_jour_fin']}"
            heures = []
            if plage["valeur_heure_debut_1"] and plage["valeur_heure_fin_1"]:
                heures.append(
                    f"{format_time(plage['valeur_heure_debut_1'])} à {format_time(plage['valeur_heure_fin_1'])}"
                )
            if plage["valeur_heure_debut_2"] and plage["valeur_heure_fin_2"]:
                heures.append(
                    f"{format_time(plage['valeur_heure_debut_2'])} à {format_time(plage['valeur_heure_fin_2'])}"
                )
            heure_str = " et de ".join(heures)
            line = f"{jour} : {heure_str}" if heure_str else f"{jour}"
            if plage.get("commentaire"):
                line += f". {plage['commentaire']}"
            lines.append(line.strip())
        schedule = "\n".join(lines)
    else:
        return plages
    return schedule


def remove_folder(folder_path: str):
    """
    Removes a folder and all its contents if it exists.

    Args:
        folder_path (str): The path to the folder to be removed.

    Returns:
        None

    Logs:
        - INFO: When the folder is successfully removed or doesn't exist.
        - ERROR: When there is an error while removing the folder.
    """
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
            logger.info(f"Removed folder: {folder_path}")
        except Exception as e:
            logger.error(f"Error removing folder {folder_path}: {e}")
    else:
        logger.info(f"Folder {folder_path} does not exist")


def remove_file(file_path: str):
    """
    Removes a file if it exists.

    Args:
        file_path (str): The path to the file to be removed.

    Returns:
        None

    Logs:
        - INFO: When the file is successfully removed or doesn't exist.
        - ERROR: When there is an error while removing the file.
    """
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.debug(f"Removed file: {file_path}")
        except Exception as e:
            logger.error(f"Error removing file {file_path}: {e}")
    else:
        logger.info(f"File {file_path} does not exist")


def _extract_distinct_data(data_type: str, source_table: str = "legi") -> list[str]:
    """
    Extract distinct data from a PostgreSQL table.

    Args:
        data_type (str): Type of data to extract - e.g : 'category' or 'codes'
        source_table (str): Name of the source table, default is 'legi'

    Returns:
        list[str]: List of distinct categories or code titles, empty list for invalid type
    """

    # Connect to PostgreSQL database
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )
    cursor = conn.cursor()

    if data_type == "codes":
        # Listing all titles for 'CODE' category in the 'legi' table
        cursor.execute(f"""
            SELECT DISTINCT full_title FROM {source_table.upper()} WHERE category = 'CODE'
        """)
        data_list = list(
            set(title[0].split(",")[0].strip() for title in cursor.fetchall())
        )
    else:
        # Listing all distinct {data_type} in the table
        cursor.execute(
            f"""SELECT DISTINCT {data_type.lower()} FROM {source_table.upper()}"""
        )
        data_list = [row[0] for row in cursor.fetchall()]

    conn.close()
    return data_list


def correct_wrong_column_contents(
    table_name: str, column_to_correct: str, column_helper: str
) -> str:
    """
    Correct null or empty content in a column using a helper column for prefix matching.
    Built to correct 'category' column in the 'legi' table.

    Args:
        table_name: Table to correct
        column_to_correct: Column with missing/empty values
        column_helper: Helper column used for matching

    Returns:
        Success message with correction count
    """
    # Get valid values and create lookup dict
    valid_values = _extract_distinct_data(
        data_type=column_to_correct, source_table=table_name
    )
    valid_values = [v for v in valid_values if v and v.strip()]

    if not valid_values:
        logger.warning(f"No valid values found for '{column_to_correct}'")
        return "No corrections needed - no valid values found."

    logger.info(f"Found {len(valid_values)} valid values for '{column_to_correct}'")

    # Connect to database
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )

    try:
        cursor = conn.cursor()

        # Get rows to correct
        cursor.execute(f"""
            SELECT chunk_id, {column_helper}
            FROM {table_name.upper()}
            WHERE {column_to_correct} IS NULL OR {column_to_correct} = ''
        """)
        rows_to_correct = cursor.fetchall()

        if not rows_to_correct:
            logger.info(f"No rows to correct in '{column_to_correct}'")
            return "No corrections needed - all rows valid."

        logger.info(f"Processing {len(rows_to_correct)} rows...")
        # Process and update rows
        corrections = 0
        for chunk_id, helper_value in rows_to_correct:
            if not helper_value:
                continue

            # Find matching value by prefix
            formatted = format_to_table_name(helper_value).upper()
            matched = next(
                (
                    valid_value
                    for valid_value in valid_values
                    if formatted.startswith(valid_value)
                ),
                None,
            )

            if matched:
                cursor.execute(
                    f"UPDATE {table_name.upper()} SET {column_to_correct} = %s WHERE chunk_id = %s",
                    (matched, chunk_id),
                )
                corrections += 1

        conn.commit()
        logger.info(f"Successfully corrected {corrections}/{len(rows_to_correct)} rows")

    except Exception as e:
        conn.rollback()
        logger.error(f"Error correcting '{column_to_correct}': {e}")
        raise
    finally:
        conn.close()


def format_to_table_name(name: str) -> str:
    """
    Convert a string to a valid table name format.

    Removes accents, replaces special characters with underscores,
    removes parentheses, and converts to lowercase.

    Args:
        name: The input string to format

    Returns:
        A formatted string suitable for use as a table name
    """
    name = unidecode(name)

    cleaned_title = name
    for char, replacement in {
        "'": "_",
        " ": "_",
        "`": "_",
        "-": "_",
        ",": "_",
        "(": "",
        ")": "",
    }.items():
        cleaned_title = cleaned_title.replace(char, replacement)
    formated_name = cleaned_title.lower()
    return formated_name


### Imported functions from the pyalbert library


def doc_to_chunk(doc: dict) -> str | None:
    context = ""
    if doc.get("context"):
        context = "  ( > ".join(doc["context"]) + ")"
    if doc.get("introduction") not in doc["text"]:
        chunk_text = "\n".join(
            [doc["title"] + context, doc["introduction"], doc["text"]]
        )
    else:
        chunk_text = "\n".join([doc["title"] + context, doc["text"]])

    return chunk_text


def load_sheets(storage_dir: str, sources: str | list[str]):
    documents = RagSource.get_sheets(
        storage_dir=storage_dir,
        sources=sources,
        structured=False,
    )
    documents = [d for d in documents if d["text"][0]]
    for doc in documents:
        doc["text"] = doc["text"][0]

    return documents
