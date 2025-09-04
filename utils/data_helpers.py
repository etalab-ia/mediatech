import os
import shutil
import re
import json
import hashlib
import tarfile
import psycopg2
import duckdb
from datetime import datetime
from unidecode import unidecode
from .sheets_parser import RagSource
from config import (
    get_logger,
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    parquet_files_folder,
)

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


def make_schedule(plages: list) -> str:
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


def export_table_to_parquet(
    table_name: str,
    output_folder: str = parquet_files_folder,
    rows_per_file: int = 50000,
):
    """
    Exports tables from the PostgreSQL database to Parquet files.

    Args:
        table_name (str): The name of the table to export, or "all" for all tables.
        output_folder (str): The path where the Parquet files will be saved.
        rows_per_file (int): The number of rows to include in each Parquet file. Defaults to 50000 (250-300 MB per file approximately).

    Returns:
        None
    """
    try:
        conn = duckdb.connect()
        conn.execute("INSTALL postgres")
        conn.execute("LOAD postgres")

        conn.execute(f"""
            ATTACH 'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}' 
            AS postgres_db (TYPE postgres)
        """)

        def _export_single_table(table_name: str):
            """Exports a single table, splitting it into multiple flat Parquet files."""
            try:
                conn.execute(f"SELECT COUNT(*) FROM postgres_db.{table_name}")
                table_row_count = conn.fetchone()[0]

                if table_row_count == 0:
                    logger.warning(f"No data found in table '{table_name}', skipping.")
                    return

                num_files = (
                    table_row_count + rows_per_file - 1
                ) // rows_per_file  # Number of files

                logger.info(
                    f"Exporting {table_row_count} rows from table '{table_name}' into {num_files} Parquet file(s)..."
                )

                os.makedirs(os.path.join(output_folder, table_name), exist_ok=True)

                for i in range(num_files):
                    offset = i * rows_per_file
                    output_path = os.path.join(
                        output_folder, table_name, f"{table_name}_part_{i}.parquet"
                    )
                    logger.debug(f"Exporting part {i + 1}/{num_files} to {output_path}")

                    conn.execute(f"""
                        COPY (
                            SELECT * FROM postgres_db.{table_name}
                            LIMIT {rows_per_file} OFFSET {offset}
                        ) TO '{output_path}'
                        (FORMAT PARQUET, COMPRESSION 'ZSTD', PARQUET_VERSION 'V2', ROW_GROUP_SIZE 50000)
                    """)

                # Verifying the total number of rows in all created files
                global_path = os.path.join(
                    output_folder, table_name, f"{table_name}_part_*.parquet"
                )
                parquet_row_count = conn.execute(
                    f"SELECT COUNT(*) FROM read_parquet('{global_path}')"
                ).fetchone()[0]

                logger.info(
                    f"Successfully exported table '{table_name}': {table_row_count} rows -> {parquet_row_count} rows in {num_files} file(s)."
                )

            except Exception as table_error:
                logger.error(f"Error processing table '{table_name}': {table_error}")
                if table_name != "all":
                    raise

        os.makedirs(output_folder, exist_ok=True)

        if table_name == "all":
            conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE';"
            )
            tables = [row[0] for row in conn.fetchall()]
            logger.info(f"Found {len(tables)} tables to export: {tables}")

            for table in tables:
                _export_single_table(table_name=table)
        else:
            _export_single_table(table_name=table_name)

    except Exception as e:
        logger.error(f"An error occurred during SQL Table export: {e}")
        raise
    finally:
        if "conn" in locals():
            conn.close()


def extract_legi_data(data_type: str) -> list[str]:
    """
    Extract distinct data from the 'legi' PostgreSQL table.

    Args:
        data_type (str): Type of data to extract - 'categories' or 'codes'

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

    if data_type == "categories":
        # Listing all distinct categories in the 'legi' table
        cursor.execute("""SELECT DISTINCT category FROM legi""")
        data_list = [row[0] for row in cursor.fetchall()]
    elif data_type == "codes":
        # Listing all titles for 'CODE' category in the 'legi' table
        cursor.execute("""
            SELECT DISTINCT full_title FROM legi WHERE category = 'CODE'
        """)
        data_list = list(
            set(title[0].split(",")[0].strip() for title in cursor.fetchall())
        )
    else:
        data_list = []

    conn.close()
    return data_list


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


def _add_space_after_punctuation(text: str):
    return re.sub(r"([.,;:!?])([^\s\d])", r"\1 \2", text)


def load_experiences(storage_dir: str):
    with open(os.path.join(storage_dir, "export-expa-c-riences.json")) as f:
        documents = json.load(f)

    for d in documents:
        descr = d["description"]
        d["description"] = _add_space_after_punctuation(descr)

    return documents


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
