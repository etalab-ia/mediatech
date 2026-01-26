import gc
import json
import os
import sys
import tarfile
import xml.etree.ElementTree as ET
from datetime import datetime

import pandas as pd
import xxhash
from openai import PermissionDeniedError

from config import BASE_PATH, SOURCE_MAP, config_file_path, get_logger
from database import insert_data, refresh_table, remove_data
from utils import (
    CheckpointManager,
    CorpusHandler,
    _dole_cut_exp_memo,
    _dole_cut_file_content,
    _make_schedule,
    format_subtitles,
    generate_embeddings_with_retry,
    load_config,
    make_chunks,
    make_chunks_sheets,
    make_directory_text,
    remove_file,
    remove_folder,
)

logger = get_logger(__name__)

# Setting a higher recursion limit for processing large files
sys.setrecursionlimit(limit=10000)


def _process_data_gouv_content(
    df: pd.DataFrame,
    table_name: str,
    checkpoint: CheckpointManager,
    last_processed_index: int,
    model: str,
):
    """
    Process data.gouv.fr content with checkpoint support for resume capability.

    Args:
        df (pd.DataFrame): DataFrame containing the data to process
        table_name (str): Name of the database table to insert data into
        checkpoint (CheckpointManager): Checkpoint manager for saving progress
        last_processed_index (int): Index of the last processed row
        model (str): Model name for embedding generation
    """

    df = df[
        df["description"].str.len() >= 100
    ]  # Filter out rows with short descriptions
    df["chunk_text"] = (
        df["title"].astype(str)
        + "\n"
        + df["organization"].astype(str)
        + "\n"
        + df["description"].astype(str)
    )

    total_rows = len(df)
    logger.info(
        f"Total rows to process: {total_rows}, "
        f"Starting from index: {last_processed_index + 1}"
    )

    for idx, (_, row) in enumerate(df.iterrows()):
        # Skip already processed rows
        if idx <= last_processed_index:
            continue

        # Log progress every 100 rows
        if idx > 0 and idx % 100 == 0:
            logger.info(
                f"Processing row {idx}/{total_rows} ({(idx / total_rows) * 100:.1f}%)"
            )

        # Replace nan values with None in the current row
        row = row.where(pd.notna(row), None)
        # Making chunks
        chunk_text = make_chunks(
            text=row["chunk_text"],
            chunk_size=1000,
            chunk_overlap=0,
            length_function="len",
        )[
            0
        ]  # Only keep the first chunks because a too long description is not interesting for this kind of dataset

        chunk_xxh64 = xxhash.xxh64(chunk_text.encode("utf-8"), seed=2025).hexdigest()

        embeddings = generate_embeddings_with_retry(
            data=chunk_text, attempts=5, model=model
        )[0]

        doc_id = row.get("slug", None)

        new_data = (
            row.get("id"),  # Primary key (chunk_id)
            doc_id,
            chunk_xxh64,  # Hash of chunk_text
            row.get("title", None),
            row.get("acronym", None),
            row.get("url", None),
            row.get("organization", None),
            row.get("organization_id", None),
            row.get("owner", None),
            row.get("owner_id", None),
            row.get("description", None),
            row.get("frequency", None),
            row.get("license", None),
            row.get("temporal_coverage.start", None),
            row.get("temporal_coverage.end", None),
            row.get("spatial.granularity", None),
            row.get("spatial.zones", None),
            row.get("featured", None),
            row.get("created_at", None),
            row.get("last_modified", None),
            row.get("tags", None),  # Convert tags to JSON string
            row.get("archived", None),
            row.get("resources_count", None),
            row.get("main_resources_count", None),
            row.get("resources_formats", None),
            row.get("harvest.backend", None),
            row.get("harvest.domain", None),
            row.get("harvest.created_at", None),
            row.get("harvest.modified_at", None),
            row.get("harvest.remote_url", None),
            row.get("quality_score", None),
            row.get("metric.discussions", None),
            row.get("metric.reuses", None),
            row.get("metric.reuses_by_months", None),
            row.get("metric.followers", None),
            row.get("metric.followers_by_months", None),
            row.get("metric.views", None),
            row.get("metric.resources_downloads", None),
            chunk_text,  # The text chunk for embedding
            embeddings,  # The embedding vector
        )

        try:
            insert_data(data=[new_data], table_name=table_name)
            # Save checkpoint after successful insertion
            checkpoint.save(idx, metadata={"doc_id": doc_id, "table": table_name})
        except Exception as e:
            logger.error(f"Error inserting data for row {idx} (doc_id: {doc_id}): {e}")
            logger.error(
                "Progress saved. Restart the process to resume from this point."
            )
            raise e

    # All rows processed successfully
    checkpoint.remove()
    logger.info(f"Successfully processed all {total_rows} rows")


def process_data_gouv_files(table_name: str, model: str = "BAAI/bge-m3"):
    """
    Process data.gouv.fr files by generating embeddings and storing them in database.
    The workflow depends on the file.

    Implements a checkpoint system to resume processing from the last successfully
    processed row in case of errors.

    Args:
        table_name (str): Name of the table to process
        model (str): Model name for embedding generation. Defaults to "BAAI/bge-m3"

    """
    config = load_config(config_file_path=config_file_path)
    data_sources = config.get(table_name.lower(), {})

    for data_source, attributes in data_sources.items():
        if attributes.get("type") == "data_gouv":
            target_dir = os.path.join(BASE_PATH, attributes.get("download_folder", ""))

            csv_path = f"{target_dir}/{data_source}.csv"

            # Initialiser le checkpoint manager
            checkpoint = CheckpointManager(csv_path)
            last_processed_index = checkpoint.load()

            df = pd.read_csv(csv_path, sep=";", encoding="utf-8")

            if last_processed_index == -1:
                # First run: use refresh_table for optimization
                logger.info("First run detected, using refresh_table for optimization")
                with refresh_table(table_name, model):
                    _process_data_gouv_content(
                        df=df,
                        table_name=table_name,
                        checkpoint=checkpoint,
                        last_processed_index=last_processed_index,
                        model=model,
                    )
            else:
                # Resume: normal processing without refresh
                logger.info(
                    f"Resuming from checkpoint index {last_processed_index + 1}"
                )
                _process_data_gouv_content(
                    df=df,
                    table_name=table_name,
                    checkpoint=checkpoint,
                    last_processed_index=last_processed_index,
                    model=model,
                )

        else:
            logger.error(
                f"Unknown target directory '{target_dir}' for processing data.gouv.fr files."
            )
            raise ValueError(
                f"Unknown target directory '{target_dir}' for processing data.gouv.fr files."
            )


def _process_directories_content(
    directory: list,
    table_name: str,
    checkpoint: CheckpointManager,
    last_processed_index: int,
    model: str,
):
    """
    Process directory content with checkpoint support for resume capability.

    Workflow:
        1. Extracts and processes various fields such as addresses, phone numbers, types, SIRET/SIREN, URLs, emails,
           opening hours, mobile applications, social networks, additional information, people in charge, and hierarchy.
        2. Generates text chunks and embeddings for each directory entry.
        3. Inserts the processed data, including embeddings, into a table in the database.
    Args:
        directory (list): List of directory entries to process
        table_name (str): Name of the database table to insert data into
        checkpoint (CheckpointManager): Checkpoint manager for saving progress
        last_processed_index (int): Index of the last processed entry
        model (str): Model name for embedding generation
    """
    total_entries = len(directory)
    logger.info(f"Starting to process {total_entries} entries")

    ## Processing data
    for k, data in enumerate(directory):
        # Skip already processed entries
        if k <= last_processed_index:
            continue

        # Log progress every 100 entries
        if k > 0 and k % 100 == 0:
            logger.info(
                f"Processing entry {k}/{total_entries} ({(k / total_entries) * 100:.1f}%)"
            )

        chunk_id = data.get("id", "")
        name = data.get("nom", "")
        directory_url = data.get("url_service_public", "")

        # Addresses
        addresses = []
        try:
            for adresse in data.get("adresse", [{}]):
                # Metadata
                addresses.append(
                    {
                        "adresse": f"{adresse.get('complement1', '')} {adresse.get('complement2', '')} {adresse.get('numero_voie', '')}".strip(),
                        "code_postal": adresse.get("code_postal", ""),
                        "commune": adresse.get("nom_commune", ""),
                        "pays": adresse.get("pays", ""),
                        "longitude": adresse.get("longitude", ""),
                        "latitude": adresse.get("latitude", ""),
                    }
                )

        except Exception:
            pass

        # Phone numbers
        phone_numbers = []
        try:
            for telephone in data.get("telephone", [{}]):
                if telephone.get("description", ""):
                    phone_numbers.append(
                        f"{telephone.get('valeur', '')}. {telephone.get('description', '')}"
                    )
                else:
                    phone_numbers.append(f"{telephone.get('valeur', '')}")
        except Exception:
            pass

        # File modification date
        try:
            date_str = data.get("date_modification", "")
            if date_str:
                modification_date_dt = datetime.strptime(date_str, "%d/%m/%Y %H:%M:%S")
                modification_date = modification_date_dt.strftime("%Y-%m-%d")
            else:
                modification_date = ""
        except ValueError:
            modification_date = ""
            logger.debug(f"Date format error for value: {date_str}")

        # Types
        types = ""
        type_list = []
        pivot_data = data.get("pivot", [])
        type_organisme = data.get("type_organisme", "")

        if pivot_data and type_organisme:
            for pivot in pivot_data:
                if isinstance(pivot, dict) and "type_service_local" in pivot:
                    type_list.append(f"{pivot['type_service_local']}")
            types = ", ".join(type_list)
            types += f" ({type_organisme})"
        elif pivot_data and not type_organisme:
            for pivot in pivot_data:
                if isinstance(pivot, dict) and "type_service_local" in pivot:
                    type_list.append(f"{pivot['type_service_local']}")
            types = ", ".join(type_list)
        elif type_organisme and not pivot_data:
            types += f"{type_organisme}"

        # SIRET and SIREN
        siret = data.get("siret", "")
        siren = data.get("siren", "")

        # URLs
        urls = []
        try:
            for site in data.get("site_internet", [{}]):
                if isinstance(site.get("valeur", ""), list):
                    urls.extend(site.get("valeur", []))
                else:
                    urls.append(site.get("valeur", ""))
        except Exception:
            pass

        # Contact forms
        contact_forms = []
        try:
            for formulaire in data.get("formulaire_contact", []):
                if isinstance(formulaire, list):
                    contact_forms.extend(formulaire)
                else:
                    contact_forms.append(formulaire)
        except Exception:
            pass

        # Emails
        mails = []
        try:
            for mail in data.get("adresse_courriel", []):
                if isinstance(mail, list):
                    mails.extend(mail)
                else:
                    mails.append(mail)
        except Exception:
            pass

        # Opening hours
        opening_hours = _make_schedule(data.get("plage_ouverture", []))

        # Mobile applications
        mobile_applications = []
        try:
            for application in data.get("application_mobile", [{}]):
                mobile_applications.append(
                    f"{application.get('description', '')} ({application.get('custom_dico2', '')}) : {application.get('valeur', '')}"
                )
        except Exception:
            pass

        # Social medias
        social_medias = []
        try:
            for reseau in data.get("reseau_social", [{}]):
                if reseau.get("description", ""):
                    social_medias.append(
                        f"{reseau.get('custom_dico2', '')} ({reseau.get('description', '')}) : {reseau.get('valeur', '')}"
                    )
                else:
                    social_medias.append(
                        f"{reseau.get('custom_dico2', '')} : {reseau.get('valeur', '')}"
                    )
        except Exception:
            pass

        # Additional information and mission description
        additional_information = data.get("information_complementaire", "")
        mission_description = data.get("mission", "")

        # People in charge
        people_in_charge = data.get("affectation_personne", [{}])

        # Organizational chart and hierarchy
        organizational_chart = []
        try:
            for org in data.get("organigramme", []):
                if org.get("libelle"):
                    organizational_chart.append(
                        f"{org.get('libelle', '')} : {org.get('valeur', '')}"
                    )
                else:
                    organizational_chart.append(f"{org.get('valeur', '')}")
        except Exception:
            pass

        hierarchy = data.get("hierarchie", [])

        chunk_text = make_directory_text(
            nom=name,
            mission=mission_description,
            responsables=people_in_charge,
            adresses=addresses,
        )

        chunk_xxh64 = xxhash.xxh64(chunk_text.encode("utf-8"), seed=2025).hexdigest()

        embeddings = generate_embeddings_with_retry(
            data=chunk_text, attempts=5, model=model
        )[0]

        doc_id = (
            chunk_id  # Using chunk_id as doc_id because each document is a single entry
        )

        ## Insert data into the database
        new_data = (
            chunk_id,
            doc_id,
            chunk_xxh64,  # Hash of chunk_text
            types,
            name,
            mission_description,
            json.dumps(addresses),  # Converts to string
            phone_numbers,
            mails,
            urls,
            social_medias,
            mobile_applications,
            opening_hours,
            contact_forms,
            additional_information,
            modification_date,
            siret,
            siren,
            json.dumps(people_in_charge),
            organizational_chart,
            json.dumps(hierarchy),
            directory_url,
            chunk_text,
            embeddings,
        )

        try:
            insert_data(
                data=[new_data],
                table_name=table_name,
            )
            # Save checkpoint after successful insertion
            checkpoint.save(k, metadata={"doc_id": doc_id, "table": table_name})

        except Exception as e:
            logger.error(f"Error inserting data for entry {k} (doc_id: {doc_id}): {e}")
            logger.error(
                "Progress saved. Restart the process to resume from this point."
            )
            raise e

    # All entries processed successfully
    checkpoint.remove()
    logger.info(f"Successfully processed all {total_entries} directory entries")


def process_directories(table_name: str, model: str = "BAAI/bge-m3"):
    """
    Processes directory data from JSON files specified in a configuration file, extracts and transforms relevant fields,
    generates embeddings for each directory, and inserts the processed data into a database.

    Implements a checkpoint system to resume processing from the last successfully
    processed directory entry in case of errors.

    Args:
        table_name (str): The name of the table to process.
        model (str): The identifier for the embedding model to use. Defaults to "BAAI/bge-m3".
    Raises:
        FileNotFoundError: If the configuration file or any specified directory JSON file is not found.
        json.JSONDecodeError: If there is an error decoding JSON from the configuration or data files.
        Exception: For any other unexpected errors during file loading or embedding generation.

    Workflow:
        1. Loads the configuration file to determine which directory JSON files to process.
        2. Reads and aggregates directory data from the specified JSON files.
        3. Process, embeds, and inserts each directory entry into the database.

    Logging:
        Logs errors and information throughout the process, including file loading issues, JSON decoding errors,
        embedding generation retries, and the number of directories loaded.
    """
    config = load_config(config_file_path=config_file_path)
    data_sources = config.get(table_name.lower(), {})

    for data_source, attributes in data_sources.items():
        if attributes.get("type") == "directory":
            target_dir = os.path.join(BASE_PATH, attributes.get("download_folder", ""))

            # Initialiser le checkpoint manager
            json_path = f"{target_dir}/{data_source}.json"
            checkpoint = CheckpointManager(json_path)
            last_processed_index = checkpoint.load()

            ### Loading directory
            directory = []
            try:
                with open(json_path, encoding="utf-8") as json_file:
                    json_data = json.load(json_file)
                    if not directory:  # First file
                        directory = json_data["service"]
                    else:
                        directory.extend(json_data["service"])
                    logger.info(
                        f"Loaded {len(directory)} lines of data from {target_dir}, "
                        f"Starting from index: {last_processed_index + 1}"
                    )
            except FileNotFoundError:
                logger.error(f"File not found: {json_path}.")
                raise
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from the file: {json_path}.")
                raise
            except Exception as e:
                logger.error(f"Unexpected error while loading file {json_path}: {e}")
                raise

            if last_processed_index == -1:
                # First run: use refresh_table for optimization
                logger.info("First run detected, using refresh_table for optimization")
                with refresh_table(table_name, model):
                    _process_directories_content(
                        directory=directory,
                        table_name=table_name,
                        checkpoint=checkpoint,
                        last_processed_index=last_processed_index,
                        model=model,
                    )
            else:
                # Resume: normal processing without refresh
                logger.info(
                    f"Resuming from checkpoint index {last_processed_index + 1}"
                )
                _process_directories_content(
                    directory=directory,
                    table_name=table_name,
                    checkpoint=checkpoint,
                    last_processed_index=last_processed_index,
                    model=model,
                )

        else:
            logger.error(
                f"Unknown data type for source '{data_source}' in processing directories."
            )
            raise ValueError(
                f"Unknown data type for source '{data_source}' in processing directories."
            )


def _process_dila_xml_content(root: ET.Element, file_name: str, model: str):
    """Processes a single DILA XML file, prepares, and inserts its content into the database.

    This function acts as a dispatcher based on the filename prefix. It handles
    different XML structures for various legal document types (`LEGIARTI`, `CNILTEXT`,
    `CONSTEXT`, `JORFDOLE`).

    For each document, it extracts metadata and textual content, splits the text
    into manageable chunks, generates a vector embedding for each chunk, and
    then performs a batch insertion of the processed data into the corresponding
    database table.

    Args:
        root (ET.Element): The root element of the parsed XML file.
        file_name (str): The name of the XML file, used to determine the
                         processing logic.
        model (str): The identifier for the embedding model to be used.

    Raises:
        PermissionDeniedError: If the embedding generation fails due to an
                               API key issue.
        Exception: For any other errors encountered during file processing,
                   which are logged and re-raised.
    """
    if file_name.startswith("LEGIARTI") and file_name.endswith(".xml"):
        table_name = "legi"
        try:
            status = root.find(".//ETAT").text
            cid = root.find(".//ID").text  # doc_id
            nature = root.find(".//NATURE").text
            title = (
                root.find(".//CONTEXTE//TEXTE//TITRE_TXT")
                .get("c_titre_court")
                .strip(".")
            )
            category = root.find(".//CONTEXTE//TEXTE").get("nature", None)
            ministry = root.find(".//CONTEXTE//TEXTE").get("ministere", None)
            subtitles = []
            for elem in root.find(".//CONTEXTE//TEXTE").iter("TITRE_TM"):
                subtitles.append(elem.text)
            subtitles = " - ".join(subtitles)
            if not subtitles:
                subtitles = None
            number = root.find(".//NUM").text

            start_date = datetime.strptime(
                root.find(".//DATE_DEBUT").text, "%Y-%m-%d"
            ).strftime("%Y-%m-%d")
            end_date = datetime.strptime(
                root.find(".//DATE_FIN").text, "%Y-%m-%d"
            ).strftime("%Y-%m-%d")
            full_title = root.find(".//TITRE_TXT").text

            nota = []
            contenu_nota = root.find(".//NOTA//CONTENU")
            for paragraph in contenu_nota.findall(".//p"):
                nota.append(paragraph.text)
            nota = "\n".join(nota).strip()
            if not nota:
                nota = None

            links = []
            for link in root.find(".//LIENS"):
                links.append(
                    {
                        "text_doc_id": link.get("cidtexte"),
                        "text_signature_date": link.get("datesignatexte"),
                        "doc_id": link.get("id"),
                        "category": link.get("naturetexte"),
                        "nor": link.get("nortexte"),
                        "number": link.get("num"),
                        "text_number": link.get("numtexte"),
                        "link_direction": link.get("sens"),
                        "link_type": link.get("typelien"),
                        "title": link.text,
                    }
                )

            contenu = root.find(".//BLOC_TEXTUEL/CONTENU")
            text_content = []

            if contenu is not None:
                # Extract all text
                content = ET.tostring(contenu, encoding="unicode", method="xml")
                content = "".join(ET.fromstring(content).itertext())
                # Post-process the text to improve readability
                lines = content.splitlines()  # Split the text into lines
                cleaned_lines = [
                    line for line in lines if line
                ]  # Remove empty lines and extra spaces
                content = "\n".join(
                    cleaned_lines
                )  # Rejoin the cleaned lines with a newline
                text_content.append(content)
            text_content = "\n".join(text_content)

            chunks = make_chunks(
                text=text_content,
                chunk_size=1024,
                chunk_overlap=0,
                length_function="bge_m3_tokenizer",
            )
            data_to_insert = []

            for k, text in enumerate(chunks):
                try:
                    chunk_index = k + 1  # Start chunk numbering from 1
                    chunk_text = f"{full_title}"
                    if number:
                        chunk_text += f" - Article {number}"
                    # Adding subtitles only if the text is long enough
                    if subtitles and len(text) > 200:
                        context = format_subtitles(subtitles=subtitles)
                        if context and len(context) < len(text):
                            chunk_text += f"\n{context}"  # Augment the chunk text with subtitles concepts
                    chunk_text += f"\n{text}"

                    chunk_xxh64 = xxhash.xxh64(
                        chunk_text.encode("utf-8"), seed=2025
                    ).hexdigest()

                    embeddings = generate_embeddings_with_retry(
                        data=chunk_text, attempts=5, model=model
                    )[0]
                    chunk_id = f"{cid}_{chunk_index}"  # Unique ID for each chunk

                    new_data = (
                        chunk_id,  # Primary key
                        cid,  # Original document ID
                        chunk_index,  # Chunk number
                        chunk_xxh64,  # Hash of chunk_text
                        nature,
                        category,
                        ministry,
                        status,
                        title,
                        full_title,
                        subtitles,
                        number,
                        start_date,
                        end_date,
                        nota,
                        json.dumps(links),  # Convert links to JSON string
                        text,  # Original text
                        chunk_text,  # Augmented text for better search
                        embeddings,  # Embedding of chunk_text
                    )
                    data_to_insert.append(new_data)
                except PermissionDeniedError as e:
                    logger.error(
                        f"PermissionDeniedError (API key issue) for chunk {chunk_index} of file {file_name}: {e}"
                    )
                    raise e

            # Inserting all chunks at once
            if data_to_insert:
                insert_data(data=data_to_insert, table_name=table_name)

        except Exception as e:
            logger.error(f"Error processing file {file_name}: {e}")
            raise e

    elif file_name.startswith("CNILTEXT") and file_name.endswith(".xml"):
        table_name = "cnil"
        try:
            status = root.find(".//ETAT_JURIDIQUE").text

            cid = root.find(".//ID").text
            nature = root.find(".//NATURE").text
            nature_delib = root.find(".//NATURE_DELIB").text
            title = root.find(".//TITRE").text
            full_title = root.find(".//TITREFULL").text
            number = root.find(".//NUMERO").text
            date = datetime.strptime(
                root.find(".//DATE_TEXTE").text, "%Y-%m-%d"
            ).strftime("%Y-%m-%d")

            contenu = root.find(".//BLOC_TEXTUEL/CONTENU")
            text_content = []

            if contenu is not None:
                # Extract all text
                content = ET.tostring(contenu, encoding="unicode", method="xml")
                content = "".join(ET.fromstring(content).itertext())
                # Post-process the text to improve readability
                lines = content.splitlines()  # Split the content into lines
                cleaned_lines = [
                    line for line in lines if line
                ]  # Remove empty lines and extra spaces

                content = "\n".join(
                    cleaned_lines
                )  # Rejoin the cleaned lines with a newline
                text_content.append(content)
            text_content = "\n".join(text_content)

            chunks = make_chunks(
                text=text_content,
                chunk_size=1500,
                chunk_overlap=0,
                length_function="len",
            )
            data_to_insert = []

            for k, text in enumerate(chunks):
                try:
                    chunk_index = k + 1  # Start chunk numbering from 1
                    chunk_text = f"{title}\n{text}"

                    chunk_xxh64 = xxhash.xxh64(
                        chunk_text.encode("utf-8"), seed=2025
                    ).hexdigest()

                    embeddings = generate_embeddings_with_retry(
                        data=chunk_text, attempts=5, model=model
                    )[0]

                    chunk_id = f"{cid}_{chunk_index}"  # Unique ID for each chunk

                    new_data = (
                        chunk_id,  # Primary key
                        cid,  # Original document ID
                        chunk_index,  # Chunk number
                        chunk_xxh64,  # Hash of chunk_text
                        nature,
                        status,
                        nature_delib,
                        title,
                        full_title,
                        number,
                        date,
                        text,  # Original text
                        chunk_text,
                        embeddings,
                    )
                    data_to_insert.append(new_data)
                except PermissionDeniedError as e:
                    logger.error(
                        f"PermissionDeniedError (API key issue) for chunk {chunk_index} of file {file_name}: {e}"
                    )
                    raise

            # Inserting all chunks at once
            if data_to_insert:
                insert_data(data=data_to_insert, table_name=table_name)

        except Exception as e:
            logger.error(f"Error processing file {file_name}: {e}")
            raise e

    elif file_name.startswith("CONSTEXT") and file_name.endswith(".xml"):
        table_name = "constit"
        try:
            cid = root.find(".//ID").text
            nature = root.find(".//NATURE").text
            title = root.find(".//TITRE").text
            number = root.find(".//NUMERO").text
            solution = root.find(".//SOLUTION").text
            decision_date = datetime.strptime(
                root.find(".//DATE_DEC").text, "%Y-%m-%d"
            ).strftime("%Y-%m-%d")
            contenu = root.find(".//BLOC_TEXTUEL//CONTENU")

            text_content = []

            if contenu is not None:
                # Extract all text
                content = ET.tostring(contenu, encoding="unicode", method="xml")
                content = "".join(ET.fromstring(content).itertext())
                # Post-process the content to improve readability
                lines = content.splitlines()  # Split the content into lines
                cleaned_lines = [
                    line for line in lines if line
                ]  # Remove empty lines and extra spaces
                content = "\n".join(
                    cleaned_lines
                )  # Rejoin the cleaned lines with a newline
                text_content.append(content)
            text_content = "\n".join(text_content)

            chunks = make_chunks(
                text=text_content,
                chunk_size=1500,
                chunk_overlap=0,
                length_function="len",
            )
            data_to_insert = []

            for k, text in enumerate(chunks):
                try:
                    chunk_index = k + 1  # Start chunk numbering from 1
                    chunk_text = f"{title}\n{text}"

                    chunk_xxh64 = xxhash.xxh64(
                        chunk_text.encode("utf-8"), seed=2025
                    ).hexdigest()

                    embeddings = generate_embeddings_with_retry(
                        data=chunk_text, attempts=5, model=model
                    )[0]

                    chunk_id = f"{cid}_{chunk_index}"  # Unique ID for each chunk

                    new_data = (
                        chunk_id,  # Primary key
                        cid,  # Original document ID
                        chunk_index,  # Chunk number
                        chunk_xxh64,  # Hash of chunk_text
                        nature,
                        solution,
                        title,
                        number,
                        decision_date,
                        text,  # Original text
                        chunk_text,
                        embeddings,
                    )
                    data_to_insert.append(new_data)
                except PermissionDeniedError as e:
                    logger.error(
                        f"PermissionDeniedError (API key issue) for chunk {chunk_index} of file {file_name}: {e}"
                    )
                    raise e

            # Inserting all chunks at once
            if data_to_insert:
                insert_data(data=data_to_insert, table_name=table_name)

        except Exception as e:
            logger.error(f"Error processing file {file_name}: {e}")
            raise e

    elif file_name.startswith("JORFDOLE") and file_name.endswith(".xml"):
        table_name = "dole"
        try:
            cid = root.find(".//ID").text  # doc_id
            title = root.find(".//TITRE").text
            number = root.find(".//NUMERO").text
            category = root.find(".//TYPE").text
            wording = root.find(".//LIBELLE").text  # Libellé
            creation_date = datetime.strptime(
                root.find(".//DATE_CREATION").text, "%Y-%m-%d"
            ).strftime("%Y-%m-%d")

            exp_memo = root.find(
                ".//EXPOSE_MOTIF"
            )  # Explanatory Memorandum (Exposé des motifs)

            if exp_memo:
                # Extract all text
                content = ET.tostring(exp_memo, method="xml")
                content = "".join(ET.fromstring(content).itertext())
                exp_memo = _dole_cut_exp_memo(text=content, section="introduction")
                articles_synthesis_dict = _dole_cut_exp_memo(
                    text=content, section="articles"
                )
            else:
                exp_memo = None
                articles_synthesis_dict = []

            # Creating chunks for explanatory memorandum
            chunks = make_chunks(
                text=exp_memo, chunk_size=8000, chunk_overlap=0, length_function="len"
            )
            data_to_insert = []
            if not chunks:
                chunk_text = title
                try:
                    embeddings = generate_embeddings_with_retry(
                        data=chunk_text, attempts=5, model="BAAI/bge-m3"
                    )[0]
                    chunk_index = 1  # Since there is only one chunk
                    content_type = "explanatory_memorandum"
                    chunk_id = f"{cid}_{chunk_index}"
                    chunk_xxh64 = xxhash.xxh64(
                        chunk_text.encode("utf-8"), seed=2025
                    ).hexdigest()
                    new_data = (
                        chunk_id,
                        cid,  # doc_id
                        chunk_index,
                        chunk_xxh64,  # Hash of chunk_text
                        category,
                        content_type,
                        title,
                        number if number else None,
                        wording,
                        creation_date,
                        None,  # article_number
                        None,  # article_title
                        None,  # article_synthesis
                        None,  # text
                        chunk_text,
                        embeddings,
                    )
                    data_to_insert.append(new_data)

                except PermissionDeniedError as e:
                    logger.error(
                        f"PermissionDeniedError (API key issue) for chunk {chunk_index} of file {file_name}: {e}"
                    )
                    raise
            else:
                for k, text in enumerate(chunks):
                    try:
                        chunk_index = k + 1  # Start chunk numbering from 1
                        chunk_text = (title + "\n" + text).replace(
                            "\n\n", "\n"
                        )  # Adding the title to the chunk text
                        embeddings = generate_embeddings_with_retry(
                            data=chunk_text,
                            attempts=5,
                            model="BAAI/bge-m3",
                        )[0]
                        content_type = "explanatory_memorandum"
                        chunk_id = f"{cid}_{chunk_index}"

                        chunk_xxh64 = xxhash.xxh64(
                            chunk_text.encode("utf-8"), seed=2025
                        ).hexdigest()

                        new_data = (
                            chunk_id,
                            cid,  # doc_id
                            chunk_index,
                            chunk_xxh64,  # Hash of chunk_text
                            category,
                            content_type,
                            title,
                            number if number else None,
                            wording,
                            creation_date,
                            None,  # article_number
                            None,  # article_title
                            None,  # article_synthesis
                            text,
                            chunk_text,
                            embeddings,
                        )
                        data_to_insert.append(new_data)

                    except PermissionDeniedError as e:
                        logger.error(
                            f"PermissionDeniedError (API key issue) for chunk {chunk_index} of file {file_name}: {e}"
                        )
                        raise e

            file_content_list = []
            for k in range(1, 6):  # There can be up to 5 contenu_dossier sections
                contenu_dossier = root.find(f".//CONTENU_DOSSIER_{k}")
                if contenu_dossier is not None:
                    # Extract all text
                    content = ET.tostring(contenu_dossier, method="xml")
                    content = "".join(ET.fromstring(content).itertext()).strip()

                    if len(content) > 0:
                        file_content_list.extend(_dole_cut_file_content(text=content))

            results = []
            if len(file_content_list) == 0 and len(articles_synthesis_dict) == 0:
                results = [
                    {
                        "article_number": None,
                        "article_synthesis": None,
                        "article_text": None,
                        "article_title": None,
                    }
                ]
            elif len(articles_synthesis_dict) > 0 and len(file_content_list) == 0:
                for article in articles_synthesis_dict:
                    results.append(
                        {
                            "article_number": article.get("article_number", None),
                            "article_synthesis": article.get("article_synthesis", None),
                            "article_text": None,  # Because there is no file content
                            "article_title": article.get("title_content", None),
                        }
                    )

            elif len(articles_synthesis_dict) == 0 and len(file_content_list) > 0:
                for content in file_content_list:
                    results.append(
                        {
                            "article_number": content.get("article_number", None),
                            "article_synthesis": None,  # Because there is no article synthesis
                            "article_text": content.get("article_text", None),
                            "article_title": None,  # Because there is no article synthesis
                        }
                    )

            else:  # Both articles_synthesis_dict and file_content_list are not empty
                # Merging articles_synthesis_dict and file_content_list by article_number
                d1 = {
                    d["article_number"]: d
                    for d in articles_synthesis_dict
                    if d["article_number"] is not None
                }
                d2 = {
                    d["article_number"]: d
                    for d in file_content_list
                    if d["article_number"] is not None
                }

                for num in set(d1) | set(d2):
                    try:
                        merged = {
                            "article_number": num,
                            "article_synthesis": d1.get(num, {})
                            .get("article_synthesis", None)
                            .strip()
                            if d1.get(num, {}).get("article_synthesis")
                            else None,
                            "article_text": d2.get(num, {})
                            .get("article_text", None)
                            .strip()
                            if d2.get(num, {}).get("article_text")
                            else None,
                            "article_title": d1.get(num, {})
                            .get("title_content", None)
                            .strip()
                            if d1.get(num, {}).get("title_content")
                            else None,
                        }
                        results.append(merged)
                    except Exception as e:
                        logger.error(
                            f"Error merging data for article number {num}: {e}"
                        )
                        raise e

                # Adding all articles with article_number = None
                for d in articles_synthesis_dict:
                    if d["article_number"] is None:
                        merged = {
                            "article_number": None,
                            "article_synthesis": d.get("article_synthesis").strip()
                            if d.get("article_synthesis")
                            else None,
                            "article_text": None,
                            "article_title": d.get("title_content").strip()
                            if d.get("title_content")
                            else None,
                        }
                        results.append(merged)

                for d in file_content_list:
                    if d["article_number"] is None:
                        merged = {
                            "article_number": None,
                            "article_synthesis": None,
                            "article_text": d["article_text"].strip()
                            if d.get("article_text")
                            else None,
                            "article_title": None,
                        }
                        results.append(merged)

            for result_number, result in enumerate(results):
                if (
                    result.get("article_number") is not None
                ):  # The chunks will be created and chunked by article number
                    content_type = "article"
                    chunks = [
                        str(result.get("article_synthesis", ""))
                        if result.get("article_synthesis") is not None
                        else "",
                    ]
                    if result.get("article_text"):
                        article_text = result.get("article_text")
                        if article_text is not None:
                            chunks.append(str(article_text).strip())
                    chunk_text = (
                        "\n".join(chunks).replace("\n\n", "\n").strip()
                    )  # Combining article synthesis and text
                    chunks = make_chunks(
                        text=chunk_text,
                        chunk_size=8000,
                        chunk_overlap=0,
                        length_function="len",
                    )

                    for k, text in enumerate(chunks):
                        chunk_index = k + 1  # Start chunk numbering from 1
                        chunk_id = f"{cid}_{chunk_index}"  # Unique ID for each chunk

                        if (
                            chunk_index == 1
                        ):  # Because the first chunk always contains the article number
                            chunk_text = f"{title}\n{text}"
                        else:
                            if result.get("article_number", ""):
                                chunk_text = f"{title}\nArticle {result.get('article_number', '')}:\n{text}"  # Adding the chunk number to remind which article number the chunk is related to
                            else:
                                chunk_text = f"{title}\n{text}"
                        try:
                            chunk_xxh64 = xxhash.xxh64(
                                chunk_text.encode("utf-8"), seed=2025
                            ).hexdigest()

                            embeddings = generate_embeddings_with_retry(
                                data=chunk_text,
                                attempts=5,
                                model="BAAI/bge-m3",
                            )[0]

                            new_data = (
                                chunk_id,
                                cid,  # doc_id
                                chunk_index,
                                chunk_xxh64,  # Hash of chunk_text
                                category,
                                content_type,
                                title,
                                number if number else None,
                                wording,
                                creation_date,
                                result.get("article_number", None),
                                result.get("article_title", None),
                                result.get("article_synthesis", None),
                                text,
                                chunk_text,
                                embeddings,
                            )
                            data_to_insert.append(new_data)

                        except PermissionDeniedError as e:
                            logger.error(
                                f"PermissionDeniedError (API key issue) for chunk {chunk_index} of file {file_name}: {e}"
                            )
                            raise

                else:  # The chunks will be created by classic chunking
                    chunk_index = result_number + 1
                    chunk_id = f"{cid}_{chunk_index}"  # Unique ID for each chunk
                    content_type = "dossier_content"
                    chunks = []  # As it is impossible to have an article synthesis without an article number

                    if result.get("article_text", ""):
                        chunks.append(str(result.get("article_text")).strip())
                    chunks = "\n".join(chunks).strip()

                    chunks = make_chunks(
                        text=chunks,
                        chunk_size=8000,
                        chunk_overlap=0,
                        length_function="len",
                    )

                    for i, text in enumerate(chunks):
                        try:
                            chunk_text = (title + "\n" + text).replace(
                                "\n\n", "\n"
                            )  # Adding the title to the chunk text

                            chunk_xxh64 = xxhash.xxh64(
                                chunk_text.encode("utf-8"), seed=2025
                            ).hexdigest()

                            embeddings = generate_embeddings_with_retry(
                                data=chunk_text,
                                attempts=5,
                                model="BAAI/bge-m3",
                            )[0]

                            new_data = (
                                chunk_id,
                                cid,  # doc_id
                                chunk_index,
                                chunk_xxh64,  # Hash of chunk_text
                                category,
                                content_type,
                                title,
                                number if number else None,
                                wording,
                                creation_date,
                                result.get("article_number", None),
                                result.get("article_title", None),
                                result.get("article_synthesis", None),
                                text,
                                chunk_text,
                                embeddings,
                            )
                            data_to_insert.append(new_data)

                        except PermissionDeniedError as e:
                            logger.error(
                                f"PermissionDeniedError (API key issue) for chunk {chunk_index} of file {file_name}: {e}"
                            )
                            raise

            # Insert all chunks at once
            if data_to_insert:
                insert_data(data=data_to_insert, table_name=table_name)

        except Exception as e:
            logger.error(f"Error processing file {file_name}: {e}")
            raise e


def _handle_dila_suppression_list(lines: list[str], table_name: str, source_name: str):
    """
    Processes a suppression list from DILA's files, to remove documents from a database table.
    For each line provided, it extracts a document ID from the end of the path-like
    string and calls a function to remove the corresponding data from the specified
    table.

    Args:
        lines (list[str]): A list of strings from the suppression file.
        table_name (str): The name of the database table to modify.
        source_name (str): The name of the source file for logging purposes.
    """
    try:
        doc_ids_to_remove = [
            line.strip().split("/")[-1] for line in lines if line.strip()
        ]
        if doc_ids_to_remove:
            logger.debug(
                f"Removing {len(doc_ids_to_remove)} document IDs from the '{table_name.upper()}' table based on suppression list in {source_name}"
            )
            for doc_id in doc_ids_to_remove:
                remove_data(table_name=table_name, column="doc_id", value=doc_id)
    except Exception as e:
        logger.error(
            f"Error removing document IDs from suppression list for {source_name}: {e}"
        )
        raise Exception(
            f"Error removing document IDs from suppression list for {source_name}: {e}"
        )


def process_dila_xml_files(
    source_path: str, streaming: bool = True, model: str = "BAAI/bge-m3"
):
    """Processes DILA XML files from a directory or a compressed archive.

    This function operates in two modes based on the `streaming` flag.
    - If `streaming` is True (default), it treats `source_path` as a .tar.gz
      archive, processing XML files in memory without extraction. The archive
      is deleted after processing is complete.
    - If `streaming` is False, it treats `source_path` as a directory,
      iterating through XML files on disk. Each file is deleted after being
      processed.

    The function implements a checkpoint system to resume processing from the last
    successfully processed file in case of errors.

    Args:
        source_path (str): The path to the source data. This is a path to a
            `.tar.gz` archive if streaming, or a directory if not.
        streaming (bool, optional): Determines the processing mode.
            Defaults to True.
        model (str, optional): The identifier for the embedding model to be
            used in the underlying processing. Defaults to "BAAI/bge-m3".
    """
    checkpoint = CheckpointManager(source_path)

    if streaming:
        logger.info(f"Processing files directly from archive: {source_path}")

        # Load checkpoint to resume from last position
        last_processed_index = checkpoint.load()

        try:
            with tarfile.open(source_path, "r:gz") as tar:
                files = [
                    m
                    for m in tar.getmembers()
                    if m.isfile() and m.name.endswith(".xml")
                ]

                total_files = len(files)
                logger.info(
                    f"Total files to process: {total_files}, Starting from index: {last_processed_index + 1}"
                )

                batch_size = 50
                for i in range(0, len(files), batch_size):
                    batch_files = files[i : i + batch_size]
                    batch_num = i // batch_size + 1
                    total_batches = (len(files) - 1) // batch_size + 1
                    if batch_num % 10 == 0 or batch_num == total_batches:
                        logger.info(
                            f"Processing batch {batch_num}/{total_batches} of {os.path.basename(source_path)} ({len(batch_files)} files)"
                        )

                    for idx, file in enumerate(batch_files):
                        file_global_index = i + idx

                        # Skip already processed files
                        if file_global_index <= last_processed_index:
                            logger.debug(
                                f"Skipping already processed file index {file_global_index}"
                            )
                            continue

                        file_object = None
                        file_content = None
                        root = None

                        try:
                            file_name = os.path.basename(file.name)
                            # Reading file in memory
                            file_object = tar.extractfile(file)
                            if file_object:
                                with file_object as f:
                                    file_content = f.read()
                            root = ET.fromstring(file_content)
                            _process_dila_xml_content(
                                root=root, file_name=file_name, model=model
                            )

                            # Save checkpoint after successful processing
                            checkpoint.save(
                                file_global_index,
                                metadata={"file_name": file_name, "type": "dila_xml"},
                            )

                        except ET.ParseError as e:
                            logger.error(
                                f"XML parsing error for file {file_name} (index {file_global_index}): {e}"
                            )
                            # Continue to next file for parse errors
                            checkpoint.save(
                                file_global_index,
                                metadata={
                                    "file_name": file_name,
                                    "error": "parse_error",
                                },
                            )
                            continue
                        except Exception as e:
                            logger.error(
                                f"Error processing file {file.name} in archive (index {file_global_index}): {e}"
                            )
                            logger.error(
                                "Progress saved. Restart the process to resume from this point."
                            )
                            raise e

                    gc.collect()

                # All files processed successfully, remove checkpoint
                checkpoint.remove()
                logger.info(
                    f"Successfully processed all {total_files} files from {source_path}"
                )

        except Exception as e:
            logger.error(f"Error processing archive {source_path}: {e}")
            raise e
        finally:
            # Only remove archive if all files were processed successfully
            if not checkpoint.exists():
                remove_file(file_path=source_path)
                logger.info(f"Archive removed: {source_path}")
            else:
                logger.info(f"Archive kept for resume: {source_path}")
            gc.collect()
    else:
        logger.info(f"Processing files from directory: {source_path}")

        # Load checkpoint for non-streaming mode
        last_processed_index = checkpoint.load()
        processed_count = 0

        for root_dir, dirs, files in os.walk(source_path):
            xml_files = [f for f in files if f.endswith(".xml")]

            for idx, file_name in enumerate(xml_files):
                # Skip already processed files
                if processed_count <= last_processed_index:
                    processed_count += 1
                    continue

                file_path = os.path.join(root_dir, file_name)
                try:
                    tree = ET.parse(file_path)
                    root = tree.getroot()
                    _process_dila_xml_content(
                        root=root, file_name=file_name, model=model
                    )

                    # Save checkpoint after successful processing
                    checkpoint.save(
                        processed_count,
                        metadata={"file_name": file_name, "type": "dila_xml"},
                    )
                    processed_count += 1

                except Exception as e:
                    logger.error(
                        f"Error processing file {file_name} (index {processed_count}): {e}"
                    )
                    logger.error(
                        "Progress saved. Restart the process to resume from this point."
                    )
                    raise e
                finally:
                    remove_file(file_path=file_path)  # Remove the file after processing
                    gc.collect()

        # All files processed successfully, remove checkpoint
        checkpoint.remove()
        logger.info(f"Successfully processed all files from {source_path}")


def _process_sheets_content(
    table_name: str,
    corpus_handler: CorpusHandler,
    checkpoint: CheckpointManager,
    last_processed_index: int,
    batch_size: int,
    model: str,
):
    """
    Process a batch of sheets data with checkpoint support for resume capability.
    Args:
        table_name (str): Name of the database table to insert data into
        corpus_handler (CorpusHandler): Handler for iterating over documents and embeddings
        checkpoint (CheckpointManager): Checkpoint manager for saving progress
        last_processed_index (int): Index of the last processed document
        batch_size (int): Number of documents to process per batch
        model (str): Model name for embedding generation
    """
    processed_count = 0

    if table_name == "travail_emploi":
        for (
            batch_documents,
            batch_embeddings,
        ) in corpus_handler.iter_docs_embeddings(
            batch_size=batch_size,
            model=model,
        ):
            data_to_insert = []

            for document, embeddings in zip(batch_documents, batch_embeddings):
                # Skip already processed documents
                if processed_count <= last_processed_index:
                    processed_count += 1
                    continue
                doc_id = document["sid"]
                chunk_index = document["chunk_index"]
                chunk_id = f"{doc_id}_{chunk_index}"
                chunk_xxh64 = document["chunk_xxh64"]  # Hash of chunk_text
                title = document["title"]
                surtitle = document["surtitle"]
                source = document["source"]
                introduction = document["introduction"]
                date = document["date"]
                url = document["url"]
                context = document["context"] if "context" in document else []
                text = document["text"]
                chunk_text = document["chunk_text"]

                new_data = (
                    chunk_id,
                    doc_id,
                    chunk_index,
                    chunk_xxh64,
                    title,
                    surtitle,
                    source,
                    introduction,
                    date,
                    url,
                    context,
                    text,
                    chunk_text,
                    embeddings,
                )
                data_to_insert.append(new_data)
                processed_count += 1

            if data_to_insert:
                try:
                    insert_data(data=data_to_insert, table_name=table_name)
                    # Save checkpoint after successful batch insertion
                    checkpoint.save(
                        processed_count - 1,
                        metadata={
                            "table": table_name,
                            "batch_size": len(data_to_insert),
                        },
                    )
                except Exception as e:
                    logger.error(
                        f"Error inserting batch at index {processed_count}: {e}"
                    )
                    logger.error(
                        "Progress saved. Restart the process to resume from this point."
                    )
                    raise e

    elif table_name == "service_public":
        for (
            batch_documents,
            batch_embeddings,
        ) in corpus_handler.iter_docs_embeddings(batch_size):
            data_to_insert = []

            for document, embeddings in zip(batch_documents, batch_embeddings):
                # Skip already processed documents
                if processed_count <= last_processed_index:
                    processed_count += 1
                    continue
                doc_id = document["sid"]
                chunk_index = document["chunk_index"]
                chunk_id = f"{doc_id}_{chunk_index}"
                chunk_xxh64 = document["chunk_xxh64"]  # Hash of chunk_text
                audience = document["audience"]
                theme = document["theme"]
                title = document["title"]
                surtitle = document["surtitle"]
                source = document["source"]
                introduction = document["introduction"]
                url = document["url"]
                related_questions = document["related_questions"]
                web_services = document["web_services"]
                context = document["context"] if "context" in document else ""
                text = document["text"]
                chunk_text = document["chunk_text"]

                new_data = (
                    chunk_id,
                    doc_id,
                    chunk_index,
                    chunk_xxh64,
                    audience,
                    theme,
                    title,
                    surtitle,
                    source,
                    introduction,
                    url,
                    json.dumps(related_questions),
                    json.dumps(web_services),
                    context,
                    text,
                    chunk_text,
                    embeddings,
                )
                data_to_insert.append(new_data)
                processed_count += 1

            if data_to_insert:
                try:
                    insert_data(data=data_to_insert, table_name=table_name)
                    # Save checkpoint after successful batch insertion
                    checkpoint.save(
                        processed_count - 1,
                        metadata={
                            "table": table_name,
                            "batch_size": len(data_to_insert),
                        },
                    )
                except Exception as e:
                    logger.error(
                        f"Error inserting batch at index {processed_count}: {e}"
                    )
                    logger.error(
                        "Progress saved. Restart the process to resume from this point."
                    )
                    raise e

    else:
        logger.error(f"Unknown table name '{table_name}' for sheets processing")
        raise ValueError(f"Unknown table name '{table_name}' for sheets processing")


def process_sheets(
    table_name: str,
    model: str = "BAAI/bge-m3",
    batch_size: int = 10,
):
    """
    Process sheets data with checkpoint support for resume capability.

    Args:
        table_name (str): Name of the database table to insert data into
        model (str): Model name for embedding generation
        batch_size (int): Number of documents to process per batch
    """

    config = load_config(config_file_path=config_file_path)
    data_sources = config.get(table_name.lower(), {})

    for data_source_index, (data_source, attributes) in enumerate(data_sources.items()):
        target_dir = os.path.join(BASE_PATH, attributes.get("download_folder", ""))
        make_chunks_sheets(
            storage_dir=target_dir,
            structured=True,
            chunk_size=1024,
            chunk_overlap=0,
            length_function="bge_m3_tokenizer",
        )
        json_path = os.path.join(target_dir, "sheets_as_chunks.json")
        checkpoint = CheckpointManager(source_path=json_path)
        last_processed_index = checkpoint.load()

        with open(json_path, encoding="utf-8") as f:
            documents = json.load(f)

        total_documents = len(documents)
        logger.info(
            f"Total documents to process: {total_documents}, "
            f"Starting from index: {last_processed_index + 1}"
        )

        corpus_name = target_dir.split("/")[-1]
        corpus_handler = CorpusHandler.create_handler(corpus_name, documents)

        if last_processed_index == -1 and data_source_index == 0:
            logger.info("Starting processing from the beginning. Refreshing the table.")

            with refresh_table(table_name, model):
                _process_sheets_content(
                    table_name=table_name,
                    corpus_handler=corpus_handler,
                    checkpoint=checkpoint,
                    last_processed_index=last_processed_index,
                    batch_size=batch_size,
                    model=model,
                )
        else:
            logger.info(f"Starting from checkpoint index {last_processed_index + 1}")
            _process_sheets_content(
                table_name=table_name,
                corpus_handler=corpus_handler,
                checkpoint=checkpoint,
                last_processed_index=last_processed_index,
                batch_size=batch_size,
                model=model,
            )

        checkpoint.remove()
        logger.info(
            f"Successfully processed all {total_documents} documents for {table_name}"
        )


def process_data(table_name: str, streaming: bool = True, model: str = "BAAI/bge-m3"):
    """
    Processes data files located in the specified base folder according to its type.
    Depending on the value of `base_folder`, this function performs several operations.

    Args:
        table_name (str): The name of the table to process.
        streaming (bool, optional): If True, processes DILA archive files in streaming mode, without extraction (default: True).
        If False, extracts the archive files before processing.
        model (str, optional): The model to use for processing (default: "BAAI/bge-m3").
    """
    config = load_config(config_file_path=config_file_path)
    data_sources = config.get(table_name.lower(), {})
    for data_source_index, (data_source, attributes) in enumerate(data_sources.items()):
        base_folder = os.path.join(BASE_PATH, attributes.get("download_folder", ""))

        if attributes.get("type") == "directory":
            logger.info(f"Processing directory files located in : {base_folder}")
            process_directories(
                table_name=table_name,
                model=model,
            )

            logger.info(
                logger.info(
                    f"Folder: {base_folder} successfully processed and data successfully inserted into the postgres database"
                )
            )

            remove_folder(folder_path=base_folder)
            logger.debug(f"Folder: {base_folder} successfully removed after processing")
        elif attributes.get("type") == "data_gouv":
            logger.info(f"Processing files located in : {base_folder}")

            process_data_gouv_files(table_name=table_name, model=model)

            logger.info(
                f"Folder: {base_folder} successfully processed and data successfully inserted into the postgres database"
            )

            remove_folder(folder_path=base_folder)
            logger.debug(f"Folder: {base_folder} successfully removed after processing")
        elif attributes.get("type") == "sheets":
            if (
                data_source_index == 0
            ):  # To start process only once, as there can be multiple data sources for sheets type
                logger.info(f"Processing files located in : {base_folder}")

                process_sheets(
                    table_name=table_name,
                    model=model,
                )

                logger.info(
                    f"Folder: {base_folder} successfully processed and data successfully inserted into the postgres database"
                )

                remove_folder(folder_path=base_folder)
                logger.debug(
                    f"Folder: {base_folder} successfully removed after processing"
                )
            else:
                pass

        elif attributes.get("type") == "dila_folder":
            if streaming:
                all_entities = sorted(
                    [f for f in os.listdir(base_folder) if f.endswith(".tar.gz")]
                )
                # Placing the freemium file at the beginning
                try:
                    freemium_file = next(
                        (
                            file
                            for file in all_entities
                            if file.lower().startswith("freemium")
                        ),
                        None,
                    )
                    all_entities.remove(freemium_file)
                    all_entities.insert(0, freemium_file)
                except ValueError:
                    logger.debug(f"There is no freemium file in {all_entities}")
                all_entities = [os.path.join(base_folder, f) for f in all_entities]

                for entity in (
                    all_entities
                ):  # entity is the name of each tar.gz file inside the base_folder
                    # Remove obscolete CIDs from the table based on the suppression list file
                    try:
                        with tarfile.open(entity, "r:gz") as tar:
                            for member in tar.getmembers():
                                if member.isfile() and os.path.basename(
                                    member.name
                                ).startswith("liste_suppression"):
                                    file_object = tar.extractfile(member)

                                    if file_object:
                                        with file_object as f:
                                            lines = (
                                                f.read().decode("utf-8").splitlines()
                                            )
                                        _handle_dila_suppression_list(
                                            lines=lines,
                                            table_name=table_name,
                                            source_name=entity,
                                        )
                                        break  # As we found the suppression list, no need to continue
                    except Exception as e:
                        logger.error(
                            f"Error while finding suppression list from archive {entity}: {e}"
                        )
                        continue

                    # Process the XML files in the archive file
                    process_dila_xml_files(
                        source_path=entity, streaming=streaming, model=model
                    )
                    logger.info(f"File: {entity} successfully processed")

            else:
                with os.scandir(base_folder) as it:
                    all_entities = sorted(
                        [entry.name for entry in it if entry.is_dir()]
                    )  # List of all folders inside the base_folder

                # Placing the {table_name} folder at the beginning which corresponds to the freemium exctraction (e.g. 'dole' for DOLE_DATA_FOLDER)
                try:
                    all_entities.remove(table_name)
                    all_entities.insert(0, table_name)
                except ValueError:
                    logger.debug(
                        f"There is no '{table_name}' directory in {base_folder}"
                    )

                for root_dir in (
                    all_entities
                ):  # root_dir is the name of each folder inside the base_folder
                    # Remove obscolete CIDs from the table based on the suppression list file
                    current_dir = os.path.join(base_folder, root_dir)
                    for entity in os.listdir(current_dir):
                        if entity.startswith("liste_suppression"):
                            try:
                                # doc_id_to_remove = []
                                with open(os.path.join(current_dir, entity)) as f:
                                    lines = f.readlines()
                                    _handle_dila_suppression_list(
                                        lines=lines,
                                        table_name=table_name,
                                        source_name=entity,
                                    )
                            except Exception as e:
                                logger.error(
                                    f"Error removing document IDs based on suppression list: {e}"
                                )
                                raise Exception(
                                    f"Error removing document IDs based on suppression list: {e}"
                                )

                    target_dir = os.path.join(base_folder, root_dir)

                    logger.info(f"Processing folder: {target_dir}")

                    process_dila_xml_files(
                        source_path=target_dir, streaming=streaming, model=model
                    )
                    logger.info(
                        f"Folder: {target_dir} successfully processed and data successfully inserted into the database"
                    )

                    remove_folder(folder_path=current_dir)
                    logger.debug(
                        f"Folder: {current_dir} successfully removed after processing"
                    )
        else:
            logger.error(f"Unknown base folder '{base_folder}' for processing data.")
            raise ValueError(
                f"Unknown base folder '{base_folder}' for processing data."
            )


def process_all_data(
    source_map: str = SOURCE_MAP, model: str = "BAAI/bge-m3", streaming: bool = True
):
    """
    Processes all data tables defined in the source map.

    Args:
        source_map (str): Mapping of data sources to be processed.
        model (str): The model to use for processing.
        streaming (bool): Whether to process DILA files in streaming mode.

    Note:
        This function iterates over each table name in the provided source map
        and calls the `process_data` function to handle the processing of each table.
    """
    for table_name in source_map:
        process_data(table_name=table_name, model=model, streaming=streaming)
