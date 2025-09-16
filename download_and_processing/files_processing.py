import os
import xml.etree.ElementTree as ET
import pandas as pd
import json
import psycopg2
from datetime import datetime
from openai import PermissionDeniedError
from tqdm import tqdm
import xxhash

from database import insert_data, remove_data, sync_obsolete_doc_ids
from config import (
    get_logger,
    CNIL_DATA_FOLDER,
    LEGI_DATA_FOLDER,
    CONSTIT_DATA_FOLDER,
    LOCAL_ADMINISTRATIONS_DIRECTORY_FOLDER,
    STATE_ADMINISTRATIONS_DIRECTORY_FOLDER,
    TRAVAIL_EMPLOI_DATA_FOLDER,
    SERVICE_PUBLIC_PRO_DATA_FOLDER,
    SERVICE_PUBLIC_PART_DATA_FOLDER,
    DATA_GOUV_DATASETS_CATALOG_DATA_FOLDER,
    DOLE_DATA_FOLDER,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    POSTGRES_HOST,
    POSTGRES_PORT,
)
from utils import (
    CorpusHandler,
    generate_embeddings_with_retry,
    make_chunks,
    make_chunks_directories,
    make_chunks_sheets,
    dole_cut_file_content,
    dole_cut_exp_memo,
    remove_folder,
    remove_file,
    make_schedule,
    format_subtitles,
)


logger = get_logger(__name__)


def process_data_gouv_files(target_dir: str, model: str = "BAAI/bge-m3"):
    """
    Process data.gouv.fr files by generating embeddings and storing them in database.
    The workflow depends on the file.

    Args:
        target_dir (str): Directory containing the files to process
        model (str): Model name for embedding generation. Defaults to "BAAI/bge-m3"

    """
    if DATA_GOUV_DATASETS_CATALOG_DATA_FOLDER.endswith(target_dir):
        table_name = "data_gouv_datasets_catalog"
        df = pd.read_csv(f"{target_dir}/{table_name}.csv", sep=";", encoding="utf-8")

        conn = None
        try:
            conn = psycopg2.connect(
                host=POSTGRES_HOST,
                port=POSTGRES_PORT,
                dbname=POSTGRES_DB,
                user=POSTGRES_USER,
                password=POSTGRES_PASSWORD,
            )
            cursor = conn.cursor()

            logger.info(
                f"Fetching existing document ids from table {table_name.upper()}..."
            )
            cursor.execute(f"SELECT DISTINCT doc_id FROM {table_name.upper()};")

            all_old_doc_ids = {row[0] for row in cursor.fetchall()}

        except Exception as e:
            logger.error(f"Error connecting to the database: {e}")
            return

        all_new_doc_ids = []
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

        for _, row in tqdm(
            df.iterrows(), desc=f"Processing {table_name}", total=len(df)
        ):
            # Replace nan values with None in the current row
            row = row.where(pd.notna(row), None)
            # Making chunks
            chunk_text = make_chunks(
                text=row["chunk_text"], chunk_size=1000, chunk_overlap=100
            )[
                0
            ]  # Only keep the first chunks because a too long description is not interesting for this kind of dataset

            chunk_xxh64 = xxhash.xxh64(
                chunk_text.encode("utf-8"), seed=2025
            ).hexdigest()

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
            all_new_doc_ids.append(doc_id)
            insert_data(data=[new_data], table_name=table_name)

        # Sync obsolete doc_ids (remove entries not in all_new_doc_ids) as the file we are processing is the new full dataset
        sync_obsolete_doc_ids(
            table_name=table_name,
            old_doc_ids=all_old_doc_ids,
            new_doc_ids=all_new_doc_ids,
        )
    else:
        logger.error(
            f"Unknown target directory '{target_dir}' for processing data.gouv.fr files."
        )
        raise ValueError(
            f"Unknown target directory '{target_dir}' for processing data.gouv.fr files."
        )


def process_directories(
    target_dir: str, config_file_path: str, model: str = "BAAI/bge-m3"
):
    """
    Processes directory data from JSON files specified in a configuration file, extracts and transforms relevant fields,
    generates embeddings for each directory, and inserts the processed data into a database.

    Args:
        target_dir (str): The directory path where the JSON files are located.
        config_file_path (str): The path to the configuration JSON file that specifies which directory files to process.

    Raises:
        FileNotFoundError: If the configuration file or any specified directory JSON file is not found.
        json.JSONDecodeError: If there is an error decoding JSON from the configuration or data files.
        Exception: For any other unexpected errors during file loading or embedding generation.

    Workflow:
        1. Loads the configuration file to determine which directory JSON files to process.
        2. Reads and aggregates directory data from the specified JSON files.
        3. Extracts and processes various fields such as addresses, phone numbers, types, SIRET/SIREN, URLs, emails,
           opening hours, mobile applications, social networks, additional information, people in charge, and hierarchy.
        4. Generates text chunks and embeddings for each directory entry.
        5. Inserts the processed data, including embeddings, into a table in the database.

    Logging:
        Logs errors and information throughout the process, including file loading issues, JSON decoding errors,
        embedding generation retries, and the number of directories loaded.
    """

    # Check if the target directory is valid
    if STATE_ADMINISTRATIONS_DIRECTORY_FOLDER.endswith(target_dir):
        table_name = "state_administrations_directory"
    elif LOCAL_ADMINISTRATIONS_DIRECTORY_FOLDER.endswith(target_dir):
        table_name = "local_administrations_directory"
    else:
        logger.error(
            f"Unknown target directory '{target_dir}' for processing directories."
        )
        raise ValueError(
            f"Unknown target directory '{target_dir}' for processing directories."
        )

    conn = None
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
        )
        cursor = conn.cursor()

        logger.info(
            f"Fetching existing document ids from table {table_name.upper()}..."
        )
        cursor.execute(f"SELECT DISTINCT doc_id FROM {table_name.upper()};")

        all_old_doc_ids = {row[0] for row in cursor.fetchall()}

    except Exception as e:
        logger.error(f"Error connecting to the database: {e}")
        return

    ### Loading directory
    directory = []
    try:
        with open(f"{target_dir}/{table_name}.json", encoding="utf-8") as json_file:
            json_data = json.load(json_file)
            if not directory:  # First file
                directory = json_data["service"]
            else:
                directory.extend(json_data["service"])
    except FileNotFoundError:
        logger.error(f"File not found: {target_dir}/{table_name}.json.")
        raise
    except json.JSONDecodeError:
        logger.error(
            f"Error decoding JSON from the file: {target_dir}/{table_name}.json."
        )
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error while loading file {target_dir}/{table_name}.json: {e}"
        )
        raise
    logger.info(f"Loaded {len(directory)} lines of data from {target_dir}")

    ## Processing data
    all_new_doc_ids = []
    for k, data in tqdm(
        enumerate(directory), total=len(directory), desc=f"Processing {table_name}"
    ):
        chunk_id = data.get("id", "")
        name = data.get("nom", "")
        directory_url = data.get("url_service_public", "")

        # Addresses
        addresses = []
        # addresses_to_concatenate = []
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
        opening_hours = make_schedule(data.get("plage_ouverture", []))

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

        chunk_text = make_chunks_directories(
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

        insert_data(
            data=[new_data],
            table_name=table_name,
        )

        all_new_doc_ids.append(doc_id)

    # Sync obsolete doc_ids (remove entries not in all_new_doc_ids) as the file we are processing is the new full dataset
    sync_obsolete_doc_ids(
        table_name=table_name, old_doc_ids=all_old_doc_ids, new_doc_ids=all_new_doc_ids
    )


def process_dila_xml_files(target_dir: str, model: str = "BAAI/bge-m3"):
    """
    Processes XML files in the specified target directory and extracts relevant data
    to insert into corresponding database tables.

    Parameters:
        target_dir (str): The root directory containing the XML files to process.

    Raises:
        Logs errors encountered during file processing, including parsing issues
        or missing fields.

    Notes:
        - The extracted data is immediately inserted into the database using the
          `insert_data` function.
        - Textual content is cleaned and formatted for improved readability.
        - Processed files are removed from the filesystem after successful insertion
          into the database.
    """
    for root_dir, dirs, files in os.walk(target_dir):
        for file_name in files:
            if file_name.startswith("LEGIARTI") and file_name.endswith(".xml"):
                table_name = "legi"
                file_path = os.path.join(root_dir, file_name)
                try:
                    tree = ET.parse(file_path)
                    root = tree.getroot()
                    status = root.find(".//ETAT").text
                    if status in ["VIGUEUR", "ABROGE_DIFF"]:
                        cid = root.find(".//ID").text  # doc_id
                        nature = root.find(".//NATURE").text
                        title = (
                            root.find(".//CONTEXTE//TEXTE//TITRE_TXT")
                            .get("c_titre_court")
                            .strip(".")
                        )
                        category = root.find(".//CONTEXTE//TEXTE").get("nature")
                        ministry = root.find(".//CONTEXTE//TEXTE").get(
                            "ministere", None
                        )
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

                        contenu = root.find(".//BLOC_TEXTUEL/CONTENU")
                        text_content = []

                        if contenu is not None:
                            # Extract all text
                            content = ET.tostring(
                                contenu, encoding="unicode", method="xml"
                            )
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
                            text=text_content, chunk_size=5000, chunk_overlap=250
                        )
                        data_to_insert = []

                        for chunk_index, text in enumerate(chunks):
                            try:
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
                                chunk_id = f"{cid}_{chunk_index}"  # Unique ID for each chunk, starting from 0

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
                                    text,  # Original text
                                    chunk_text,  # Augmented text for better search
                                    embeddings,  # Embedding of chunk_text
                                )
                                data_to_insert.append(new_data)
                            except PermissionDeniedError as e:
                                logger.error(
                                    f"PermissionDeniedError (API key issue) for chunk {chunk_index} of file {file_path}: {e}"
                                )
                                raise

                        # Inserting all chunks at once
                        if data_to_insert:
                            insert_data(data=data_to_insert, table_name=table_name)

                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                else:
                    remove_file(file_path=file_path)  # Remove the file after processing

            elif file_name.startswith("CNILTEXT") and file_name.endswith(".xml"):
                table_name = "cnil"
                file_path = os.path.join(root_dir, file_name)
                try:
                    tree = ET.parse(file_path)
                    root = tree.getroot()
                    status = root.find(".//ETAT_JURIDIQUE").text

                    if status in ["VIGUEUR"]:
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
                            content = ET.tostring(
                                contenu, encoding="unicode", method="xml"
                            )
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
                            text=text_content, chunk_size=1500, chunk_overlap=200
                        )
                        data_to_insert = []

                        for chunk_index, text in enumerate(chunks):
                            try:
                                chunk_text = f"{title}\n{text}"

                                chunk_xxh64 = xxhash.xxh64(
                                    chunk_text.encode("utf-8"), seed=2025
                                ).hexdigest()

                                embeddings = generate_embeddings_with_retry(
                                    data=chunk_text, attempts=5, model=model
                                )[0]

                                chunk_id = f"{cid}_{chunk_index}"  # Unique ID for each chunk, starting from 0

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
                                    f"PermissionDeniedError (API key issue) for chunk {chunk_index} of file {file_path}: {e}"
                                )
                                raise

                        # Inserting all chunks at once
                        if data_to_insert:
                            insert_data(data=data_to_insert, table_name=table_name)

                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                else:
                    remove_file(file_path=file_path)  # Remove the file after processing

            elif file_name.startswith("CONSTEXT") and file_name.endswith(".xml"):
                table_name = "constit"
                file_path = os.path.join(root_dir, file_name)
                try:
                    tree = ET.parse(file_path)
                    root = tree.getroot()

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
                        text=text_content, chunk_size=1500, chunk_overlap=200
                    )
                    data_to_insert = []

                    for chunk_index, text in enumerate(chunks):
                        try:
                            chunk_text = f"{title}\n{text}"

                            chunk_xxh64 = xxhash.xxh64(
                                chunk_text.encode("utf-8"), seed=2025
                            ).hexdigest()

                            embeddings = generate_embeddings_with_retry(
                                data=chunk_text, attempts=5, model=model
                            )[0]

                            chunk_id = f"{cid}_{chunk_index}"  # Unique ID for each chunk, starting from 0

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
                                f"PermissionDeniedError (API key issue) for chunk {chunk_index} of file {file_path}: {e}"
                            )
                            raise

                    # Inserting all chunks at once
                    if data_to_insert:
                        insert_data(data=data_to_insert, table_name=table_name)

                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                else:
                    remove_file(file_path=file_path)  # Remove the file after processing

            if file_name.startswith("JORFDOLE") and file_name.endswith(".xml"):
                table_name = "dole"
                file_path = os.path.join(root_dir, file_name)
                try:
                    tree = ET.parse(file_path)
                    root = tree.getroot()
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
                        exp_memo = dole_cut_exp_memo(
                            text=content, section="introduction"
                        )
                        articles_synthesis_dict = dole_cut_exp_memo(
                            text=content, section="articles"
                        )
                    else:
                        exp_memo = None
                        articles_synthesis_dict = []

                    # Creating chunks for explanatory memorandum
                    chunks = make_chunks(
                        text=exp_memo, chunk_size=8000, chunk_overlap=400
                    )
                    data_to_insert = []
                    if not chunks:
                        chunk_text = title
                        try:
                            embeddings = generate_embeddings_with_retry(
                                data=chunk_text, attempts=5, model="BAAI/bge-m3"
                            )[0]
                            chunk_index = 0
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
                                f"PermissionDeniedError (API key issue) for chunk {chunk_index} of file {file_path}: {e}"
                            )
                            raise
                    else:
                        for chunk_index, text in enumerate(chunks):
                            try:
                                chunk_text = (title + "\n" + text).replace(
                                    "\n\n", "\n"
                                )  # Adding the title to the chunk text
                                embeddings = generate_embeddings_with_retry(
                                    data=chunk_text, attempts=5, model="BAAI/bge-m3"
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
                                    f"PermissionDeniedError (API key issue) for chunk {chunk_index} of file {file_path}: {e}"
                                )
                                raise

                    file_content_list = []
                    for k in range(
                        1, 6
                    ):  # There can be up to 5 contenu_dossier sections
                        contenu_dossier = root.find(f".//CONTENU_DOSSIER_{k}")
                        if contenu_dossier is not None:
                            # Extract all text
                            content = ET.tostring(contenu_dossier, method="xml")
                            content = "".join(ET.fromstring(content).itertext()).strip()

                            if len(content) > 0:
                                file_content_list.extend(
                                    dole_cut_file_content(text=content)
                                )

                    results = []
                    if (
                        len(file_content_list) == 0
                        and len(articles_synthesis_dict) == 0
                    ):
                        results = [
                            {
                                "article_number": None,
                                "article_synthesis": None,
                                "article_text": None,
                                "article_title": None,
                            }
                        ]
                    elif (
                        len(articles_synthesis_dict) > 0 and len(file_content_list) == 0
                    ):
                        for article in articles_synthesis_dict:
                            results.append(
                                {
                                    "article_number": article.get(
                                        "article_number", None
                                    ),
                                    "article_synthesis": article.get(
                                        "article_synthesis", None
                                    ),
                                    "article_text": None,  # Because there is no file content
                                    "article_title": article.get("title_content", None),
                                }
                            )

                    elif (
                        len(articles_synthesis_dict) == 0 and len(file_content_list) > 0
                    ):
                        for content in file_content_list:
                            results.append(
                                {
                                    "article_number": content.get(
                                        "article_number", None
                                    ),
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

                        # Adding all articles with article_number = None
                        for d in articles_synthesis_dict:
                            if d["article_number"] is None:
                                merged = {
                                    "article_number": None,
                                    "article_synthesis": d.get(
                                        "article_synthesis"
                                    ).strip()
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
                                text=chunk_text, chunk_size=8000, chunk_overlap=400
                            )

                            for chunk_index, text in enumerate(chunks):
                                chunk_id = f"{cid}_{chunk_index}"  # Unique ID for each chunk, starting from 0

                                if (
                                    chunk_index == 0
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
                                        data=chunk_text, attempts=5, model="BAAI/bge-m3"
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
                                        f"PermissionDeniedError (API key issue) for chunk {chunk_index} of file {file_path}: {e}"
                                    )
                                    raise

                        else:  # The chunks will be created by classic chunking
                            chunk_index = result_number
                            chunk_id = f"{cid}_{chunk_index}"  # Unique ID for each chunk, starting from 0
                            content_type = "dossier_content"
                            chunks = []  # As it is impossible to have an article synthesis without an article number

                            if result.get("article_text", ""):
                                chunks.append(str(result.get("article_text")).strip())
                            chunks = "\n".join(chunks).strip()

                            chunks = make_chunks(
                                text=chunks, chunk_size=8000, chunk_overlap=400
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
                                        data=chunk_text, attempts=5, model="BAAI/bge-m3"
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
                                        f"PermissionDeniedError (API key issue) for chunk {chunk_index} of file {file_path}: {e}"
                                    )
                                    raise

                    # Insert all chunks at once
                    if data_to_insert:
                        insert_data(data=data_to_insert, table_name=table_name)

                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                else:
                    remove_file(file_path=file_path)  # Remove the file after processing


def process_sheets(target_dir: str, model: str = "BAAI/bge-m3", batch_size: int = 10):
    table_name = ""
    if SERVICE_PUBLIC_PRO_DATA_FOLDER.endswith(
        target_dir
    ) or SERVICE_PUBLIC_PART_DATA_FOLDER.endswith(target_dir):
        table_name = "service_public"
    elif TRAVAIL_EMPLOI_DATA_FOLDER.endswith(target_dir):
        table_name = "travail_emploi"
    else:
        logger.error(f"Unknown target directory '{target_dir}' for processing sheets.")
        raise ValueError(
            f"Unknown target directory '{target_dir}' for processing sheets."
        )

    conn = None
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
        )
        cursor = conn.cursor()

        logger.info(
            f"Fetching existing document ids from table {table_name.upper()}..."
        )
        cursor.execute(f"SELECT DISTINCT doc_id FROM {table_name.upper()};")

        all_old_doc_ids = {row[0] for row in cursor.fetchall()}

    except Exception as e:
        logger.error(f"Error connecting to the database: {e}")
        return

    with open(os.path.join(target_dir, "sheets_as_chunks.json")) as f:
        documents = json.load(f)

    corpus_name = target_dir.split("/")[-1]
    corpus_handler = CorpusHandler.create_handler(corpus_name, documents)

    if table_name == "travail_emploi":
        all_new_doc_ids = []
        for batch_documents, batch_embeddings in corpus_handler.iter_docs_embeddings(
            batch_size=batch_size,
            model=model,
        ):
            data_to_insert = []

            for document, embeddings in zip(batch_documents, batch_embeddings):
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
                all_new_doc_ids.append(doc_id)
                data_to_insert.append(new_data)
            if data_to_insert:
                insert_data(data=data_to_insert, table_name=table_name)

        # Sync obsolete doc_ids (remove entries not in all_new_doc_ids) as the file we are processing is the new full dataset
        sync_obsolete_doc_ids(
            table_name=table_name,
            old_doc_ids=all_old_doc_ids,
            new_doc_ids=all_new_doc_ids,
        )

    elif table_name == "service_public":
        all_new_doc_ids = []
        for batch_documents, batch_embeddings in corpus_handler.iter_docs_embeddings(
            batch_size
        ):
            data_to_insert = []

            for document, embeddings in zip(batch_documents, batch_embeddings):
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

                all_new_doc_ids.append(doc_id)
                data_to_insert.append(new_data)

            if data_to_insert:
                insert_data(data=data_to_insert, table_name=table_name)

        # Sync obsolete doc_ids (remove entries not in all_new_doc_ids) as the file we are processing is the new full dataset
        sync_obsolete_doc_ids(
            table_name=table_name,
            old_doc_ids=all_old_doc_ids,
            new_doc_ids=all_new_doc_ids,
        )

    else:
        logger.error(
            f"Unknown table name '{table_name}' for target directory '{target_dir}'."
        )
        return


def process_data(base_folder: str, model: str = "BAAI/bge-m3"):
    """
    Processes data files located in the specified base folder according to its type.
    Depending on the value of `base_folder`, this function performs several operations.

    Args:
        base_folder (str): The path to the base folder containing the data to process.

    Raises:
        Any exceptions raised by the underlying processing or file operations are propagated.
    """

    all_dirs = sorted(os.listdir(base_folder))

    if STATE_ADMINISTRATIONS_DIRECTORY_FOLDER.endswith(
        base_folder
    ) or LOCAL_ADMINISTRATIONS_DIRECTORY_FOLDER.endswith(base_folder):
        logger.info(f"Processing directory files located in : {base_folder}")
        process_directories(
            target_dir=base_folder,
            config_file_path="config/data_config.json",
            model=model,
        )
        logger.info(
            logger.info(
                f"Folder: {base_folder} successfully processed and data successfully inserted into the postgres database"
            )
        )

        remove_folder(folder_path=base_folder)
        logger.debug(f"Folder: {base_folder} successfully removed after processing")
    elif DATA_GOUV_DATASETS_CATALOG_DATA_FOLDER.endswith(base_folder):
        logger.info(f"Processing files located in : {base_folder}")

        process_data_gouv_files(target_dir=base_folder, model=model)

        logger.info(
            f"Folder: {base_folder} successfully processed and data successfully inserted into the postgres database"
        )

        remove_folder(folder_path=base_folder)
        logger.debug(f"Folder: {base_folder} successfully removed after processing")
    elif TRAVAIL_EMPLOI_DATA_FOLDER.endswith(base_folder):
        logger.info(f"Processing files located in : {base_folder}")

        make_chunks_sheets(
            storage_dir=base_folder,
            structured=True,
            chunk_size=1500,
            chunk_overlap=200,
        )

        process_sheets(target_dir=base_folder, model=model)

        logger.info(
            f"Folder: {base_folder} successfully processed and data successfully inserted into the postgres database"
        )

        remove_folder(folder_path=base_folder)
        logger.debug(f"Folder: {base_folder} successfully removed after processing")

    elif SERVICE_PUBLIC_PRO_DATA_FOLDER.endswith(
        base_folder
    ) or SERVICE_PUBLIC_PART_DATA_FOLDER.endswith(base_folder):
        logger.info(f"Processing files located in : {base_folder}")

        make_chunks_sheets(
            storage_dir=base_folder,
            structured=True,
            chunk_size=1500,
            chunk_overlap=200,
        )
        process_sheets(target_dir=base_folder, model=model)

        logger.info(
            f"Folder: {base_folder} successfully processed and data successfully inserted into the postgres database"
        )

        remove_folder(folder_path=base_folder)
        logger.debug(f"Folder: {base_folder} successfully removed after processing")

    elif CNIL_DATA_FOLDER.endswith(base_folder):
        try:
            all_dirs.remove("cnil")
            all_dirs.insert(0, "cnil")  # Placing the 'cnil' folder at the beginning
        except ValueError:
            logger.debug(f"There is no 'cnil' directory in {base_folder}")

        for (
            root_dir
        ) in all_dirs:  # root_dir is the name of each folder inside the base_folder
            # Remove obscolete CIDs from the table based on the suppression list file
            for entity in os.listdir(os.path.join(base_folder, root_dir)):
                if entity.startswith("liste_suppression"):
                    try:
                        cid_to_remove = []
                        with open(os.path.join(base_folder, root_dir, entity)) as f:
                            lines = f.readlines()
                        cid_to_remove = [
                            line.strip().split("/")[-1]
                            for line in lines
                            if line.strip()
                        ]
                        logger.debug(
                            f"Removing {len(cid_to_remove)} CIDs from the 'CNIL' table based on {entity}"
                        )
                        for cid in cid_to_remove:
                            remove_data(table_name="cnil", column="cid", value=cid)
                    except Exception as e:
                        logger.error(f"Error reading {entity}: {e}")
                        continue
            # Process the XML files in the target directory
            target_dir = os.path.join(base_folder, root_dir, "cnil/global/CNIL/TEXT")
            folder_to_remove = os.path.join(base_folder, root_dir)
            if root_dir == "cnil":
                # This is the freemium extracted folder
                target_dir = os.path.join(base_folder, "cnil/global/CNIL/TEXT")
                logger.info(f"Processing folder: {target_dir}")

                process_dila_xml_files(target_dir=target_dir, model=model)
                logger.info(f"Folder: {target_dir} successfully processed")

                remove_folder(folder_path=folder_to_remove)
                logger.debug(
                    f"Folder: {folder_to_remove} successfully removed after processing"
                )
            else:  # for each folder except the freemium one
                logger.info(f"Processing folder: {target_dir}")

                process_dila_xml_files(target_dir=target_dir, model=model)
                logger.info(
                    f"Folder: {target_dir} successfully processed and data successfully inserted into the postgres database"
                )

                remove_folder(folder_path=folder_to_remove)
                logger.debug(
                    f"Folder: {folder_to_remove} successfully removed after processing"
                )

    elif CONSTIT_DATA_FOLDER.endswith(base_folder):
        try:
            all_dirs.remove("constit")
            all_dirs.insert(
                0, "constit"
            )  # Placing the 'constit' folder at the beginning
        except ValueError:
            logger.debug(f"There is no 'constit' directory in {base_folder}")

        for (
            root_dir
        ) in all_dirs:  # root_dir is the name of each folder inside the base_folder
            # Remove obscolete CIDs from the table based on the suppression list file
            for entity in os.listdir(os.path.join(base_folder, root_dir)):
                if entity.startswith("liste_suppression"):
                    try:
                        cid_to_remove = []
                        with open(os.path.join(base_folder, root_dir, entity)) as f:
                            lines = f.readlines()
                        cid_to_remove = [
                            line.strip().split("/")[-1]
                            for line in lines
                            if line.strip()
                        ]
                        logger.debug(
                            f"Removing {len(cid_to_remove)} CIDs from the 'CONSTIT' table based on {entity}"
                        )
                        for cid in cid_to_remove:
                            remove_data(table_name="constit", column="cid", value=cid)
                    except Exception as e:
                        logger.error(f"Error reading {entity}: {e}")
                        continue
            # Process the XML files in the target directory
            target_dir = os.path.join(base_folder, root_dir, "constit/global/CONS/TEXT")
            folder_to_remove = os.path.join(base_folder, root_dir)
            if root_dir == "constit":
                # This is the freemium extracted folder
                target_dir = os.path.join(base_folder, "constit/global/CONS/TEXT")
                logger.info(f"Processing folder: {target_dir}")

                process_dila_xml_files(target_dir=target_dir, model=model)
                logger.info(
                    f"Folder: {target_dir} successfully processed and data successfully inserted into the postgres database"
                )

                remove_folder(folder_path=folder_to_remove)
                logger.debug(
                    f"Folder: {folder_to_remove} successfully removed after processing"
                )
            else:  # for each folder except the freemium one
                logger.info(f"Processing folder: {target_dir}")

                process_dila_xml_files(target_dir=target_dir, model=model)
                logger.info(
                    f"Folder: {target_dir} successfully processed and data successfully inserted into the postgres database"
                )

                remove_folder(folder_path=folder_to_remove)
                logger.debug(
                    f"Folder: {folder_to_remove} successfully removed after processing"
                )

    elif DOLE_DATA_FOLDER.endswith(base_folder):
        try:
            all_dirs.remove("dole")
            all_dirs.insert(0, "dole")  # Placing the 'dole' folder at the beginning
        except ValueError:
            logger.debug(f"There is no 'dole' directory in {base_folder}")

        for (
            root_dir
        ) in all_dirs:  # root_dir is the name of each folder inside the base_folder
            # Remove obscolete CIDs from the table based on the suppression list file
            for entity in os.listdir(os.path.join(base_folder, root_dir)):
                if entity.startswith("liste_suppression"):
                    try:
                        cid_to_remove = []
                        with open(os.path.join(base_folder, root_dir, entity)) as f:
                            lines = f.readlines()
                        cid_to_remove = [
                            line.strip().split("/")[-1]
                            for line in lines
                            if line.strip()
                        ]
                        logger.debug(
                            f"Removing {len(cid_to_remove)} CIDs from the 'DOLE' table based on {entity}"
                        )
                        for cid in cid_to_remove:
                            remove_data(table_name="dole", column="cid", value=cid)
                    except Exception as e:
                        logger.error(f"Error reading {entity}: {e}")
                        continue

            # Process the XML files in the target directory
            target_dir = os.path.join(base_folder, root_dir, "dole/global/JORF/DOLE")
            folder_to_remove = os.path.join(base_folder, root_dir)
            if root_dir == "dole":
                # This is the freemium extracted folder
                target_dir = os.path.join(base_folder, "dole/global/JORF/DOLE")
                logger.info(f"Processing folder: {target_dir}")

                process_dila_xml_files(target_dir=target_dir, model=model)
                logger.info(
                    f"Folder: {target_dir} successfully processed and data successfully inserted into the postgres database"
                )

                remove_folder(folder_path=folder_to_remove)
                logger.debug(
                    f"Folder: {folder_to_remove} successfully removed after processing"
                )

            else:  # for each folder except the freemium one
                logger.info(f"Processing folder: {target_dir}")

                process_dila_xml_files(target_dir=target_dir, model=model)
                logger.info(
                    f"Folder: {target_dir} successfully processed and data successfully inserted into the postgres database"
                )

                remove_folder(folder_path=folder_to_remove)
                logger.debug(
                    f"Folder: {folder_to_remove} successfully removed after processing"
                )

    elif base_folder.endswith(LEGI_DATA_FOLDER):
        try:
            all_dirs.remove("legi")
            all_dirs.insert(0, "legi")  # Placing the 'legi' folder at the beginning
        except ValueError:
            logger.debug(f"There is no 'legi' directory in {base_folder}")

        for (
            root_dir
        ) in all_dirs:  # root_dir is the name of each folder inside the base_folder
            # Remove obscolete CIDs from the table based on the suppression list file
            for entity in os.listdir(os.path.join(base_folder, root_dir)):
                if entity.startswith("liste_suppression"):
                    try:
                        cid_to_remove = []
                        with open(os.path.join(base_folder, root_dir, entity)) as f:
                            lines = f.readlines()
                        cid_to_remove = [
                            line.strip().split("/")[-1]
                            for line in lines
                            if line.strip()
                        ]
                        logger.debug(
                            f"Removing {len(cid_to_remove)} CIDs from the 'LEGI' table based on {entity}"
                        )
                        for cid in cid_to_remove:
                            remove_data(table_name="legi", column="cid", value=cid)
                    except Exception as e:
                        logger.error(f"Error reading {entity}: {e}")
                        continue
            # Process the XML files in the target directory
            target_dir = os.path.join(
                base_folder, root_dir, "legi/global/code_et_TNC_en_vigueur"
            )
            folder_to_remove = os.path.join(base_folder, root_dir)

            if root_dir == "legi":
                # This is the freemium extracted folder
                target_dir = os.path.join(
                    base_folder, "legi/global/code_et_TNC_en_vigueur"
                )
                logger.info(f"Processing folder: {target_dir}")

                process_dila_xml_files(target_dir=target_dir, model=model)
                logger.info(
                    f"Folder: {target_dir} successfully processed and data successfully inserted into the database"
                )

                remove_folder(folder_path=folder_to_remove)
                logger.debug(
                    f"Folder: {folder_to_remove} successfully removed after processing"
                )

            else:  # for each folder except the freemium one
                logger.info(f"Processing folder: {target_dir}")

                process_dila_xml_files(target_dir=target_dir, model=model)
                logger.info(
                    f"Folder: {target_dir} successfully processed and data successfully inserted into the database"
                )

                remove_folder(folder_path=folder_to_remove)
                logger.debug(
                    f"Folder: {folder_to_remove} successfully removed after processing"
                )


def process_all_data(unprocessed_data_folder: str, model: str = "BAAI/bge-m3"):
    """
    Processes all data directories within the specified unprocessed data folder.

    Args:
        unprocessed_data_folder (str): Path to the folder containing unprocessed data directories.

    Returns:
        None

    Note:
        This function iterates over the contents of the given folder, constructs the full path for each subdirectory,
        and processes the data using the `process_data` function.
    """
    for directory in os.listdir(unprocessed_data_folder):
        base_folder = os.path.join(unprocessed_data_folder, directory)
        process_data(base_folder=base_folder, model=model)
