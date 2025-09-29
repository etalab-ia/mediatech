import psycopg2
from psycopg2.extras import RealDictCursor
from qdrant_client import QdrantClient, models
import json
from fastembed import SparseTextEmbedding
from tqdm import tqdm
import uuid
from config import (
    get_logger,
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    config_file_path,
)
from utils import (
    generate_embeddings_with_retry,
    format_model_name,
    format_to_table_name,
    extract_legi_data,
)

logger = get_logger(__name__)


def create_all_tables(model="BAAI/bge-m3", delete_existing: bool = False):
    """
    Creates the necessary tables in the PostgreSQL database as specified in the data configuration file.
    Optionally deletes existing tables before creation.
    This function:
    - Connects to the PostgreSQL database using credentials from environment variables.
    - Ensures the `pgvector` extension is enabled for vector-based operations.
    - Reads the table configuration from a JSON file.
    - Iterates through the configured table names, and for each:
        - Optionally drops the table if it exists and `delete_existing` is True.
        - Checks if the table already exists; if not, creates it with the appropriate schema.
        - Adds a vector column for embeddings and creates an HNSW index for efficient similarity search.
    - Commits all changes and logs the process.
    Args:
        delete_existing (bool, optional): If True, existing tables will be dropped before creation. Defaults to False.
    Raises:
        Logs errors if database connection, extension enabling, table creation, or index creation fails.
    """

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
        logger.info("Connected to PostgreSQL database")
        probe_vector = generate_embeddings_with_retry(data="Hey, I'am a probe", model=model)[0]
        embedding_size = len(probe_vector)

        model_name = format_model_name(model)

        # Enabling Pgvector extension
        try:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()

            # Checks if the extension is enabled
            cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
            if cursor.fetchone() is None:
                logger.error(
                    "pgvector extension could not be enabled. Please check if it's installed in your PostgreSQL instance."
                )
                raise Exception("pgvector extension not enabled")
            logger.info("pgvector extension enabled successfully")
        except Exception as e:
            logger.error(f"Error enabling pgvector extension: {e}")
            raise e

        with open(config_file_path, "r") as file:
            config = json.load(file)

        # Listing all the tables in the config file
        table_names = []
        for category, data in config.items():
            if category.lower().startswith(
                "service_public"
            ):  # Gathering service public pro and part sheets in one table
                if "SERVICE_PUBLIC" not in table_names:
                    table_names.append("SERVICE_PUBLIC")
                else:
                    pass
            elif category.lower() == "travail_emploi":
                table_names.append("TRAVAIL_EMPLOI")
            else:
                table_names.append(category)

        for table_name in table_names:
            if delete_existing:
                # Drop the table if it exists
                cursor.execute(f"DROP TABLE IF EXISTS {table_name.upper()} CASCADE;")

                conn.commit()
                logger.info(
                    f"Table '{table_name.upper()}' dropped successfully in database {POSTGRES_DB}"
                )

            # Checking if the table already exists
            cursor.execute(f"""
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_name = '{table_name.lower()}'
                );
            """)
            table_exists = cursor.fetchone()[0]

            if table_exists:
                logger.info(
                    f"Table '{table_name.upper()}' already exists in database {POSTGRES_DB}"
                )
            else:
                # Create table if doesn't exist

                if table_name.lower().endswith("directory"):
                    cursor.execute(f"""
                        CREATE TABLE {table_name.upper()} (
                            chunk_id TEXT PRIMARY KEY,
                            doc_id TEXT NOT NULL,
                            chunk_xxh64 TEXT NOT NULL,
                            types TEXT,
                            name TEXT,
                            mission_description TEXT,
                            addresses JSONB,
                            phone_numbers TEXT[],
                            mails TEXT[],
                            urls TEXT[],
                            social_medias TEXT[],
                            mobile_applications TEXT[],
                            opening_hours TEXT,
                            contact_forms TEXT[],
                            additional_information TEXT,
                            modification_date TEXT,
                            siret TEXT,
                            siren TEXT,
                            people_in_charge JSONB,
                            organizational_chart TEXT[],
                            hierarchy JSONB,
                            directory_url TEXT,
                            chunk_text TEXT,
                            "embeddings_{model_name}" vector({embedding_size}),
                            UNIQUE(chunk_id)
                        )
                    """)

                elif table_name.lower() == "travail_emploi":
                    cursor.execute(f"""
                        CREATE TABLE TRAVAIL_EMPLOI (
                            chunk_id TEXT PRIMARY KEY,
                            doc_id TEXT NOT NULL,
                            chunk_index INTEGER NOT NULL,
                            chunk_xxh64 TEXT NOT NULL,
                            title TEXT,
                            surtitle TEXT,
                            source TEXT,
                            introduction TEXT,
                            date TEXT,
                            url TEXT,
                            context TEXT[],
                            text TEXT,
                            chunk_text TEXT,
                            "embeddings_{model_name}" vector({embedding_size}),
                            UNIQUE(chunk_id)
                        )
                    """)
                elif table_name.lower() == "service_public":
                    cursor.execute(f"""
                        CREATE TABLE SERVICE_PUBLIC (
                            chunk_id TEXT PRIMARY KEY,
                            doc_id TEXT NOT NULL,
                            chunk_index INTEGER NOT NULL,
                            chunk_xxh64 TEXT NOT NULL,
                            audience TEXT,
                            theme TEXT,
                            title TEXT,
                            surtitle TEXT,
                            source TEXT,
                            introduction TEXT,
                            url TEXT,
                            related_questions JSONB,
                            web_services JSONB,
                            context TEXT[],
                            text TEXT,
                            chunk_text TEXT,
                            "embeddings_{model_name}" vector({embedding_size}),
                            UNIQUE(chunk_id)
                        )
                    """)

                elif table_name.lower() == "cnil":
                    cursor.execute(f"""
                        CREATE TABLE CNIL (
                            chunk_id TEXT PRIMARY KEY,
                            doc_id TEXT NOT NULL,
                            chunk_index INTEGER NOT NULL,
                            chunk_xxh64 TEXT NOT NULL,
                            nature TEXT,
                            status TEXT,
                            nature_delib TEXT,
                            title TEXT,
                            full_title TEXT,
                            number TEXT,
                            date TEXT,
                            text TEXT,
                            chunk_text TEXT,
                            "embeddings_{model_name}" vector({embedding_size}),
                            UNIQUE(chunk_id)
                        )
                    """)

                elif table_name.lower() == "constit":
                    cursor.execute(f"""
                        CREATE TABLE CONSTIT (
                            chunk_id TEXT PRIMARY KEY,
                            doc_id TEXT NOT NULL,
                            chunk_index INTEGER NOT NULL,
                            chunk_xxh64 TEXT NOT NULL,
                            nature TEXT,
                            solution TEXT,
                            title TEXT,
                            number TEXT,
                            decision_date TEXT,
                            text TEXT,
                            chunk_text TEXT,
                            "embeddings_{model_name}" vector({embedding_size}),
                            UNIQUE(chunk_id)
                        )
                    """)

                elif table_name.lower() == "dole":
                    cursor.execute(f"""
                        CREATE TABLE DOLE (
                            chunk_id TEXT PRIMARY KEY,
                            doc_id TEXT NOT NULL,
                            chunk_index INTEGER NOT NULL,
                            chunk_xxh64 TEXT NOT NULL,
                            category TEXT,
                            content_type TEXT,
                            title TEXT,
                            number TEXT,
                            wording TEXT,
                            creation_date TEXT,
                            article_number INTEGER,
                            article_title TEXT,
                            article_synthesis TEXT,
                            text TEXT,
                            chunk_text TEXT,
                            "embeddings_{model_name}" vector({embedding_size}),
                            UNIQUE(chunk_id)
                        )
                    """)

                elif table_name.lower() == "legi":
                    cursor.execute(f"""
                        CREATE TABLE LEGI (
                            chunk_id TEXT PRIMARY KEY,
                            doc_id TEXT NOT NULL,
                            chunk_index INTEGER NOT NULL,
                            chunk_xxh64 TEXT NOT NULL,
                            nature TEXT,
                            category TEXT,
                            ministry TEXT,
                            status TEXT,
                            title TEXT,
                            full_title TEXT,
                            subtitles TEXT,
                            number TEXT,
                            start_date TEXT,
                            end_date TEXT,
                            nota TEXT,
                            links JSONB,
                            text TEXT,
                            chunk_text TEXT,
                            "embeddings_{model_name}" vector({embedding_size}),
                            UNIQUE(chunk_id)
                        )
                    """)

                elif table_name.lower() == "data_gouv_datasets_catalog":
                    cursor.execute(f"""
                        CREATE TABLE DATA_GOUV_DATASETS_CATALOG (
                            chunk_id TEXT PRIMARY KEY,
                            doc_id TEXT,
                            chunk_xxh64 TEXT NOT NULL,
                            title TEXT,
                            acronym TEXT,
                            url TEXT,
                            organization TEXT,
                            organization_id TEXT,
                            owner TEXT,
                            owner_id TEXT,
                            description TEXT,
                            frequency TEXT,
                            license TEXT,
                            temporal_coverage_start TEXT,
                            temporal_coverage_end TEXT,
                            spatial_granularity TEXT,
                            spatial_zones TEXT,
                            featured BOOLEAN,
                            created_at TEXT,
                            last_modified TEXT,
                            tags TEXT,
                            archived TEXT,
                            resources_count INTEGER,
                            main_resources_count INTEGER,
                            resources_formats TEXT,
                            harvest_backend TEXT,
                            harvest_domain TEXT,
                            harvest_created_at TEXT,
                            harvest_modified_at TEXT,
                            harvest_remote_url TEXT,
                            quality_score REAL,
                            metric_discussions INTEGER,
                            metric_reuses INTEGER,
                            metric_reuses_by_months TEXT,
                            metric_followers INTEGER,
                            metric_followers_by_months TEXT,
                            metric_views INTEGER,
                            metric_resources_downloads REAL,
                            chunk_text TEXT,
                            "embeddings_{model_name}" vector({embedding_size}),
                            UNIQUE(chunk_id)
                        )
                    """)

                # Create index for vector similarity search
                try:
                    cursor.execute(f"""
                        CREATE INDEX ON {table_name.upper()} USING hnsw ("embeddings_{model_name}" vector_cosine_ops)
                        WITH (m = 16, ef_construction = 128);
                    """)
                except Exception as e:
                    logger.error(
                        f"Error creating HNSW index on {table_name.upper()} table: {e}"
                    )
                    raise e

                conn.commit()
                logger.info(
                    f"Table '{table_name.upper()}' created successfully in database {POSTGRES_DB}"
                )

    except Exception as e:
        logger.error(f"Error creating tables in PostgreSQL: {e}")
        raise e
    finally:
        if conn:
            conn.close()
            logger.debug("PostgreSQL connection closed")


def create_table_from_existing(
    source_table: str, target_table: str, include_indexes: bool = True
):
    """
    Copy the structure of a PostgreSQL table without its data.

    Args:
        source_table (str): Name of the source table to copy from
        target_table (str): Name of the new table to create
        include_indexes (bool): Whether to include indexes and constraints

    Raises:
        Logs errors if any exception occurs during database operations.
    """
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
        )
        cursor = conn.cursor()

        # Check if source table exists
        cursor.execute(f"""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_name = '{source_table.lower()}'
            );
        """)

        if not cursor.fetchone()[0]:
            logger.error(f"Source table '{source_table.upper()}' does not exist")
            return

        # Check if target table already exists
        cursor.execute(f"""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_name = '{target_table.lower()}'
            );
        """)

        if cursor.fetchone()[0]:
            logger.info(f"Target table '{target_table.upper()}' already exists")
            return

        if include_indexes:
            # Copy structure with all constraints and indexes
            cursor.execute(f"""
                CREATE TABLE {target_table.upper()} 
                (LIKE {source_table.upper()} INCLUDING ALL);
            """)
        else:
            # Copy only column structure
            cursor.execute(f"""
                CREATE TABLE {target_table.upper()} 
                (LIKE {source_table.upper()} INCLUDING DEFAULTS INCLUDING CONSTRAINTS);
            """)

        conn.commit()
        logger.info(
            f"Table structure successfully copied from '{source_table.upper()}' to '{target_table.upper()}'"
        )

    except Exception as e:
        logger.error(f"Error copying table structure: {e}")
    finally:
        if conn:
            conn.close()
            logger.debug("PostgreSQL connection closed")


def split_table(source_table: str, target_table: str, data_type: str, value: str):
    """
    Split data from source table to target table based on specified criteria.

    E.g., insert data from a source table into a target table based on a specific LEGI category or code.

    Args:
        source_table (str): Name of the source table to query from
        target_table (str): Name of the target table to insert data into
        data_type (str): Type of filter to apply ('category' or 'code')
        value (str): Value to filter by (category name or code title)

    Returns:
        None: Prints success/error messages to logs
    """
    try:
        # Connect to PostgreSQL database
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
        )
        cursor = conn.cursor()

        if data_type == "category":
            cursor.execute(f"""
            SELECT * FROM {source_table} WHERE LOWER(category) = '{value.lower()}'
            """)
            rows = cursor.fetchall()
            if not rows:
                logger.error(
                    f"No data found for category '{value.upper()}' in table '{source_table.upper()}'"
                )
                return
            else:
                columns = [f'"{column[0]}"' for column in cursor.description]
                placeholders = ", ".join(["%s"] * len(columns))
                insert_query = f"""
                INSERT INTO {target_table.upper()} ({", ".join(columns)})
                VALUES ({placeholders})
                ON CONFLICT (chunk_id) DO UPDATE SET
                {", ".join([f"{col}=EXCLUDED.{col}" for col in columns if col != '"chunk_id"'])};
                """
                cursor.executemany(insert_query, rows)
                conn.commit()
                conn.close()
                logger.info(
                    f"Data inserted into table '{target_table.upper()}' for category '{value.upper()}'"
                )
        elif data_type == "code":
            cursor.execute(f"""
            SELECT * FROM LEGI WHERE LOWER(category) ='code' AND LOWER(unaccent(full_title)) LIKE LOWER(unaccent('%{value.lower().replace("'", "''")}%'))
            """)
            rows = cursor.fetchall()
            if not rows:
                logger.error(
                    f"No data found for code '{value.upper()}' in table '{source_table.upper()}'"
                )
                return
            else:
                columns = [f'"{column[0]}"' for column in cursor.description]
                placeholders = ", ".join(["%s"] * len(columns))
                insert_query = f"""
                INSERT INTO {target_table.upper()} ({", ".join(columns)})
                VALUES ({placeholders})
                ON CONFLICT (chunk_id) DO UPDATE SET
                {", ".join([f"{col}=EXCLUDED.{col}" for col in columns if col != '"chunk_id"'])};
                """
                cursor.executemany(insert_query, rows)
                conn.commit()
                conn.close()
                logger.info(
                    f"Data successfully inserted into table '{target_table.upper()}' for code '{value.upper()}'"
                )
        else:
            logger.error(f"Invalid type '{type}' specified.")
            return
    except Exception as e:
        logger.error(f"Error splitting table data: {e}")
    finally:
        if conn:
            conn.close()
            logger.debug("PostgreSQL connection closed")


def split_legi_table():
    """
    Split the main legi table into separate tables based on codes and categories.

    Creates individual tables for each legal code and category, copying structure
    and data from the source legi table while maintaining indexes.
    """
    legi_codes = extract_legi_data(data_type="codes")
    legi_categories = extract_legi_data(data_type="categories")
    if "CODE" in legi_categories:
        legi_categories.remove(
            "CODE"
        )  # Remove 'CODE' as it is already handled separately

    for code in legi_codes:
        create_table_from_existing(
            source_table="legi",
            target_table=f"legi_{format_to_table_name(code)}",
            include_indexes=True,
        )
        split_table(
            source_table="legi",
            target_table=f"legi_{format_to_table_name(code)}",
            data_type="code",
            value=code,
        )

    for category in legi_categories:
        create_table_from_existing(
            source_table="legi",
            target_table=f"legi_{format_to_table_name(category)}",
            include_indexes=True,
        )
        split_table(
            source_table="legi",
            target_table=f"legi_{format_to_table_name(category)}",
            data_type="category",
            value=category,
        )


def insert_data(data: list, table_name: str, model="BAAI/bge-m3"):
    """
    Inserts a list of data rows into the specified PostgreSQL table, handling upserts and duplicate avoidance.

    Depending on the table name, constructs the appropriate INSERT ... ON CONFLICT SQL statement and executes it for all provided data rows.
    For tables other than "directories", existing rows with the same 'doc_id' are deleted before insertion to avoid duplicates and outdated data.

    Args:
        data (list): A list of tuples, each representing a row to insert into the database.
        table_name (str): The name of the target table. Supported values are "directories", "cnil", "constit", and "legi".

    Raises:
        Logs errors if any exception occurs during database operations.

    Notes:
        - Uses psycopg2 for PostgreSQL connection and execution.
        - Table and column names are hardcoded for each supported table.
        - Performs upsert (insert or update on conflict) based on the primary key 'chunk_id'.
        - Logs an error and returns if an unknown table name is provided.
    """
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
        )
        cursor = conn.cursor()

        model_name = format_model_name(model)

        if table_name.upper() in [
            "LEGI",
            "CNIL",
            "CONSTIT",
            "DOLE",
        ]:  # Only for data having a DILA cid (doc_id)
            # Delete the existing data for the same doc_id in order to avoid duplicates and outdated data
            source_doc_id = data[0][
                1
            ]  # Assuming doc_id is the second element in the tuple
            delete_query = f"DELETE FROM {table_name.upper()} WHERE doc_id = %s"
            cursor.execute(delete_query, (source_doc_id,))

        if table_name.lower().endswith("directory"):
            insert_query = f"""
                INSERT INTO {table_name.upper()} (chunk_id, doc_id, chunk_xxh64, types, name, mission_description, addresses, phone_numbers, mails, urls, social_medias, mobile_applications, opening_hours, contact_forms, additional_information, modification_date, siret, siren, people_in_charge, organizational_chart, hierarchy, directory_url, chunk_text, "embeddings_{model_name}")
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                doc_id = EXCLUDED.doc_id,
                chunk_xxh64 = EXCLUDED.chunk_xxh64,
                types = EXCLUDED.types,
                name = EXCLUDED.name,
                mission_description = EXCLUDED.mission_description,
                addresses = EXCLUDED.addresses,
                phone_numbers = EXCLUDED.phone_numbers,
                mails = EXCLUDED.mails,
                urls = EXCLUDED.urls,
                social_medias = EXCLUDED.social_medias,
                mobile_applications = EXCLUDED.mobile_applications,
                opening_hours = EXCLUDED.opening_hours,
                contact_forms = EXCLUDED.contact_forms,
                additional_information = EXCLUDED.additional_information,
                modification_date = EXCLUDED.modification_date,
                siret = EXCLUDED.siret,
                siren = EXCLUDED.siren,
                people_in_charge = EXCLUDED.people_in_charge,
                organizational_chart = EXCLUDED.organizational_chart,
                hierarchy = EXCLUDED.hierarchy,
                directory_url = EXCLUDED.directory_url,
                chunk_text = EXCLUDED.chunk_text,
                "embeddings_{model_name}" = EXCLUDED."embeddings_{model_name}";
                """

        elif table_name.lower() == "travail_emploi":
            insert_query = f"""
                INSERT INTO TRAVAIL_EMPLOI (chunk_id, doc_id, chunk_index, chunk_xxh64, title, surtitle, source, introduction, date, url, context, text, chunk_text, "embeddings_{model_name}")
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                doc_id = EXCLUDED.doc_id,
                chunk_index = EXCLUDED.chunk_index,
                chunk_xxh64 = EXCLUDED.chunk_xxh64,
                title = EXCLUDED.title,
                surtitle = EXCLUDED.surtitle,
                source = EXCLUDED.source,
                introduction = EXCLUDED.introduction,
                date = EXCLUDED.date,
                url = EXCLUDED.url,
                context = EXCLUDED.context,
                text = EXCLUDED.text,
                chunk_text = EXCLUDED.chunk_text,
                "embeddings_{model_name}" = EXCLUDED."embeddings_{model_name}";
            """
        elif table_name.lower() == "service_public":
            insert_query = f"""
                INSERT INTO SERVICE_PUBLIC (chunk_id, doc_id, chunk_index, chunk_xxh64, audience, theme, title, surtitle, source, introduction, url, related_questions, web_services, context, text, chunk_text, "embeddings_{model_name}")
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                doc_id = EXCLUDED.doc_id,
                chunk_index = EXCLUDED.chunk_index,
                chunk_xxh64 = EXCLUDED.chunk_xxh64,
                audience = EXCLUDED.audience,
                theme = EXCLUDED.theme,
                title = EXCLUDED.title,
                surtitle = EXCLUDED.surtitle,
                source = EXCLUDED.source,
                introduction = EXCLUDED.introduction,
                url = EXCLUDED.url,
                related_questions = EXCLUDED.related_questions,
                web_services = EXCLUDED.web_services,
                context = EXCLUDED.context,
                text = EXCLUDED.text,
                chunk_text = EXCLUDED.chunk_text,
                "embeddings_{model_name}" = EXCLUDED."embeddings_{model_name}";
            """
        elif table_name.lower() == "cnil":
            insert_query = f"""
                INSERT INTO CNIL (chunk_id, doc_id, chunk_index, chunk_xxh64, nature, status, nature_delib, title, full_title, number, date, text, chunk_text, "embeddings_{model_name}")
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                doc_id = EXCLUDED.doc_id,
                chunk_index = EXCLUDED.chunk_index,
                chunk_xxh64 = EXCLUDED.chunk_xxh64,
                nature = EXCLUDED.nature,
                status = EXCLUDED.status,
                nature_delib = EXCLUDED.nature_delib,
                title = EXCLUDED.title,
                full_title = EXCLUDED.full_title,
                number = EXCLUDED.number,
                date = EXCLUDED.date,
                text = EXCLUDED.text,
                chunk_text = EXCLUDED.chunk_text,
                "embeddings_{model_name}" = EXCLUDED."embeddings_{model_name}";
            """
        elif table_name.lower() == "constit":
            insert_query = f"""
                INSERT INTO CONSTIT (chunk_id, doc_id, chunk_index, chunk_xxh64, nature, solution, title, number, decision_date, text, chunk_text, "embeddings_{model_name}")
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                doc_id = EXCLUDED.doc_id,
                chunk_index = EXCLUDED.chunk_index,
                chunk_xxh64 = EXCLUDED.chunk_xxh64,
                nature = EXCLUDED.nature,
                solution = EXCLUDED.solution,
                title = EXCLUDED.title,
                number = EXCLUDED.number,
                decision_date = EXCLUDED.decision_date,
                text = EXCLUDED.text,
                chunk_text = EXCLUDED.chunk_text,
                "embeddings_{model_name}" = EXCLUDED."embeddings_{model_name}";
            """
        elif table_name.lower() == "dole":
            insert_query = f"""
                INSERT INTO DOLE (chunk_id, doc_id, chunk_index, chunk_xxh64, category, content_type, title, number, wording, creation_date, article_number, article_title, article_synthesis, text, chunk_text, "embeddings_{model_name}")
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                doc_id = EXCLUDED.doc_id,
                chunk_index = EXCLUDED.chunk_index,
                chunk_xxh64 = EXCLUDED.chunk_xxh64,
                category = EXCLUDED.category,
                content_type = EXCLUDED.content_type,
                title = EXCLUDED.title,
                number = EXCLUDED.number,
                wording = EXCLUDED.wording,
                creation_date = EXCLUDED.creation_date,
                article_number = EXCLUDED.article_number,
                article_title = EXCLUDED.article_title,
                article_synthesis = EXCLUDED.article_synthesis,
                text = EXCLUDED.text,
                chunk_text = EXCLUDED.chunk_text,
                "embeddings_{model_name}" = EXCLUDED."embeddings_{model_name}";
            """

        elif table_name.lower() == "legi":
            insert_query = f"""
                INSERT INTO LEGI (chunk_id, doc_id, chunk_index, chunk_xxh64, nature, category, ministry, status, title, full_title, subtitles, number, start_date, end_date, nota, links, text, chunk_text, "embeddings_{model_name}")
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                doc_id = EXCLUDED.doc_id,
                chunk_index = EXCLUDED.chunk_index,
                chunk_xxh64 = EXCLUDED.chunk_xxh64,
                nature = EXCLUDED.nature,
                category = EXCLUDED.category,
                ministry = EXCLUDED.ministry,
                status = EXCLUDED.status,
                title = EXCLUDED.title,
                full_title = EXCLUDED.full_title,
                subtitles = EXCLUDED.subtitles,
                number = EXCLUDED.number,
                start_date = EXCLUDED.start_date,
                end_date = EXCLUDED.end_date,
                nota = EXCLUDED.nota,
                links = EXCLUDED.links,
                text = EXCLUDED.text,
                chunk_text = EXCLUDED.chunk_text,
                "embeddings_{model_name}" = EXCLUDED."embeddings_{model_name}";
            """
        elif table_name.lower() == "data_gouv_datasets_catalog":
            insert_query = f"""
                INSERT INTO DATA_GOUV_DATASETS_CATALOG (chunk_id, doc_id, chunk_xxh64, title, acronym, url, organization, organization_id, owner, owner_id, description, frequency, license, temporal_coverage_start, temporal_coverage_end, spatial_granularity, spatial_zones, featured, created_at, last_modified, tags, archived, resources_count, main_resources_count, resources_formats, harvest_backend, harvest_domain, harvest_created_at, harvest_modified_at, harvest_remote_url, quality_score, metric_discussions, metric_reuses, metric_reuses_by_months, metric_followers, metric_followers_by_months, metric_views, metric_resources_downloads, chunk_text, "embeddings_{model_name}")
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                doc_id = EXCLUDED.doc_id,
                chunk_xxh64 = EXCLUDED.chunk_xxh64,
                title = EXCLUDED.title,
                acronym = EXCLUDED.acronym,
                url = EXCLUDED.url,
                organization = EXCLUDED.organization,
                organization_id = EXCLUDED.organization_id,
                owner = EXCLUDED.owner,
                owner_id = EXCLUDED.owner_id,
                description = EXCLUDED.description,
                frequency = EXCLUDED.frequency,
                license = EXCLUDED.license,
                temporal_coverage_start = EXCLUDED.temporal_coverage_start,
                temporal_coverage_end = EXCLUDED.temporal_coverage_end,
                spatial_granularity = EXCLUDED.spatial_granularity,
                spatial_zones = EXCLUDED.spatial_zones,
                featured = EXCLUDED.featured,
                created_at = EXCLUDED.created_at,
                last_modified = EXCLUDED.last_modified,
                tags = EXCLUDED.tags,
                archived = EXCLUDED.archived,
                resources_count = EXCLUDED.resources_count,
                main_resources_count = EXCLUDED.main_resources_count,
                resources_formats = EXCLUDED.resources_formats,
                harvest_backend = EXCLUDED.harvest_backend,
                harvest_domain = EXCLUDED.harvest_domain,
                harvest_created_at = EXCLUDED.harvest_created_at,
                harvest_modified_at = EXCLUDED.harvest_modified_at,
                harvest_remote_url = EXCLUDED.harvest_remote_url,
                quality_score = EXCLUDED.quality_score,
                metric_discussions = EXCLUDED.metric_discussions,
                metric_reuses = EXCLUDED.metric_reuses,
                metric_reuses_by_months = EXCLUDED.metric_reuses_by_months,
                metric_followers = EXCLUDED.metric_followers,
                metric_followers_by_months = EXCLUDED.metric_followers_by_months,
                metric_views = EXCLUDED.metric_views,
                metric_resources_downloads = EXCLUDED.metric_resources_downloads,
                chunk_text = EXCLUDED.chunk_text,
                "embeddings_{model_name}" = EXCLUDED."embeddings_{model_name}";
            """
        else:
            logger.error(f"Unknown table name: {table_name}")
            conn.commit()
            conn.close()
            return
        cursor.executemany(insert_query, data)
        conn.commit()
        logger.debug("Data inserted into PostgreSQL database")
    except Exception as e:
        logger.error(f"Error inserting data into PostgreSQL: {e}\n{data}")
        raise e
    finally:
        if conn:
            conn.close()
            logger.debug("PostgreSQL connection closed")


def postgres_to_qdrant(
    table_name: str,
    qdrant_client: QdrantClient,
    collection_name: str,
    model: str = "BAAI/bge-m3",
    delete_existing: bool = False,
):
    """
    Transfer data from a PostgreSQL table to a Qdrant vector database collection.

    This function reads data from a specified PostgreSQL table, generates embeddings
    for hybrid search using the BM25 model, and stores the data in a Qdrant collection
    with vector and sparse vector configurations.

    Args:
        table_name (str): Name of the PostgreSQL table to read data from.
        qdrant_client (QdrantClient): Initialized Qdrant client for database operations.
        collection_name (str): Name of the Qdrant collection to write data to.
        delete_existing (bool, optional): Whether to delete existing collection data.
            Defaults to False (though the collection is recreated regardless).

    Raises:
        Exception: Any error encountered during database operations is logged.

    Note:
        The function uses BAAI/bge-m3 for dense vector embeddings and Qdrant/bm25 by default for
        sparse vector embeddings to support hybrid search.
    """

    probe_vector = generate_embeddings_with_retry(data="Hey, I'am a probe", model=model)[0]
    embedding_size = len(probe_vector)
    model_name = format_model_name(model)
    bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25")  # For hybrid search

    if delete_existing:
        # Drop the collection if it exists
        try:
            qdrant_client.delete_collection(collection_name=collection_name)
            logger.info(f"Collection '{collection_name}' deleted successfully")
        except Exception as e:
            logger.error(f"Error deleting collection '{collection_name}': {e}")
            raise e

    # Create the Qdrant collection if it doesn't exist
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config={
            model: models.VectorParams(
                size=embedding_size, distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )  # For hybrid search
        },
    )

    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
        )
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Read data from PostgreSQL
        cursor.execute(f"SELECT * FROM {table_name.upper()}")
        rows = cursor.fetchall()

        # Prepare data for Qdrant
        for row in tqdm(rows, desc="Inserting data into Qdrant", unit="rows"):
            bm25_embeddings = list(
                bm25_embedding_model.passage_embed(row["chunk_text"])
            )
            chunk_id = row["chunk_id"]
            embeddings = json.loads(row[f"embeddings_{model_name}"])
            metadata = dict(row)
            del (
                metadata[f"embeddings_{model_name}"],
            )  # Remove unnecessary fields from metadata

            # Generate UUID from chunk_id for consistency
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

            qdrant_client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector={
                            model: embeddings,
                            "bm25": bm25_embeddings[0].as_object(),
                        },
                        payload=metadata,
                    )
                ],
            )

        conn.commit()
    except Exception as e:
        logger.error(f"Error inserting data into Qdrant: {e}")
        raise e
    finally:
        if conn:
            conn.close()
            logger.debug("PostgreSQL connection closed")


def remove_data(table_name: str, column: str, value: str):
    """
    Remove data from a PostgreSQL table based on a specific column and value.

    Args:
        table_name (str): Name of the PostgreSQL table to remove data from.
        column (str): Column name to filter the rows to be removed.
        value (str): Value in the specified column to match for removal.

    Raises:
        Exception: Any error encountered during database operations is logged.
    """
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
        )
        cursor = conn.cursor()

        delete_query = f"DELETE FROM {table_name.upper()} WHERE {column} = %s"
        cursor.execute(delete_query, (value,))
        conn.commit()
        logger.info(
            f"Data removed from {table_name.upper()} table where {column} = {value} (if exists)"
        )
    except Exception as e:
        logger.error(f"Error removing data from PostgreSQL: {e}")
    finally:
        if conn:
            conn.close()
            logger.debug("PostgreSQL connection closed")


def sync_obsolete_doc_ids(table_name: str, old_doc_ids: list, new_doc_ids: list):
    """
    Synchronizes a table by deleting rows with obsolete document ids.

    This function compares the provided lists of old_doc_ids and new_doc_ids against all existing document ids in the table.
    Any document id present in the table but not in the new list is considered obsolete and all its corresponding
    rows are deleted in a single, efficient operation.

    Args:
        table_name (str): The name of the table to synchronize.
        old_doc_ids (list): A list of all existing document ids in the table.
        new_doc_ids (list): A list of all current, valid document ids.
    """
    if not new_doc_ids or not old_doc_ids:
        logger.warning(
            f"Received an empty list of new or old document ids for table {table_name}. Skipping deletion."
        )
        return

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

        old_doc_ids_set = set(old_doc_ids)
        logger.debug(f"Found {len(old_doc_ids_set)} unique existing document ids.")

        # Comparing old and new document ids to find obsolete ones
        new_doc_ids_set = set(new_doc_ids)
        logger.debug(f"Received {len(new_doc_ids_set)} new document ids.")

        doc_ids_to_delete = old_doc_ids_set - new_doc_ids_set

        # Delete all obsolete document ids in a single query
        if doc_ids_to_delete:
            logger.info(
                f"Found {len(doc_ids_to_delete)} obsolete document ids to delete."
            )
            delete_query = f"DELETE FROM {table_name.upper()} WHERE doc_id IN %s;"
            cursor.execute(delete_query, (tuple(doc_ids_to_delete),))
            conn.commit()
            logger.info(
                f"Successfully deleted {cursor.rowcount} rows for {len(doc_ids_to_delete)} obsolete document ids from {table_name.upper()}."
            )
        else:
            logger.info(
                f"No obsolete document ids found in {table_name.upper()}. No deletion needed."
            )

    except Exception as e:
        logger.error(
            f"Error during obsolete document id synchronization for table {table_name}: {e}"
        )
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
            logger.debug("PostgreSQL connection closed")
