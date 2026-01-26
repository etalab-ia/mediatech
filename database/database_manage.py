import json
import os
import uuid
from contextlib import contextmanager

import duckdb
import psycopg2
from fastembed import SparseTextEmbedding
from psycopg2.extras import RealDictCursor
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from config import (
    POSTGRES_DB,
    POSTGRES_HOST,
    POSTGRES_PASSWORD,
    POSTGRES_PORT,
    POSTGRES_USER,
    config_file_path,
    get_logger,
    parquet_files_folder,
)
from utils import (
    _extract_distinct_data,
    format_model_name,
    format_to_table_name,
    generate_embeddings_with_retry,
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
        - Creates an index on doc_id column for faster queries.
    - Commits all changes and logs the process.
    Args:
        model (str): The embedding model to use. Defaults to "BAAI/bge-m3".
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
        probe_vector = generate_embeddings_with_retry(
            data="Hey, I'am a probe", model=model
        )[0]
        embedding_size = len(probe_vector)

        model_name = format_model_name(model)
        CONFIG_TABLES = ["table_mapping"]
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

        table_names = CONFIG_TABLES
        table_names.extend([category for category in config.keys()])

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

                if table_name.lower() not in CONFIG_TABLES:
                    # Check if doc_id index exists
                    index_name = f"idx_{table_name.lower()}_doc_id"
                    cursor.execute(f"""
                        SELECT EXISTS (
                            SELECT 1 FROM pg_indexes 
                            WHERE tablename = '{table_name.lower()}' 
                            AND indexname = '{index_name}'
                        );
                    """)
                    index_exists = cursor.fetchone()[0]

                    if not index_exists:
                        logger.info(
                            f"Creating missing index {index_name} on existing table..."
                        )
                        cursor.execute(f"""
                            CREATE INDEX IF NOT EXISTS {index_name} 
                            ON {table_name.upper()}(doc_id);
                        """)
                        conn.commit()
                        logger.info(f"Index {index_name} created successfully")
                    else:
                        logger.info(f"Index {index_name} already exists")

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

                elif table_name.lower() == "table_mapping":
                    cursor.execute("""
                        CREATE TABLE TABLE_MAPPING (
                            table_name VARCHAR(63) PRIMARY KEY,
                            full_table_name VARCHAR NOT NULL
                        )
                    """)

                # Create HNSW index for vector similarity search
                try:
                    if table_name.lower() not in CONFIG_TABLES:
                        cursor.execute(f"""
                            CREATE INDEX ON {table_name.upper()} USING hnsw ("embeddings_{model_name}" vector_cosine_ops)
                            WITH (m = 16, ef_construction = 128);
                        """)
                        logger.debug(f"HNSW index created on {table_name.upper()}")
                except Exception as e:
                    logger.error(
                        f"Error creating HNSW index on {table_name.upper()} table: {e}"
                    )
                    raise e

                # Create index on doc_id for faster GROUP BY and WHERE operations
                try:
                    if table_name.lower() not in CONFIG_TABLES:
                        cursor.execute(f"""
                            CREATE INDEX idx_{table_name.lower()}_doc_id 
                            ON {table_name.upper()}(doc_id);
                        """)
                        logger.debug(
                            f"B-tree index on doc_id created for {table_name.upper()}"
                        )
                except Exception as e:
                    logger.error(
                        f"Error creating doc_id index on {table_name.upper()} table: {e}"
                    )
                    raise e

                conn.commit()
                logger.info(
                    f"Table '{table_name.upper()}' created successfully in database {POSTGRES_DB} with indexes"
                )

                # Mapping table entry
                update_mapping_table(
                    table_name=table_name[:63], full_table_name=table_name
                )

    except Exception as e:
        logger.error(f"Error creating tables in PostgreSQL: {e}")
        raise e
    finally:
        if conn:
            conn.close()
            logger.debug("PostgreSQL connection closed")


def update_mapping_table(table_name: str, full_table_name: str):
    """
    Inserts or updates a mapping entry in the TABLE_MAPPING table.

    Args:
        table_name (str): The short name of the table (max 63 characters by default in PostgreSQL).
        full_table_name (str): The full name of the table.
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
        cursor.execute(
            """
            INSERT INTO TABLE_MAPPING (table_name, full_table_name)
            VALUES (%s, %s)
            ON CONFLICT (table_name) DO UPDATE SET
                full_table_name = EXCLUDED.full_table_name;
        """,
            (table_name, full_table_name),
        )
        conn.commit()
        logger.debug(f"Inserted/Updated mapping: '{table_name}' -> '{full_table_name}'")
    except Exception as e:
        logger.error(f"Error inserting/updating mapping table: {e}")
    finally:
        if conn:
            conn.close()
            logger.debug("PostgreSQL connection closed")


@contextmanager
def refresh_table(table_name: str, model: str = "BAAI/bge-m3"):
    """
    Context manager for refreshing a PostgreSQL table by dropping indexes, truncating data,
    and recreating indexes.

    Args:
        table_name (str): Name of the table to refresh
        model (str): Embedding model name for index recreation
    """
    conn = None
    try:
        # Drop indexes + Truncate
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
        )
        cursor = conn.cursor()
        model_name = format_model_name(model)

        # Check if table exists
        cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = %s
            );
        """,
            (table_name.lower(),),
        )

        if not cursor.fetchone()[0]:
            logger.warning(f"Table '{table_name.upper()}' does not exist")
            yield
            return

        logger.info(f"Starting refresh for table {table_name.upper()}")

        # Drop HNSW index
        cursor.execute(
            """
            SELECT indexname FROM pg_indexes
            WHERE tablename = %s AND indexdef LIKE %s;
        """,
            (table_name.lower(), f"%embeddings_{model_name}%"),
        )

        hnsw_result = cursor.fetchone()
        if hnsw_result:
            cursor.execute(f'DROP INDEX IF EXISTS "{hnsw_result[0]}";')
            logger.debug(f"Dropped HNSW index on {table_name.upper()}")

        # Drop doc_id index
        doc_id_idx = f"idx_{table_name.lower()}_doc_id"
        cursor.execute(f"DROP INDEX IF EXISTS {doc_id_idx};")
        logger.debug(f"Dropped B-tree index on doc_id for {table_name.upper()}")

        # Truncate table
        cursor.execute(f"TRUNCATE TABLE {table_name.upper()} RESTART IDENTITY;")
        conn.commit()
        logger.info(f"Table {table_name.upper()} truncated successfully")

        conn.close()

        yield

        # Recreate indexes
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
        )
        cursor = conn.cursor()

        logger.info(f"Recreating indexes on {table_name.upper()}...")

        # Recreate HNSW index
        cursor.execute(
            f"""CREATE INDEX ON {table_name.upper()} USING hnsw ("embeddings_{model_name}" vector_cosine_ops) WITH (m = 16, ef_construction = 128);"""
        )
        logger.debug(f"HNSW index recreated on {table_name.upper()}")

        # Recreate doc_id index
        cursor.execute(
            f"""CREATE INDEX idx_{table_name.lower()}_doc_id ON {table_name.upper()}(doc_id);"""
        )
        logger.debug(f"B-tree index on doc_id recreated for {table_name.upper()}")

        # Update statistics
        cursor.execute(f"ANALYZE {table_name.upper()};")
        conn.commit()
        logger.debug(
            f"All indexes recreated and statistics updated for {table_name.upper()}"
        )

    except Exception as e:
        logger.error(f"Error during table refresh for '{table_name.upper()}': {e}")
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()
            logger.debug("PostgreSQL connection closed")


def drop_table(table_name: str, cascade: bool = False):
    """
    Drop a PostgreSQL table.

    Args:
        table_name (str): Name of the table to drop
        cascade (bool): If True, automatically drop objects that depend on the table.
                       Defaults to False.

    Raises:
        Logs errors if any exception occurs during database operations.
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

        # Check if table exists
        cursor.execute(f"""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_name = '{table_name.lower()}'
            );
        """)

        if not cursor.fetchone()[0]:
            logger.warning(f"Table '{table_name.upper()}' does not exist")
            return

        # Drop the table
        cascade_clause = "CASCADE" if cascade else ""
        cursor.execute(f"DROP TABLE {table_name.upper()} {cascade_clause};")

        conn.commit()
        logger.info(
            f"Table '{table_name.upper()}' dropped successfully from database {POSTGRES_DB}"
        )

    except Exception as e:
        logger.error(f"Error dropping table '{table_name.upper()}': {e}")
        if conn:
            conn.rollback()
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

        # Register the table name mapping
        update_mapping_table(
            table_name=target_table.lower()[:63], full_table_name=target_table.lower()
        )  # Truncate to 63 chars for PostgreSQL table name limit

        conn.commit()
        logger.info(
            f"Table structure successfully copied from '{source_table.upper()}' to '{target_table.upper()}'"
        )
        logger.debug(f"Registered mapping: '{target_table}' -> '{target_table}'")

    except Exception as e:
        logger.error(f"Error copying table structure: {e}")
    finally:
        if conn:
            conn.close()
            logger.debug("PostgreSQL connection closed")


def _split_table(
    source_table: str,
    target_table: str,
    data_type: str,
    value: str,
    batch_size: int = 50000,
):
    """
    Split data from source table to target table based on specified criteria.

    E.g., insert data from a source table into a target table based on a specific LEGI category or code.

    Args:
        source_table (str): Name of the source table to query from
        target_table (str): Name of the target table to insert data into
        data_type (str): Type of filter to apply ('category' or 'code')
        value (str): Value to filter by (category name or code title)
        batch_size (int): Number of rows to process per batch. Default is 50,000.

    Returns:
        None: Prints success/error messages to logs
    """
    conn = None
    cursor = None
    insert_cursor = None

    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
        )

        # Building WHERE clause based on data_type
        if data_type == "category":
            where_clause = f"LOWER(category) = '{value.lower()}'"
        elif data_type == "code" and source_table.lower() == "legi":
            escaped_value = value.lower().replace("'", "''")
            where_clause = f"LOWER(category) = 'code' AND LOWER(unaccent(full_title)) LIKE LOWER(unaccent('%{escaped_value}%'))"
        else:
            logger.error(f"Invalid type '{data_type}' specified.")
            return

        # Check first if there is data to copy
        check_cursor = conn.cursor()
        check_query = f"""
            SELECT COUNT(*) FROM {source_table.upper()}
            WHERE {where_clause}
        """
        check_cursor.execute(check_query)
        row_count = check_cursor.fetchone()[0]
        check_cursor.close()

        if row_count == 0:
            logger.warning(
                f"No data found for {data_type} '{value}' in table '{source_table.upper()}'"
            )
            return

        logger.info(f"Found {row_count:,} rows to copy for {data_type} '{value}'")

        # Retrieve column names BEFORE creating the named cursor
        metadata_cursor = conn.cursor()
        metadata_cursor.execute(f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = '{source_table.lower()}'
            ORDER BY ordinal_position
        """)
        columns = [row[0] for row in metadata_cursor.fetchall()]
        metadata_cursor.close()

        escaped_columns = [f'"{col}"' for col in columns]

        # Create a named cursor (server-side cursor)
        cursor = conn.cursor(name=f"split_cursor_{uuid.uuid4().hex[:8]}")
        cursor.itersize = batch_size

        # SELECT query with ORDER BY for consistency
        select_query = f"""
            SELECT * FROM {source_table.upper()}
            WHERE {where_clause}
            ORDER BY chunk_id
        """

        cursor.execute(select_query)

        # Create a second cursor for INSERT operations
        insert_cursor = conn.cursor()

        # Retrieve JSONB columns ONCE
        insert_cursor.execute(f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = '{target_table.lower()}'
            AND data_type = 'jsonb';
        """)
        jsonb_columns = {row[0] for row in insert_cursor.fetchall()}
        jsonb_indices = {i for i, col in enumerate(columns) if col in jsonb_columns}

        # Prepare INSERT query
        placeholders = ", ".join(["%s"] * len(columns))
        update_clause = ", ".join(
            [f'"{col}" = EXCLUDED."{col}"' for col in columns if col != "chunk_id"]
        )
        insert_query = f"""
            INSERT INTO {target_table.upper()} ({", ".join(escaped_columns)})
            VALUES ({placeholders})
            ON CONFLICT (chunk_id) DO UPDATE SET {update_clause};
        """
        processed_rows = 0
        batch_number = 0

        # Processing batches
        while True:
            batch_number += 1

            rows = cursor.fetchmany(batch_size)
            if not rows:
                break

            # Convert JSONB columns
            converted_rows = []
            for row in rows:
                converted_row = list(row)
                for idx in jsonb_indices:
                    if converted_row[idx] is not None and isinstance(
                        converted_row[idx], (dict, list)
                    ):
                        converted_row[idx] = json.dumps(converted_row[idx])
                converted_rows.append(tuple(converted_row))

            # Execute the insertion
            insert_cursor.executemany(insert_query, converted_rows)
            processed_rows += len(rows)

            if batch_number % 10 == 0:  # Log every 10 batches
                logger.info(f"Batch {batch_number}: {processed_rows:,} rows processed")

        conn.commit()

        logger.info(
            f"Data successfully inserted into table '{target_table.upper()}' "
            f"for {data_type} '{value.upper()}' ({processed_rows:,} rows in {batch_number} batches)"
        )

    except Exception as e:
        logger.error(f"Error splitting table data for {data_type} '{value}': {e}")
        if conn:
            conn.rollback()
        raise e
    finally:
        if cursor:
            try:
                cursor.close()
            except Exception:
                pass
        if insert_cursor:
            try:
                insert_cursor.close()
            except Exception:
                pass
        if conn:
            conn.close()
            logger.debug("PostgreSQL connection closed")


def split_legi_table(source_table: str = "legi", export_to_parquet: bool = False):
    """
    Split the main legi table into separate tables based on codes and categories.

    Creates individual tables for each legal code and category, copying structure
    and data from the source legi table while maintaining indexes.

    Args:
        source_table (str): Name of the source legi table. Defaults to "legi".
        export_to_parquet (bool): If True, exports the created tables to Parquet files
                                  and drops them from the database. Defaults to False.
    """
    legi_codes = _extract_distinct_data(data_type="codes", source_table=source_table)
    legi_categories = _extract_distinct_data(
        data_type="category", source_table=source_table
    )

    # Remove 'CODE' as it is already handled separately
    if "CODE" in legi_categories:
        legi_categories.remove("CODE")

    def _process_legi_split(items: list, data_type: str):
        """
        Helper to process codes or categories uniformly.

        Args:
            items (list): List of codes or categories to process.
            data_type (str): Type of data being processed ('code' or 'category').
        """
        for item in sorted(items):
            if not item:
                if data_type == "category":
                    not_item = "uncategorized"
                else:
                    not_item = "misc"
                target_table = f"{source_table.lower()}_{format_to_table_name(not_item)}"  # Full table name
            else:
                target_table = f"{source_table.lower()}_{format_to_table_name(item)}"  # Full table name

            truncated_target_table = target_table[:63]  # Truncated for PostgreSQL limit

            create_table_from_existing(
                source_table=source_table,
                target_table=target_table,  # Automatically truncated to 63 chars by PostgreSQL (table name limit)
                include_indexes=True,
            )

            _split_table(
                source_table=source_table,
                target_table=truncated_target_table,  # Truncate to 63 chars for PostgreSQL table name limit
                data_type=data_type,
                value=item,
            )

            if export_to_parquet:
                try:
                    export_table_to_parquet(table_name=truncated_target_table)
                    drop_table(table_name=truncated_target_table)

                except Exception as e:
                    logger.error(
                        f"Failed to export table {target_table} to Parquet: {e}"
                    )

    _process_legi_split(items=legi_codes, data_type="code")
    _process_legi_split(items=legi_categories, data_type="category")


def export_table_to_parquet(
    table_name: str,
    parquet_folder: str = parquet_files_folder,
    rows_per_file: int = 50000,
):
    """
    Exports tables from the PostgreSQL database to Parquet files.
    Groups rows by doc_id to ensure all chunks of the same document stay together.

    Args:
        table_name (str): The name of the table to export, or "all" for all tables.
        parquet_folder (str): The path where the Parquet files will be saved.
        rows_per_file (int): Target number of rows per file. Defaults to 50000.
                            Actual count may vary to keep doc_id groups intact.

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

        def _export_single_table(table_name: str, folder_name: str = ""):
            """Exports a single table, grouping by doc_id into multiple Parquet files."""
            try:
                # Count total rows in the table
                conn.execute(f"SELECT COUNT(*) FROM postgres_db.{table_name}")
                table_row_count = conn.fetchone()[0]

                if table_row_count == 0:
                    logger.warning(f"No data found in table '{table_name}', skipping.")
                    return

                # Retrieve all doc_id with their chunk count, sorted
                conn.execute(f"""
                    SELECT doc_id, COUNT(*) as chunk_count 
                    FROM postgres_db.{table_name} 
                    GROUP BY doc_id 
                    ORDER BY doc_id
                """)
                doc_id_counts = conn.fetchall()

                total_doc_ids = len(doc_id_counts)
                logger.info(
                    f"Exporting {table_row_count} rows from table '{table_name}' "
                    f"({total_doc_ids} distinct doc_ids)..."
                )

                # Reading table_mapping
                conn.execute(f"""SELECT full_table_name FROM postgres_db.table_mapping
                                 WHERE table_name = '{table_name}';""")
                full_table_name = conn.fetchone()[0]

                full_output_folder = os.path.join(parquet_folder, folder_name)
                os.makedirs(full_output_folder, exist_ok=True)

                file_index = 0
                current_batch_doc_ids = []
                current_row_count = 0

                has_chunk_index = (
                    conn.execute(
                        f"""SELECT COUNT(*) FROM information_schema.columns 
                        WHERE table_schema = 'public' 
                        AND LOWER(table_name) = LOWER('{table_name}') 
                        AND column_name = 'chunk_index'"""
                    ).fetchone()[0]
                    > 0
                )

                for doc_id, chunk_count in doc_id_counts:
                    # If adding this doc_id exceeds the limit AND we already have doc_ids
                    if current_batch_doc_ids and (
                        current_row_count + chunk_count > rows_per_file
                    ):
                        # Export the current batch
                        _export_batch(
                            conn=conn,
                            table_name=table_name,
                            full_table_name=full_table_name,
                            doc_ids=current_batch_doc_ids,
                            file_index=file_index,
                            output_folder=full_output_folder,
                            row_count=current_row_count,
                            has_chunk_index=has_chunk_index,
                        )
                        file_index += 1
                        current_batch_doc_ids = []
                        current_row_count = 0

                    # Add this doc_id to the current batch
                    current_batch_doc_ids.append(doc_id)
                    current_row_count += chunk_count

                # Export the last batch if there are remaining doc_ids
                if current_batch_doc_ids:
                    _export_batch(
                        conn=conn,
                        table_name=table_name,
                        full_table_name=full_table_name,
                        doc_ids=current_batch_doc_ids,
                        file_index=file_index,
                        output_folder=full_output_folder,
                        row_count=current_row_count,
                        has_chunk_index=has_chunk_index,
                    )
                    file_index += 1

                # Check the total number of rows exported
                global_path = os.path.join(
                    full_output_folder,
                    full_table_name,
                    f"{full_table_name}_part_*.parquet",
                )
                parquet_row_count = conn.execute(
                    f"SELECT COUNT(*) FROM read_parquet('{global_path}')"
                ).fetchone()[0]

                logger.info(
                    f"Successfully exported table '{table_name}': "
                    f"{table_row_count} rows -> {parquet_row_count} rows in {file_index} file(s)."
                )

            except Exception as table_error:
                logger.error(f"Error processing table '{table_name}': {table_error}")
                if table_name != "all":
                    raise

        def _export_batch(
            conn,
            table_name: str,
            full_table_name: str,
            doc_ids: list,
            file_index: int,
            output_folder: str,
            row_count: int,
            has_chunk_index: bool,
        ):
            """
            Exports a batch of doc_ids to a single Parquet file.

            Args:
                conn: DuckDB connection object.
                table_name (str): Name of the table to export from.
                full_table_name (str): Full table name for output folder naming.
                doc_ids (list): List of doc_ids to include in this batch.
                file_index (int): Index of the output file.
                output_folder (str): Base output folder path.
                row_count (int): Number of rows in this batch.
                has_chunk_index (bool): Whether the table has a chunk_index column.
            """
            try:
                final_output_folder = os.path.join(output_folder, full_table_name)
                os.makedirs(final_output_folder, exist_ok=True)
                output_path = os.path.join(
                    final_output_folder,
                    f"{full_table_name}_part_{file_index}.parquet",
                )

                # Creating WHERE clause
                doc_ids_escaped = [doc_id.replace("'", "''") for doc_id in doc_ids]
                doc_ids_str = "', '".join(doc_ids_escaped)

                logger.debug(
                    f"Exporting part {file_index}: {len(doc_ids)} doc_ids, "
                    f"~{row_count} rows to {output_path}"
                )

                if has_chunk_index:
                    conn.execute(f"""
                        COPY (
                            SELECT * FROM postgres_db.{table_name}
                            WHERE doc_id IN ('{doc_ids_str}')
                            ORDER BY doc_id, chunk_index
                        ) TO '{output_path}'
                        (FORMAT PARQUET, COMPRESSION 'ZSTD', PARQUET_VERSION 'V2', ROW_GROUP_SIZE 50000)
                    """)
                else:
                    conn.execute(f"""
                    COPY (
                        SELECT * FROM postgres_db.{table_name}
                        WHERE doc_id IN ('{doc_ids_str}')
                        ORDER BY doc_id
                    ) TO '{output_path}'
                    (FORMAT PARQUET, COMPRESSION 'ZSTD', PARQUET_VERSION 'V2', ROW_GROUP_SIZE 50000)
                """)

                logger.debug(
                    f"Successfully exported batch {file_index} to {output_path}"
                )

            except Exception as e:
                logger.error(
                    f"Error exporting batch {file_index} for table '{table_name}': {e}"
                )
                logger.error(
                    f"Failed doc_ids: {doc_ids[:5]}..."
                )  # Log first 5 doc_ids for debugging
                raise

        os.makedirs(parquet_folder, exist_ok=True)

        if table_name == "all":
            conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public' AND table_type='BASE TABLE';"
            )
            tables = [row[0] for row in conn.fetchall()]
            logger.info(f"Found {len(tables)} tables to export: {tables}")
            if "table_mapping" in tables:
                tables.remove("table_mapping")  # Exclude table_mapping table
            for table in tables:
                _export_single_table(table_name=table)
        elif table_name.startswith("legi_"):
            _export_single_table(table_name=table_name, folder_name="legi")
        else:
            _export_single_table(table_name=table_name)

    except Exception as e:
        logger.error(f"An error occurred during SQL Table export: {e}")
        raise
    finally:
        if "conn" in locals():
            conn.close()


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
        logger.error(f"Error inserting data into PostgreSQL: {e}\n{str(data)[:200]}...")
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


### LEGACY FUNCTIONS ###


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

    probe_vector = generate_embeddings_with_retry(
        data="Hey, I'am a probe", model=model
    )[0]
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

    conn = None
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


def get_distinct_values(table_name: str, column: str) -> list:
    """
    Retrieves all unique values from a specified column in a PostgreSQL table.

    Args:
        table_name (str): The name of the table to query.
        column (str): The name of the column to retrieve distinct values from.

    Returns:
        list: A list containing unique values from the specified column, or an empty list if the table is empty.
    """
    conn = None
    all_values = []
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
            f"Fetching existing values from column {column} in table {table_name.upper()}..."
        )
        cursor.execute(f"SELECT DISTINCT {column} FROM {table_name.upper()};")

        all_values = [row[0] for row in cursor.fetchall()]

    except Exception as e:
        logger.error(f"Error connecting to the database: {e}")
        raise e
    finally:
        if conn:
            conn.close()
            logger.debug("Database connection closed.")
        return all_values


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
