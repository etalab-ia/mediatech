import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Detect the environment (Docker or local)
if os.getenv("RUNNING_IN_DOCKER", "false").lower() == "true":
    # In Docker, using absolute paths
    base_path = "/tmp/mediatech"
    postgres_host = "postgres"
    postgres_port = "5432"
else:
    # Locally, using relative paths
    base_path = "."
    postgres_host = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port = os.getenv("POSTGRES_PORT", "5433")


# PostgreSQL configuration
POSTGRES_DB = os.getenv("POSTGRES_DB", "mediatech")
POSTGRES_USER = os.getenv("POSTGRES_USER", "user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
POSTGRES_HOST = postgres_host
POSTGRES_PORT = postgres_port

BASE_PATH = base_path

# Paths for configurations and data history
config_file_path = os.path.join(BASE_PATH, "config", "data_config.json")
data_history_path = os.path.join(BASE_PATH, "config", "data_history.json")

# Export folders
parquet_files_folder = os.path.join(BASE_PATH, "data", "parquet")


def get_env_variable_path(var_name: str, default_value: str = None):
    """
    Get an environment variable with a default value and construct its absolute path.

    Args:
        var_name (str): The name of the environment variable.
        default_value: The value to return if the variable is not set.

    Returns:
        str: The value of the environment variable or the default value.
    """
    path = os.getenv(var_name, default_value)
    return os.path.join(BASE_PATH, path)


# Data folders
CNIL_DATA_FOLDER = get_env_variable_path("CNIL_DATA_FOLDER", "data/unprocessed/cnil")
CONSTIT_DATA_FOLDER = get_env_variable_path(
    "CONSTIT_DATA_FOLDER", "data/unprocessed/constit"
)
LOCAL_ADMINISTRATIONS_DIRECTORY_FOLDER = get_env_variable_path(
    "LOCAL_ADMINISTRATIONS_DIRECTORY_FOLDER",
    "data/unprocessed/local_administrations_directory",
)
STATE_ADMINISTRATIONS_DIRECTORY_FOLDER = get_env_variable_path(
    "STATE_ADMINISTRATIONS_DIRECTORY_FOLDER",
    "data/unprocessed/state_administrations_directory",
)
DOLE_DATA_FOLDER = get_env_variable_path("DOLE_DATA_FOLDER", "data/unprocessed/dole")
LEGI_DATA_FOLDER = get_env_variable_path("LEGI_DATA_FOLDER", "data/unprocessed/legi")
TRAVAIL_EMPLOI_DATA_FOLDER = get_env_variable_path(
    "TRAVAIL_EMPLOI_DATA_FOLDER", "data/unprocessed/travail_emploi"
)
SERVICE_PUBLIC_PRO_DATA_FOLDER = get_env_variable_path(
    "SERVICE_PUBLIC_PRO_DATA_FOLDER",
    "data/unprocessed/service_public_pro",
)
SERVICE_PUBLIC_PART_DATA_FOLDER = get_env_variable_path(
    "SERVICE_PUBLIC_PART_DATA_FOLDER",
    "data/unprocessed/service_public_part",
)
DATA_GOUV_DATASETS_CATALOG_DATA_FOLDER = get_env_variable_path(
    "DATA_GOUV_DATASETS_CATALOG_DATA_FOLDER",
    "data/unprocessed/data_gouv_datasets_catalog",
)

# OpenAI API configuration
API_URL = os.getenv("API_URL", "https://albert.api.etalab.gouv.fr/v1")
API_KEY = os.getenv("API_KEY", "your_api_key_here")

# Hugging Face configuration
HF_TOKEN = os.getenv("HF_TOKEN", "your_hugging_face_token_here")
