import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# PostgreSQL configuration
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5433")
POSTGRES_DB = os.getenv("POSTGRES_DB", "mediatech")
POSTGRES_USER = os.getenv("POSTGRES_USER", "user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")

# Paths for configurations and data history
config_file_path = "config/data_config.json"
data_history_path = "config/data_history.json"

# Export folders
parquet_files_folder = "data/parquet"

# Data folders
CNIL_DATA_FOLDER = os.getenv("CNIL_DATA_FOLDER", "data/unprocessed/cnil")
CONSTIT_DATA_FOLDER = os.getenv("CONSTIT_DATA_FOLDER", "data/unprocessed/constit")
LOCAL_ADMINISTRATIONS_DIRECTORY_FOLDER = os.getenv(
    "LOCAL_ADMINISTRATIONS_DIRECTORY_FOLDER",
    "data/unprocessed/local_administrations_directory",
)
STATE_ADMINISTRATIONS_DIRECTORY_FOLDER = os.getenv(
    "STATE_ADMINISTRATIONS_DIRECTORY_FOLDER",
    "data/unprocessed/state_administrations_directory",
)
DOLE_DATA_FOLDER = os.getenv("DOLE_DATA_FOLDER", "data/unprocessed/dole")
LEGI_DATA_FOLDER = os.getenv("LEGI_DATA_FOLDER", "data/unprocessed/legi")
TRAVAIL_EMPLOI_DATA_FOLDER = os.getenv(
    "TRAVAIL_EMPLOI_DATA_FOLDER", "data/unprocessed/travail_emploi"
)
SERVICE_PUBLIC_PRO_DATA_FOLDER = os.getenv(
    "SERVICE_PUBLIC_PRO_DATA_FOLDER",
    "data/unprocessed/service_public_pro",
)
SERVICE_PUBLIC_PART_DATA_FOLDER = os.getenv(
    "SERVICE_PUBLIC_PART_DATA_FOLDER",
    "data/unprocessed/service_public_part",
)
DATA_GOUV_DATASETS_CATALOG_DATA_FOLDER = os.getenv(
    "DATA_GOUV_DATASETS_CATALOG_DATA_FOLDER",
    "data/unprocessed/data_gouv_datasets_catalog",
)

# OpenAI API configuration
API_URL = os.getenv("API_URL", "https://albert.api.etalab.gouv.fr/v1")
API_KEY = os.getenv("API_KEY", "your_api_key_here")

# Hugging Face configuration
HF_TOKEN = os.getenv("HF_TOKEN", "your_hugging_face_token_here")
