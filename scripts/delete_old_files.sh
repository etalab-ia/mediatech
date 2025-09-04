#!/bin/bash

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATE=$(date +%Y%m%d)

LOG_FOLDER="$PROJECT_DIR/logs"
mkdir -p "$LOG_FOLDER"

if [ "$RUNNING_IN_DOCKER" = "true" ]; then
    AIRFLOW_LOG_FOLDER="/opt/airflow/logs"
else
    AIRFLOW_LOG_FOLDER="$PROJECT_DIR/airflow_config/logs"
fi

PG_BACKUP_FOLDER="$PROJECT_DIR/backups/postgres"
CONFIG_BACKUP_FOLDER="$PROJECT_DIR/backups/config"
LOG_FILE="$PROJECT_DIR/logs/delete_old_files_$DATE.log"

# Load environment variables from .env file
export $(grep -v '^#' .env | xargs)

# Defining logging function
log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "$level" in
        "INFO")
            echo "[$timestamp] [INFO] $message" | tee -a "$LOG_FILE"
            ;;
        "DEBUG")
            echo "[$timestamp] [DEBUG] $message" | tee -a "$LOG_FILE"
            ;;
        "ERROR")
            echo "[$timestamp] [ERROR] $message" | tee -a "$LOG_FILE"
            ;;
        "WARNING")
            echo "[$timestamp] [WARNING] $message" | tee -a "$LOG_FILE"
            ;;
        *)
            echo "[$timestamp] [INFO] $level $message" | tee -a "$LOG_FILE"
            ;;
    esac
}

# Delete all files older than RETENTION_DAYS
if [ -d "$LOG_FOLDER" ]; then
	find "$LOG_FOLDER" -name "*.log" -mtime +$RETENTION_DAYS -delete
	log "INFO" "Old log files older than $RETENTION_DAYS days have been deleted from $LOG_FOLDER."
else
	log "WARNING" "Log folder $LOG_FOLDER does not exist." 
    # Note : It will not write anything to the log file as the log folder does not exist
fi

if [ -d "$AIRFLOW_LOG_FOLDER" ]; then
    find "$AIRFLOW_LOG_FOLDER" -name "*.log" -mtime +$RETENTION_DAYS -delete
    # Delete empty folders after deleting log files inside them
    find "$AIRFLOW_LOG_FOLDER" -type d -empty -delete
    log "INFO" "Old Airflow log files older than $RETENTION_DAYS days have been deleted from $AIRFLOW_LOG_FOLDER."
else
    log "WARNING" "Airflow log folder $AIRFLOW_LOG_FOLDER does not exist." 
fi

if [ -d "$PG_BACKUP_FOLDER" ]; then
    find "$PG_BACKUP_FOLDER" -type f -mtime +$RETENTION_DAYS -delete
    log "INFO" "Old PostgreSQL backup files older than $RETENTION_DAYS days have been deleted from $PG_BACKUP_FOLDER."
else
    log "WARNING" "PostgreSQL backup folder $PG_BACKUP_FOLDER does not exist."
fi

if [ -d "$CONFIG_BACKUP_FOLDER" ]; then
	find "$CONFIG_BACKUP_FOLDER" -type f -mtime +$RETENTION_DAYS -delete
	log "INFO" "Old configuration backup files older than $RETENTION_DAYS days have been deleted from $CONFIG_BACKUP_FOLDER."
else
	log "WARNING" "Configuration backup folder $CONFIG_BACKUP_FOLDER does not exist." 
fi

# Docker Cleanup
log "INFO" "Starting Docker cleanup..."
if command -v docker &> /dev/null; then
    # Clean up dangling build cache
    log "INFO" "Pruning Docker builder cache..."
    sudo docker builder prune -f

    log "INFO" "Docker cleanup finished."
else
    log "WARNING" "Docker command not found. Skipping Docker cleanup."
fi