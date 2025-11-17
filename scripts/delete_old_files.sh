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
TCHAP_SCRIPT="$PROJECT_DIR/scripts/write_tchap_message.sh"

# Load environment variables from .env file
export $(grep -v '^#' .env | xargs)

# --- FUNCTIONS ---

# Notification function
send_tchap_notification() {
    local message="$1"
    if [ -f "$TCHAP_SCRIPT" ]; then
        bash "$TCHAP_SCRIPT" "$message\n"
    else
        log "WARNING" "Tchap script not found at $TCHAP_SCRIPT. Skipping notification."
    fi
}

notify_step() {
    local message="$1"
    log "INFO" "========================================="
    log "INFO" "$message"
    log "INFO" "========================================="
    send_tchap_notification "**$message**"
}

common_log() {
    local message="$1"
    log "INFO" "$message"
    send_tchap_notification "$message\n"
}
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

# --- MAIN ---

log "INFO" "========================================="
log "INFO" "Starting old files cleanup process"
log "INFO" "Date: $(date '+%Y-%m-%d %H:%M:%S')"
log "INFO" "========================================="

send_tchap_notification "# 🚀🧹 Cleanup - Starting process..."
send_tchap_notification "🕒 **Date:** $(date '+%Y-%m-%d %H:%M:%S')"

# 1. Delete all files older than RETENTION_DAYS
notify_step "📌 Step 1: Deleting old local log files"
if [ -d "$LOG_FOLDER" ]; then
	find "$LOG_FOLDER" -name "*.log" -mtime +$RETENTION_DAYS -delete
    log "INFO" "Old log files older than $RETENTION_DAYS days have been deleted from $LOG_FOLDER."
else
	log "WARNING" "Log folder $LOG_FOLDER does not exist." 
    # Note : It will not write anything to the log file as the log folder does not exist
fi

# 2. Delete old Airflow logs
notify_step "📌 Step 2: Deleting old Airflow log files"
if [ -d "$AIRFLOW_LOG_FOLDER" ]; then
    find "$AIRFLOW_LOG_FOLDER" -name "*.log" -mtime +$RETENTION_DAYS -delete
    # Delete empty folders after deleting log files inside them
    find "$AIRFLOW_LOG_FOLDER" -type d -empty -delete
    log "INFO" "Old Airflow log files older than $RETENTION_DAYS days have been deleted from $AIRFLOW_LOG_FOLDER."
else
    log "WARNING" "Airflow log folder $AIRFLOW_LOG_FOLDER does not exist." 
fi

# 3. Delete old PostgreSQL backups
notify_step "📌 Step 3: Deleting old PostgreSQL backup files"
if [ -d "$PG_BACKUP_FOLDER" ]; then
    find "$PG_BACKUP_FOLDER" -type f -mtime +$RETENTION_DAYS -delete
    log "INFO" "Old PostgreSQL backup files older than $RETENTION_DAYS days have been deleted from $PG_BACKUP_FOLDER."
else
    log "WARNING" "PostgreSQL backup folder $PG_BACKUP_FOLDER does not exist."
fi

# 4. Delete old configuration backups
notify_step "📌 Step 4: Deleting old configuration backup files"
if [ -d "$CONFIG_BACKUP_FOLDER" ]; then
	find "$CONFIG_BACKUP_FOLDER" -type f -mtime +$RETENTION_DAYS -delete
    log "INFO" "Old configuration backup files older than $RETENTION_DAYS days have been deleted from $CONFIG_BACKUP_FOLDER."
else
	log "WARNING" "Configuration backup folder $CONFIG_BACKUP_FOLDER does not exist." 
fi

# 5. Clean up Hugging Face cache
notify_step "📌 Step 5: Cleaning Hugging Face cache"
HF_CACHE_DIR="$HOME/.cache/huggingface/hub"
if [ -d "$HF_CACHE_DIR" ]; then
    log "INFO" "Removing Hugging Face cache directory: $HF_CACHE_DIR"
    if rm -rf "$HF_CACHE_DIR"; then
        log "INFO" "Hugging Face cache directory removed successfully."
    else
        log "ERROR" "Failed to remove Hugging Face cache directory."
        send_tchap_notification "### ❌ **ERROR: Failed to remove Hugging Face cache**"
    fi
else
    log "WARNING" "Hugging Face cache directory does not exist at $HF_CACHE_DIR."
fi

# 6. Docker Cleanup
notify_step "📌 Step 6: Docker Cleanup"
if command -v docker &> /dev/null; then
    # Clean up dangling build cache
    log "INFO" "Pruning Docker builder cache..."
    if docker builder prune -f 2>>"$LOG_FILE"; then
        log "INFO" "Docker builder cache pruned successfully."
    else
        log "ERROR" "Docker builder prune failed."
        send_tchap_notification "### ❌ **ERROR: Docker builder prune failed**"
        exit 1
    fi
else
    log "WARNING" "Docker command not found. Skipping Docker cleanup."
fi

log "INFO" "========================================="
log "INFO" "Cleanup process completed at $(date '+%Y-%m-%d %H:%M:%S')"
log "INFO" "========================================="

send_tchap_notification "#### ✅ **Cleanup process completed at $(date '+%Y-%m-%d %H:%M:%S')**"