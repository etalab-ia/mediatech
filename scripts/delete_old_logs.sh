#!/bin/bash

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATE=$(date +%Y%m%d_%H%M%S)
LOG_FOLDER="$PROJECT_DIR/logs"
LOG_FILE="$PROJECT_DIR/logs/delete_old_logs_$DATE.log"
# Number of days to retain logs
RETENTION_DAYS=14

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

# Delete all log files older than RETENTION_DAYS
if [ -d "$LOG_FOLDER" ]; then
	find "$LOG_FOLDER" -name "*.log" -mtime +$RETENTION_DAYS -delete
	log "INFO" "Old log files older than $RETENTION_DAYS days have been deleted from $LOG_FOLDER."
else
	log "ERROR" "Log folder $LOG_FOLDER does not exist." 
    # Note : It will not write anything to the log file as the log folder does not exist
fi