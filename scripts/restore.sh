#!/bin/bash

set -e

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PG_BACKUP_DIR="$PROJECT_DIR/backups/postgres"
CONFIG_BACKUP_DIR="$PROJECT_DIR/backups/config"
CONTAINER_NAME="pgvector_container"
DB_NAME="${POSTGRES_DB}"
DB_USER="${POSTGRES_USER}"
DATE=$(date +%Y%m%d)
LOG_FILE="$PROJECT_DIR/logs/restore_$DATE.log"

# Creating logs directory if it doesn't exist
mkdir -p "$PROJECT_DIR/logs"

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

# Check arguments and display usage
if [ -z "$1" ]; then
    log "ERROR" "Missing backup file argument"
    echo ""
    echo "Usage: $0 <backup_file.dump.gz> [config_file.tar.gz]"
    echo ""
    echo "Available database backups:"
    if [ -d "$PG_BACKUP_DIR" ]; then
        ls -la "$PG_BACKUP_DIR/"
    else
        echo "  No backup directory found: $PG_BACKUP_DIR"
    fi
    echo ""
    echo "Available configuration backups:"
    if [ -d "$CONFIG_BACKUP_DIR" ]; then
        ls -la "$CONFIG_BACKUP_DIR/"
    else
        echo "  No config backup directory found: $CONFIG_BACKUP_DIR"
    fi
    exit 1
fi

BACKUP_FILE="$1"
CONFIG_FILE="$2"

log "INFO" "========================================="
log "INFO" "Starting Mediatech restore process"
log "INFO" "Date: $(date '+%Y-%m-%d %H:%M:%S')"
log "INFO" "========================================="
log "INFO" "Database backup: $BACKUP_FILE"
if [ -n "$CONFIG_FILE" ]; then
    log "INFO" "Configuration backup: $CONFIG_FILE"
else
    log "INFO" "Configuration backup: None (database only)"
fi
log "INFO" "========================================="

# Load environment variables
cd "$PROJECT_DIR"
if [ ! -f .env ]; then
    log "ERROR" ".env file not found in $PROJECT_DIR"
    exit 1
fi

source .env
log "DEBUG" "Environment variables loaded from .env"
log "DEBUG" "Database: $DB_NAME, User: $DB_USER, Container: $CONTAINER_NAME"

# Verification checks
log "INFO" "========================================="
log "INFO" "Step 1: Pre-restore Verification"
log "INFO" "========================================="

# Check if backup file exists
if [ ! -f "$BACKUP_FILE" ]; then
    log "ERROR" "Backup file not found: $BACKUP_FILE"
    exit 1
fi
log "INFO" "Database backup file exists: $BACKUP_FILE"

# Check if config file exists (if provided)
if [ -n "$CONFIG_FILE" ]; then
    if [ ! -f "$CONFIG_FILE" ]; then
        log "ERROR" "Configuration backup file not found: $CONFIG_FILE"
        exit 1
    fi
    log "INFO" "Configuration backup file exists: $CONFIG_FILE"
fi

# Check Docker container
if ! docker ps --format "table {{.Names}}" | grep -q "$CONTAINER_NAME"; then
    log "ERROR" "PostgreSQL container '$CONTAINER_NAME' not found or not running"
    exit 1
fi
log "INFO" "PostgreSQL container is running"

# Confirmation prompt
log "WARNING" "========================================="
log "WARNING" "RESTORE CONFIRMATION REQUIRED"
log "WARNING" "========================================="
log "WARNING" "This will REPLACE the current database content!"
log "WARNING" "Database: $DB_NAME"
log "WARNING" "Backup: $BACKUP_FILE"
if [ -n "$CONFIG_FILE" ]; then
    log "WARNING" "Config: $CONFIG_FILE"
fi
log "WARNING" "========================================="

read -p "Are you sure you want to proceed? (yes/no): " -r
if [[ ! $REPLY =~ ^(yes|YES|y|Y)$ ]]; then
    log "INFO" "Restore operation cancelled by user"
    exit 0
fi

# Step 2: Configuration restore (if provided)
if [ -n "$CONFIG_FILE" ]; then
    log "INFO" "========================================="
    log "INFO" "Step 2: Configuration Files Restore"
    log "INFO" "========================================="
    log "INFO" "Restoring configuration from: $CONFIG_FILE"
    
    if tar -xzf "$CONFIG_FILE" --overwrite-dir 2>&1 | tee -a "$LOG_FILE"; then
        log "INFO" "Configuration files restored successfully"
        # Reload environment variables after restore
        source .env
        log "DEBUG" "Environment variables reloaded"
    else
        log "ERROR" "Failed to restore configuration files"
        exit 1
    fi
fi

# Step 3: Database restore
log "INFO" "========================================="
log "INFO" "Step 3: Database Restore"
log "INFO" "========================================="
log "INFO" "Starting database restore process..."

# Decompress backup file
log "DEBUG" "Decompressing backup file..."
TEMP_DUMP="/tmp/restore_$(basename "$BACKUP_FILE" .gz)"

if gunzip -c "$BACKUP_FILE" > "$TEMP_DUMP"; then
    log "INFO" "Backup file decompressed successfully"
    backup_size=$(du -h "$TEMP_DUMP" | cut -f1)
    log "DEBUG" "Decompressed size: $backup_size"
else
    log "ERROR" "Failed to decompress backup file"
    exit 1
fi

# Copy to container
log "DEBUG" "Copying backup to container..."
if docker cp "$TEMP_DUMP" "$CONTAINER_NAME:/tmp/"; then
    log "INFO" "Backup copied to container successfully"
else
    log "ERROR" "Failed to copy backup to container"
    rm -f "$TEMP_DUMP"
    exit 1
fi

# Perform restore
log "INFO" "Performing database restore..."
log "WARNING" "This may take several minutes depending on database size..."

if docker exec "$CONTAINER_NAME" pg_restore \
    -U "$DB_USER" \
    -d "$DB_NAME" \
    --verbose \
    --clean \
    --if-exists \
    "/tmp/$(basename "$TEMP_DUMP")" 2>&1 | tee -a "$LOG_FILE"; then
    
    log "INFO" "Database restore completed successfully"
else
    log "ERROR" "Database restore failed"
    # Cleanup on failure
    rm -f "$TEMP_DUMP"
    docker exec "$CONTAINER_NAME" rm -f "/tmp/$(basename "$TEMP_DUMP")" 2>/dev/null || true
    exit 1
fi

# Step 4: Cleanup
log "INFO" "========================================="
log "INFO" "Step 4: Cleanup Temporary Files"
log "INFO" "========================================="

# Remove local temp file
if rm -f "$TEMP_DUMP"; then
    log "DEBUG" "Local temporary file removed"
else
    log "WARNING" "Failed to remove local temporary file"
fi

# Remove container temp file
if docker exec "$CONTAINER_NAME" rm -f "/tmp/$(basename "$TEMP_DUMP")"; then
    log "DEBUG" "Container temporary file removed"
else
    log "WARNING" "Failed to remove container temporary file"
fi

# Step 5: Verification
log "INFO" "========================================="
log "INFO" "Step 5: Post-restore Verification"
log "INFO" "========================================="

# Check database connection
log "DEBUG" "Verifying database connection..."
if docker exec "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME" -c "SELECT version();" > /dev/null 2>&1; then
    log "INFO" "Database connection verified"
else
    log "WARNING" "Database connection verification failed"
fi

# Check if data_history.json was restored (if config restore was performed)
if [ -n "$CONFIG_FILE" ] && [ -f "config/data_history.json" ]; then
    log "INFO" "data_history.json file verified"
    log "DEBUG" "Last LEGI update: $(grep -o '"last_hf_upload_date": "[^"]*"' config/data_history.json | cut -d'"' -f4)"
else
    if [ -n "$CONFIG_FILE" ]; then
        log "WARNING" "data_history.json file not found after config restore"
    fi
fi

# Final report
log "INFO" "========================================="
log "INFO" "Step 6: Final Report"
log "INFO" "========================================="
log "INFO" "Restore process completed successfully!"
log "INFO" "Restored from:"
log "INFO" "  Database: $BACKUP_FILE ($(du -h "$BACKUP_FILE" | cut -f1))"
if [ -n "$CONFIG_FILE" ]; then
    log "INFO" "  Configuration: $CONFIG_FILE ($(du -h "$CONFIG_FILE" | cut -f1))"
fi
log "INFO" "Target:"
log "INFO" "  Database: $DB_NAME"
log "INFO" "  Container: $CONTAINER_NAME"
log "INFO" "Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
log "INFO" "========================================="