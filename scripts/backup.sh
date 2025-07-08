#!/bin/bash

set -e

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PG_BACKUP_DIR="$PROJECT_DIR/backups/postgres"
CONFIG_BACKUP_DIR="$PROJECT_DIR/backups/config"
CONTAINER_NAME="pgvector_container"
DB_NAME="${POSTGRES_DB}"
DB_USER="${POSTGRES_USER}"
RETENTION_DAYS=7 # Number of days to retain logs
DATE=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$PROJECT_DIR/logs/backup_$DATE.log"

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
# Create backup directories
mkdir -p "$PG_BACKUP_DIR" "$CONFIG_BACKUP_DIR"

# Load environment variables
cd $PROJECT_DIR
source .env

log "INFO" "========================================="
log "INFO" "Starting Albert Bibliothèque backup process"
log "INFO" "Date: $(date '+%Y-%m-%d %H:%M:%S')"
log "INFO" "========================================="

# 1. PostgreSQL backup
log "INFO" "========================================="
log "INFO" "Step 1: PostgreSQL Database Backup"
log "INFO" "========================================="
if docker exec "$CONTAINER_NAME" pg_dump \
    -U "$DB_USER" \
    -d "$DB_NAME" \
    --verbose \
    --format=custom \
    --compress=9 \
    --file="/tmp/pg_backup_$DATE.dump" 2>>"$LOG_FILE"; then
    
    log "INFO" "Database dump completed successfully"
else
    log "ERROR" "Database dump failed"
    exit 1
fi

# Copy backup out of container
log "INFO" "Copying backup from container to host..."
if docker cp "$CONTAINER_NAME:/tmp/pg_backup_$DATE.dump" "$PG_BACKUP_DIR/"; then
    log "INFO" "Backup successfully copied to $PG_BACKUP_DIR/"
else
    log "ERROR" "Failed to copy backup from container"
    exit 1
fi

# Clean up inside container
docker exec "$CONTAINER_NAME" rm -f "/tmp/pg_backup_$DATE.dump"
log "INFO" "Temporary files cleaned from container"

# 2. Critical configuration files backup
log "INFO" "========================================="
log "INFO" "Step 2: Configuration Files Backup"
log "INFO" "========================================="

CONFIG_ARCHIVE="$CONFIG_BACKUP_DIR/config_backup_$DATE.tar.gz"
log "INFO" "Creating configuration archive..."
tar -czf "$CONFIG_ARCHIVE" \
    config/data_history.json \
    .env \
    docker-compose.yml \
    pyproject.toml \
    2>/dev/null || log "WARNING : Some config files missing"

# 3. Final PostgreSQL compression
log "INFO" "========================================="
log "INFO" "Step 3: Final Compression"
log "INFO" "========================================="
gzip "$PG_BACKUP_DIR/pg_backup_$DATE.dump"

# 4. Clean up old backups
log "INFO" "========================================="
log "INFO" "Step 4: Cleanup Old Backups"
log "INFO" "========================================="
find "$PG_BACKUP_DIR" -name "pg_backup_*.dump.gz" -mtime +$RETENTION_DAYS -delete
find "$CONFIG_BACKUP_DIR" -name "config_backup_*.tar.gz" -mtime +$RETENTION_DAYS -delete

# 5. Final report
log "INFO" "========================================="
log "INFO" "Step 5: Final Report"
log "INFO" "========================================="
log "INFO" "Backup completed:"
log "  DB: $PG_BACKUP_DIR/pg_backup_$DATE.dump.gz ($(du -h "$PG_BACKUP_DIR/pg_backup_$DATE.dump.gz" | cut -f1))"
log "  Config: $CONFIG_ARCHIVE ($(du -h "$CONFIG_ARCHIVE" | cut -f1))"
log ""
log "INFO" "Available backups:"
log "  Database:"
ls -lh "$PG_BACKUP_DIR/" | tail -n 5
log "  Configuration:"
ls -lh "$CONFIG_BACKUP_DIR/" | tail -n 5

log "INFO" "========================================="
log "INFO" "Backup process completed at $(date '+%Y-%m-%d %H:%M:%S')"
log "INFO" "========================================="