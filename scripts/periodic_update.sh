#!/bin/bash

# Daily automation script for Mediatech
# This script handles: git pull, virtual environment, dependencies, execution

# Fix environment for cron
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

set -e  # Stops the script on any error

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
DATE=$(date +%Y%m%d)
LOG_FILE="$PROJECT_DIR/logs/periodic_update_$DATE.log"
LOCK_FILE="$PROJECT_DIR/tmp/update.lock"

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
log "INFO" "========================================="
log "INFO" "Starting periodic update"
log "INFO" "Date: $(date '+%Y-%m-%d %H:%M:%S')"
log "INFO" "========================================="

# Going into the project directory
cd "$PROJECT_DIR"
log "DEBUG" "Working directory: $PROJECT_DIR"

# Check if another instance is running
if [ -f "$LOCK_FILE" ]; then
    log "ERROR" "Another instance of the periodic update script is already running."
    exit 1
else
    mkdir -p "$PROJECT_DIR/tmp"
    touch "$LOCK_FILE"
    log "DEBUG" "Lock file created: $LOCK_FILE"
fi

# Remove lock file on exit or interruption
trap 'rm -f "$LOCK_FILE"' EXIT INT TERM


# 1. Git pull in order to get the latest version
log "INFO" "========================================="
log "INFO" "Step 1: Git Repository Update"
log "INFO" "========================================="
log "INFO" "Getting the latest version from the repository..."

if git pull origin main 2>&1 | tee -a "$LOG_FILE"; then
    log "INFO" "Git pull successful"
else
    log "ERROR" "Failed to pull from git repository"
    exit 1
fi

# 2. Creating the virtual environment folder if it doesn't exist
log "INFO" "========================================="
log "INFO" "Step 2: Virtual Environment Setup"
log "INFO" "========================================="
if [ ! -d "$VENV_DIR" ]; then
    log "INFO" "Creating the virtual environment..."
    if python3 -m venv "$VENV_DIR"; then
        log "INFO" "Virtual environment created successfully"
    else
        log "ERROR" "Failed to create virtual environment"
        exit 1
    fi
else
    log "DEBUG" "Virtual environment already exists"
fi

# 3. Activating the virtual environment
log "INFO" "Activating the virtual environment"
if source "$VENV_DIR/bin/activate"; then
    log "DEBUG" "Virtual environment activated"
else
    log "ERROR" "Failed to activate virtual environment"
    exit 1
fi

# 4. Install/update dependencies
log "INFO" "========================================="
log "INFO" "Step 3: Dependencies Installation"
log "INFO" "========================================="
log "INFO" "Installing dependencies..."
if pip install -e . 2>&1 | tee -a "$LOG_FILE"; then
    log "INFO" "Dependencies installed successfully"
else
    log "ERROR" "Failed to install dependencies"
    deactivate
    exit 1
fi

# 5. Verifying the existence of the .env file
log "INFO" "========================================="
log "INFO" "Step 4: Environment Configuration Check"
log "INFO" "========================================="
if [ ! -f ".env" ]; then
    log "WARNING" ".env file not found"
else
    log "DEBUG" ".env file exists"
fi

# 6. Executing the update script
log "INFO" "========================================="
log "INFO" "Step 5: Update Script Execution"
log "INFO" "========================================="
log "INFO" "Executing the update script..."

if bash scripts/update.sh; then
    log "INFO" "Update script executed successfully"
else
    log "ERROR" "Failed to execute the update script"
    deactivate
    exit 1
fi

# 7. Deactivating the virtual environment
log "DEBUG" "Deactivating the virtual environment"
deactivate

# Final report
log "INFO" "========================================="
log "INFO" "Step 6: Final Report"
log "INFO" "========================================="
log "INFO" "Periodic update completed successfully!"
log "INFO" "Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
log "INFO" "========================================="
