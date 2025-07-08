#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATE=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$PROJECT_DIR/logs/deployment_packages_$DATE.log"
APT_REQUIREMENTS_FILE="$PROJECT_DIR/config/requirements-apt.txt"

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
log "INFO" "Starting Albert Bibliothèque deployment packages setup"
log "INFO" "Date: $(date '+%Y-%m-%d %H:%M:%S')"
log "INFO" "========================================="

log "INFO" "========================================="
log "INFO" "Step 1: System Package Update"
log "INFO" "========================================="
log "INFO" "Updating package lists and installing necessary packages..."

sudo apt-get update -y

log "INFO" "========================================="
log "INFO" "Step 2: Essential Packages Installation"
log "INFO" "========================================="
log "INFO" "Installing essential packages from $APT_REQUIREMENTS_FILE..."

if [ ! -f "$APT_REQUIREMENTS_FILE" ]; then
    log "WARNING" "$APT_REQUIREMENTS_FILE not found, skipping package installation"
else
    # Browse through the requirements-apt.txt file and install each package
    while IFS= read -r package || [[ -n "$package" ]]; do
        # Skipping empty lines and comments
        if [[ -z "$package" || "$package" =~ ^[[:space:]]*# ]]; then
            continue
        fi
        
        if dpkg -s "$package" &> /dev/null; then
            log "DEBUG" "$package is already installed, skipping"
        else
            log "INFO" "Installing $package..."
            if sudo apt-get install -y "$package" 2>&1 | tee -a "$LOG_FILE"; then
                log "INFO" "$package installed successfully"
            else
                log "ERROR" "Failed to install $package"
                exit 1
            fi
        fi
    done < $APT_REQUIREMENTS_FILE
    log "INFO" "All packages from $APT_REQUIREMENTS_FILE processed"
fi

log "INFO" "========================================="
log "INFO" "Step 3: Additional installations excepted from apt requirements file"
log "INFO" "========================================="
log "INFO" "No packages for now. Skipping additional installations."


# Step 4: Permissions
log "INFO" "========================================="
log "INFO" "Step 4: Permissions"
log "INFO" "========================================="
log "INFO" "Configuring Docker permissions..."
if ! groups "$USER" | grep -q docker; then
    sudo usermod -aG docker "$USER"
    log "INFO" "User $USER added to docker group"
fi

sudo systemctl start docker || true
sudo systemctl enable docker || true

# Final report
log "INFO" "========================================="
log "INFO" "Step 5: Final Report"
log "INFO" "========================================="
log "INFO" "Deployment packages setup completed!"
log "INFO" "Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
log "INFO" "========================================="
