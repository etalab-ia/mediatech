#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATE=$(date +%Y%m%d)
LOG_FILE="$PROJECT_DIR/logs/initial_deployment_$DATE.log"

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
log "INFO" "Starting Mediatech initial deployment setup"
log "INFO" "Date: $(date '+%Y-%m-%d %H:%M:%S')"
log "INFO" "========================================="

# Step 1: System Package Update
log "INFO" "========================================="
log "INFO" "Step 1: System Package Update"
log "INFO" "========================================="
log "INFO" "Updating package lists and installing necessary packages..."

log "INFO" "Running sudo apt update..."
if sudo apt update -y 2>&1 | tee -a "$LOG_FILE"; then
    log "INFO" "apt update completed successfully"
else
    log "ERROR" "apt update failed"
    exit 1
fi

log "INFO" "Running sudo apt upgrade..."
if sudo apt upgrade -y 2>&1 | tee -a "$LOG_FILE"; then
    log "INFO" "apt upgrade completed successfully"
else
    log "ERROR" "apt upgrade failed"
    exit 1
fi

# Step 2: Install essential packages
log "INFO" "========================================="
log "INFO" "Step 2: Essential Packages Installation"
log "INFO" "========================================="
log "INFO" "Installing essential packages..."

if sudo apt install -y ca-certificates curl gnupg lsb-release apt-transport-https 2>&1 | tee -a "$LOG_FILE"; then
    log "INFO" "Essential packages installed successfully"
else
    log "ERROR" "Failed to install essential packages"
    exit 1
fi

# Step 3: GPG Key and Repository Setup
log "INFO" "========================================="
log "INFO" "Step 3: GPG Key and Repository Setup"
log "INFO" "========================================="
log "INFO" "Setting up Docker GPG key and repository..."

sudo mkdir -p /e

if curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg; then
    log "INFO" "Docker GPG key added successfully"
else
    log "ERROR" "Failed to add Docker GPG key"
    exit 1
fi

# Step 4: Docker Repository Configuration
log "INFO" "========================================="
log "INFO" "Step 4: Docker Repository Configuration"
log "INFO" "========================================="
log "INFO" "Configuring Docker repository..."

echo \
"deb [arch=$(dpkg --print-architecture) \
signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu \
$(lsb_release -cs) stable" | \
sudo tee /etc/apt/sources.list.d/docker.list | tee -a "$LOG_FILE" > /dev/null

sudo apt update -y 2>&1 | tee -a "$LOG_FILE"

# Step 5: Docker Installation
log "INFO" "=========================================="
log "INFO" "Step 5: Docker Installation"
log "INFO" "=========================================="
log "INFO" "Installing Docker and Docker Compose..."
if sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin 2>&1 | tee -a "$LOG_FILE"; then
    log "INFO" "Docker and Docker Compose installed successfully"
else
    log "ERROR" "Failed to install Docker and Docker Compose"
    exit 1
fi

# Step 6: Permissions
log "INFO" "========================================="
log "INFO" "Step 6: Permissions"
log "INFO" "========================================="
log "INFO" "Configuring Docker permissions..."
if ! groups "$USER" | grep -q docker; then
    sudo usermod -aG docker "$USER"
    log "INFO" "User $USER added to docker group"
else
    log "DEBUG" "User $USER is already in the docker group"
fi

sudo systemctl start docker
sudo systemctl enable docker

# Step 7: Final report
log "INFO" "========================================="
log "INFO" "Step 7: Final Report"
log "INFO" "========================================="
log "INFO" "Deployment packages setup completed!"
docker --version | tee -a "$LOG_FILE"
docker compose version | tee -a "$LOG_FILE"
log "INFO" "Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
log "INFO" "========================================="
