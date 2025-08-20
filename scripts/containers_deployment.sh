#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATE=$(date +%Y%m%d)
LOG_FILE="$PROJECT_DIR/logs/containers_deployment_$DATE.log"
APT_REQUIREMENTS_FILE="$PROJECT_DIR/config/requirements-apt.txt"

# Create logs directory if it doesn't exist
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
log "INFO" "Starting Mediatech container deployment"
log "INFO" "Date: $(date '+%Y-%m-%d %H:%M:%S')"
log "INFO" "========================================="

# Check if docker-compose.yml exists
if [ ! -f "$PROJECT_DIR/docker-compose.yml" ]; then
    log "ERROR" "docker-compose.yml not found in $PROJECT_DIR"
    exit 1
fi

cd "$PROJECT_DIR"

# Step 1: Prepare persistence
log "INFO" "========================================="
log "INFO" "Step 1: Preparing data history persistence"
log "INFO" "========================================="
if [ ! -f config/data_history.json ]; then
    log "INFO" "Creating config/data_history.json for persistence"
    echo '{}' > config/data_history.json
else
    log "DEBUG" "config/data_history.json already exists"
fi

# Step 2: Build and deploy
log "INFO" "========================================="
log "INFO" "Step 2: Building and deploying containers"
log "INFO" "========================================="

# Vérifier si le container airflow-init existe déjà (existe et a tourné)
if sudo docker ps -a --filter "name=airflow-init" --format '{{.Names}}' | grep -q "^airflow-init$"; then
    log "INFO" "Container 'airflow-init' already exists. Skipping airflow-init step."
    log "INFO" "Running: docker compose up -d --build --remove-orphans"
    if sudo docker compose up -d --build --remove-orphans 2>&1 | tee -a "$LOG_FILE"; then
        log "INFO" "Container deployment successful"
    else
        log "ERROR" "Container deployment failed"
        exit 1
    fi
else
    log "INFO" "Container 'airflow-init' not found. Running initialization steps."
    log "INFO" "Running: docker compose build --no-cache"
    if sudo docker compose build --no-cache 2>&1 | tee -a "$LOG_FILE"; then
        log "INFO" "Build successful"
    else
        log "ERROR" "Build failed"
        exit 1
    fi

    log "INFO" "Running: docker compose up airflow-init"
    if sudo docker compose up airflow-init 2>&1 | tee -a "$LOG_FILE"; then
        log "INFO" "Airflow init successful"
    else
        log "ERROR" "Airflow init failed"
        exit 1
    fi

    log "INFO" "Running: docker compose up -d"
    if sudo docker compose up -d 2>&1 | tee -a "$LOG_FILE"; then
        log "INFO" "Containers started successfully"
    else
        log "ERROR" "Failed to start containers"
        exit 1
    fi
fi

# Step 3: Health check
log "INFO" "========================================="
log "INFO" "Step 3: Performing health checks"
log "INFO" "========================================="
log "INFO" "Waiting for services to be ready..."

# Wait for containers to be ready (more robust than sleep)
timeout 120 bash -c 'until sudo docker compose ps | grep -q "healthy\|running"; do sleep 2; done' || {
    log "ERROR" "Services failed to start within 120 seconds"
    sudo docker compose logs --tail=50 | tee -a "$LOG_FILE"
    exit 1
}

# Test CLI availability
log "INFO" "Testing mediatech CLI availability..."
if sudo docker compose exec -T airflow-scheduler mediatech --help >/dev/null 2>&1; then
    log "INFO" "Mediatech CLI is available and working"
else
    log "ERROR" "Mediatech CLI unavailable in container"
    log "ERROR" "Container logs:"
    sudo docker compose logs --tail=20 airflow-scheduler | tee -a "$LOG_FILE"
    exit 1
fi

# Step 4: Cleanup
log "INFO" "========================================="
log "INFO" "Step 4: Cleaning up unused images"
log "INFO" "========================================="
log "INFO" "Running: docker image prune -f"
if sudo docker image prune -f 2>&1 | tee -a "$LOG_FILE"; then
    log "INFO" "Image cleanup completed"
else
    log "WARNING" "Image cleanup had issues (non-critical)"
fi

# Step 5: Final report
log "INFO" "========================================="
log "INFO" "Step 5: Final report"
log "INFO" "========================================="
log "INFO" "Active containers:"
sudo docker compose ps | tee -a "$LOG_FILE"

log "INFO" "========================================="
log "INFO" "Container deployment completed successfully!"
log "INFO" "Services status: $(sudo docker compose ps --format 'table {{.Service}}\t{{.Status}}' | tail -n +2 | wc -l) services running"
log "INFO" "Full logs: $LOG_FILE"
log "INFO" "Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
log "INFO" "========================================="