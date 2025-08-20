FROM apache/airflow:3.0.3

# Default user for Airflow
ARG AIRFLOW_USER=airflow

# Install system dependencies if needed
USER root

# Copy the system requirements file first
COPY config/requirements-apt-container.txt /tmp/requirements-apt-container.txt

RUN apt-get update && \
    apt-get install -y $(grep -v '^#' /tmp/requirements-apt-container.txt | grep -v '^docker' | xargs) && \
    rm -rf /var/lib/apt/lists/* && \
    rm /tmp/requirements-apt-container.txt

# Back to the airflow user
USER ${AIRFLOW_USER}

ENV RUNNING_IN_DOCKER=true

# Copy the entire project (context = root now)
COPY . /tmp/mediatech/

# Go to the project directory
WORKDIR /tmp/mediatech

# Check that pyproject.toml is accessible
RUN ls -la pyproject.toml

# Change ownership of the directory to the user
USER root
RUN chown -R ${AIRFLOW_USER}:root /tmp/mediatech
USER ${AIRFLOW_USER}

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Back to the Airflow home directory
WORKDIR /opt/airflow

# Check that the command is available
RUN mediatech --help
