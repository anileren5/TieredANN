#!/bin/bash

# Script to build pgvector index from binary data files

# Change to project root
cd "$(dirname "$0")/.." || exit 1

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
DATASET="sift"
DATA_PATH="data/$DATASET/${DATASET}_base.bin"

# ============================================================================
# POSTGRESQL CONFIGURATION - UPDATE THESE VALUES FOR YOUR SETUP
# ============================================================================
# PostgreSQL connection settings
# For Docker: Use service name "postgres" and default credentials
# For local: Use "localhost" and your PostgreSQL credentials
DB_HOST="${DB_HOST:-localhost}"  # Default: localhost (use "postgres" for Docker)
DB_PORT="${DB_PORT:-5432}"       # Default PostgreSQL port
DB_NAME="${DB_NAME:-postgres}"   # Default database name
DB_USER="${DB_USER:-postgres}"   # Default user
DB_PASSWORD="${DB_PASSWORD:-postgres}"  # Default password

# Table name for storing vectors
TABLE_NAME="vectors"  # PostgreSQL table name

# Detect if running inside Docker and set PostgreSQL host accordingly
if [ -f /.dockerenv ] || [ -n "$DOCKER_CONTAINER" ]; then
    DB_HOST="postgres"  # Docker internal network
fi

# Options
RECREATE=true  # Set to true to recreate table even if it exists

# Check if data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file not found: $DATA_PATH"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Add python directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/python"

# Build pgvector index
echo "Building pgvector index..."
echo "Data file: $DATA_PATH"
echo "Table name: $TABLE_NAME"
echo "PostgreSQL host: $DB_HOST"
echo "PostgreSQL port: $DB_PORT"
echo "Database: $DB_NAME"
echo "User: $DB_USER"

RECREATE_FLAG=""
if [ "$RECREATE" = true ]; then
    RECREATE_FLAG="--recreate"
    echo "Recreating table..."
fi

python3 python/build_pgvector_index.py \
    --data_path "$DATA_PATH" \
    --table_name "$TABLE_NAME" \
    --db_host "$DB_HOST" \
    --db_port "$DB_PORT" \
    --db_name "$DB_NAME" \
    --db_user "$DB_USER" \
    --db_password "$DB_PASSWORD" \
    $RECREATE_FLAG

echo ""
echo "pgvector index build completed!"



