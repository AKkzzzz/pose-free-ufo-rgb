#!/bin/bash

# ============================================================================
# ufo Experiment Setup Script
# ============================================================================
# Creates isolated experiment folders with configuration management
# Usage: ./create_experiment.sh <exp_name> <config_file> [--key value ...]
# ============================================================================

set -e  # Exit on error

# ============================================================================
# 1. ARGUMENT PARSING AND VALIDATION
# ============================================================================

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <exp_name> <config_file> [--key value ...]"
    echo ""
    echo "Arguments:"
    echo "  exp_name      Name for the experiment"
    echo "  config_file   Path to JSON configuration file"
    echo "  --key value   Optional CLI arguments to override config"
    echo ""
    echo "Example:"
    echo "  $0 my_exp configs/default.json --batch_size 16 --lr 0.001"
    exit 1
fi

EXP_NAME="$1"
CONFIG_FILE="$2"
shift 2  # Remove first two arguments, leaving CLI overrides

# Validate config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# ============================================================================
# 2. CREATE TIMESTAMPED EXPERIMENT FOLDER
# ============================================================================

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_DIR="${TIMESTAMP}_${EXP_NAME}"

echo "Creating experiment directory: $EXP_DIR"
mkdir -p "$EXP_DIR"

# ============================================================================
# 3. COPY PYTHON FILES RECURSIVELY
# ============================================================================

echo "Copying Python files..."

# Copy Python files from root directory only (non-recursive)
find . -maxdepth 1 -name "*.py" -exec cp {} "$EXP_DIR/" \; 2>/dev/null || true


# cp -r depth_anything_3 "$EXP_DIR"

# Copy Python files from specific directories recursively
for dir in preproc ufo third_party; do
    if [ -d "$dir" ]; then
        echo "Copying from $dir/..."
        rsync -av \
            --include='*/' \
            --include='*.py' \
            --exclude='*' \
            --exclude='__pycache__/' \
            --exclude='*.pyc' \
            "$dir/" "$EXP_DIR/$dir/" > /dev/null
    fi
done

echo "Python files copied successfully"

# ============================================================================
# 4. CREATE DATA SYMLINK
# ============================================================================

echo "Creating data symlink..."

# Create symlink pointing to ../data (relative to experiment folder)
cd "$EXP_DIR"
ln -s ../data data
cd ..

echo "Data symlink created: ${EXP_DIR}/data -> ../data"

# Create symlink pointing to ../ckpts (relative to experiment folder)
cd "$EXP_DIR"
ln -s ../ckpts ckpts
cd ..
echo "Checkpoint symlink created: ${EXP_DIR}/ckpts -> ../ckpts"

# copy debug bash script
cd "$EXP_DIR"
# cp ../debug.sh .
cd ..

# copy debug bash script
cd "$EXP_DIR"
# cp ../sync_tb.sh .
cd ..

# ============================================================================
# 5. MERGE CONFIGURATIONS USING EMBEDDED PYTHON
# ============================================================================

echo "Merging configurations..."

# Get absolute path to config file
ABS_CONFIG_FILE=$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")

# Build CLI args string for Python
CLI_ARGS=""
for arg in "$@"; do
    CLI_ARGS="$CLI_ARGS \"$arg\""
done

# Run embedded Python script to merge configurations
python3 - "$ABS_CONFIG_FILE" "$EXP_DIR/config.json" "$EXP_NAME" "$@" <<'PYTHON_SCRIPT'
import json
import sys

# Load base config
config_path = sys.argv[1]
output_path = sys.argv[2]
exp_name = sys.argv[3]

with open(config_path, 'r') as f:
    config = json.load(f)

# Always set exp_name from user input
config['exp_name'] = exp_name

# Parse CLI arguments (--key value format)
i = 4
while i < len(sys.argv):
    if sys.argv[i].startswith('--'):
        key = sys.argv[i][2:]  # Remove '--'
        if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('--'):
            value = sys.argv[i + 1]
            # Type inference
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif value.lower() == 'null':
                value = None
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # Keep as string
            config[key] = value
            i += 2
        else:
            # Boolean flag
            config[key] = True
            i += 1
    else:
        i += 1

# Save merged config
with open(output_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"Configuration merged and saved to: {output_path}")
PYTHON_SCRIPT

# ============================================================================
# 6. CREATE README WITH RUN INSTRUCTIONS
# ============================================================================

echo "Creating README.txt..."

# Build CLI overrides string for README
CLI_OVERRIDES=""
if [ "$#" -gt 0 ]; then
    CLI_OVERRIDES="CLI Overrides: $@"
else
    CLI_OVERRIDES="CLI Overrides: None"
fi

cat > "$EXP_DIR/README.txt" <<EOF
================================================================================
ufo Experiment: $EXP_NAME
================================================================================

Created: $TIMESTAMP
Original Config: $CONFIG_FILE
$CLI_OVERRIDES

================================================================================
DIRECTORY STRUCTURE
================================================================================

config.json         - Merged configuration (JSON + CLI overrides)
data/               - Symlink to ../data
*.py                - Python source files (copied from parent directory)

================================================================================
RUNNING THE EXPERIMENT
================================================================================

To run this experiment:

    cd $EXP_DIR
    python main_storm.py --config config.json

Or with additional overrides:

    cd $EXP_DIR
    python main_storm.py --config config.json --batch_size 16 --lr 0.001

================================================================================
NOTES
================================================================================

- All Python files have been copied to preserve the exact code state
- The data symlink points to ../data (relative to this folder)
- Configuration has been pre-merged with CLI arguments
- Results will be saved to work_dirs/ within this experiment folder

================================================================================
EOF

echo "README.txt created"

# ============================================================================
# 7. PRINT SUCCESS MESSAGE
# ============================================================================

echo ""
echo "========================================================================"
echo "Experiment setup complete!"
echo "========================================================================"
echo ""
echo "Experiment directory: $EXP_DIR"
echo "Configuration: $EXP_DIR/config.json"
echo ""
echo "To run the experiment:"
echo "  cd $EXP_DIR"
echo "  python main_storm.py --config config.json"
echo ""
echo "For more information, see: $EXP_DIR/README.txt"
echo "========================================================================"
