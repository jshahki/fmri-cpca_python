#!/bin/bash

set -euo pipefail

# ============================================================
# CONFIGURATION
# ============================================================

REPO_DIR="/scratch/st-toddwood-1/$USER/fmri-cpca_python"

DATA_DIR="$REPO_DIR/data"

SLURM_SCRIPT="$REPO_DIR/step1.sh"

# ============================================================
# SETUP
# ============================================================

cd "$REPO_DIR"

mkdir -p logs
mkdir -p vectorized_output

echo "=============================================="
echo "DISCOVERING FMRI RUNS"
echo "=============================================="

# Find all NIfTI files recursively
mapfile -t NIFTI_FILES < <(
    find "$DATA_DIR" \
        -type f \
        -name "*.nii.gz" \
        | sort -V
)

N_FILES=${#NIFTI_FILES[@]}

if [ "$N_FILES" -eq 0 ]; then
    echo "ERROR: No .nii.gz files found in $DATA_DIR"
    exit 1
fi

echo "Found $N_FILES NIfTI files."

# ============================================================
# SAVE FILE LIST
# ============================================================

FILE_LIST="$REPO_DIR/nifti_file_list.txt"

printf '%s\n' "${NIFTI_FILES[@]}" > "$FILE_LIST"

echo "Saved file list: $FILE_LIST"

# ============================================================
# SUBMIT SLURM ARRAY
# ============================================================

echo ""
echo "Submitting SLURM array..."

sbatch \
    --array=0-$((N_FILES - 1)) \
    "$SLURM_SCRIPT"

echo ""
echo "=============================================="
echo "SUBMISSION COMPLETED"
echo "=============================================="

echo "Total fMRI runs submitted: $N_FILES"
