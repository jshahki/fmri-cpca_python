#!/bin/bash
#SBATCH --job-name=step1
#SBATCH --account=st-toddwood-1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=logs/step1_%A_%a.out
#SBATCH --error=logs/step1_%A_%a.err

mkdir -p "logs"

# ============================================================
# CONFIGURATION
# ============================================================

REPO_DIR="/scratch/st-toddwood-1/$USER/fmri-cpca_python"

OUTPUT_DIR="$REPO_DIR/vectorized_output"

CODE_DIR="$REPO_DIR"

FILE_LIST="$REPO_DIR/nifti_file_list.txt"

MASK_FILE="$REPO_DIR/mask_used.img"

# ============================================================
# LOAD ENVIRONMENT
# ============================================================

cd "$REPO_DIR"

module purge
module load gcc/7.5.0
module load python/3.12.7

source ~/fmri_env/bin/activate

echo "Python version:"
python --version

python -c "import numpy as np; import nibabel as nib; print('NumPy:', np.__version__); print('Nibabel:', nib.__version__)"

# ============================================================
# GET FILE ASSOCIATED WITH ARRAY TASK
# ============================================================

TASK_ID=${SLURM_ARRAY_TASK_ID}

mapfile -t NIFTI_FILES < "$FILE_LIST"

INPUT_FILE="${NIFTI_FILES[$TASK_ID]}"

if [ -z "$INPUT_FILE" ] || [ ! -f "$INPUT_FILE" ]; then

    echo "ERROR: No valid input file for task $TASK_ID"

    exit 1

fi

# ============================================================
# IDENTIFY SUBJECT
# ============================================================

SUBJECT=$(basename "$(dirname "$INPUT_FILE")")

FILENAME=$(basename "$INPUT_FILE")

echo "=============================================="
echo "PROCESSING FMRI RUN"
echo "=============================================="

echo "Array task ID: $TASK_ID"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "Subject: $SUBJECT"
echo "Input file: $INPUT_FILE"

# ============================================================
# CREATE SUBJECT OUTPUT DIRECTORY
# ============================================================

mkdir -p "$OUTPUT_DIR/$SUBJECT"

# ============================================================
# RUN PYTHON VECTORIZATION
# ============================================================

python "$CODE_DIR/step1.py" \
    --data "$REPO_DIR/data" \
    --output "$OUTPUT_DIR" \
    --mask "$MASK_FILE" \
    --single-file "$INPUT_FILE"

# ============================================================
# COMPLETION
# ============================================================

echo ""
echo "=============================================="
echo "TASK COMPLETED SUCCESSFULLY"
echo "=============================================="

echo "Subject: $SUBJECT"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_DIR/$SUBJECT"
echo "Job ID: $SLURM_JOB_ID"
echo "Finished: $(date)"
