#!/bin/sh

#SBATCH --job-name=step3a_GZ
#SBATCH --output=logs/step3a_%A_%a.out
#SBATCH --error=logs/step3a_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --account=st-toddwood-1

# ============================================================
# STAGE 1: RUN-LEVEL GZ CALCULATIONS
# ============================================================

# Load Python environment

REPO_DIR="/scratch/st-toddwood-1/$USER/fmri-cpca_python"

source "$REPO_DIR/fmri_env/bin/activate"

echo "=========================================="
echo "STEP 3a: RUN-LEVEL GZ CALCULATION"
echo "=========================================="

echo "Array job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"

python step3a.py --index "$SLURM_ARRAY_TASK_ID"

echo "=========================================="
echo "STEP 3a COMPLETED"
echo "=========================================="
