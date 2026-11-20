#!/bin/sh

#SBATCH --job-name=step3b_subject
#SBATCH --output=logs/step3b_%A_%a.out
#SBATCH --error=logs/step3b_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1

# ============================================================
# STEP 3b: SUBJECT-LEVEL FIR REGRESSION
# ============================================================

echo "=========================================="
echo "STEP 3b: SUBJECT-LEVEL REGRESSION"
echo "=========================================="

echo "Array job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"

# Activate your Python environment if necessary
# source ~/my_environment/bin/activate

python step3b.py --index "$SLURM_ARRAY_TASK_ID"

echo "=========================================="
echo "STEP 3b COMPLETED"
echo "=========================================="
