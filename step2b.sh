#!/bin/bash
#SBATCH --job-name=step2b
#SBATCH --account=st-toddwood-1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=logs/step2b_%A_%a.out
#SBATCH --error=logs/step2b_%A_%a.err

mkdir -p logs

# Load Python
module purge
module load gcc/7.5.0
module load python/3.12.7

# Activate environment
source ~/fmri_env/bin/activate

# Move to working directory
cd $SLURM_SUBMIT_DIR

# Run vectorization
python step2b.py

echo "completed."
