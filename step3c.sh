#!/bin/sh
#SBATCH --job-name=step3c
#SBATCH --output=step3c_%j.out
#SBATCH --error=step3c_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

module purge

# Activate your Python environment.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fmri_env

cd /scratch/st-toddwood-1/jshahki/fmri-cpca_python

echo "Starting Step 3c..."
echo "Date: $(date)"

python step3c.py

echo "Step 3c finished."
echo "Date: $(date)"
