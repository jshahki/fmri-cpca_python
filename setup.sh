REPO_DIR="/scratch/st-toddwood-1/$USER/fmri-cpca_python"

cd "$REPO_DIR"

module purge
module load gcc/7.5.0
module load python/3.12.7

python -m venv "$REPO_DIR/fmri_env"

source "$REPO_DIR/fmri_env/bin/activate"

python -m pip install --upgrade pip
python -m pip install numpy scipy pandas matplotlib nibabel nilearn
