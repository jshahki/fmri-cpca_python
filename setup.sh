cd /scratch/st-toddwood-1/$USER/fmri-cpca_python

module purge
module load gcc/7.5.0
module load python/3.12.7

python -m venv ~/fmri_env

source ~/fmri_env/bin/activate

python -m pip install --upgrade pip
python -m pip install numpy scipy pandas matplotlib nibabel nilearn
