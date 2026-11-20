REPO_DIR="/scratch/st-toddwood-1/$USER/fmri-cpca_python"

mkdir -p logs

source "$REPO_DIR/fmri_env/bin/activate"

python step3b.py --create-manifest

N=$(python -c "
import json
with open('step3b_subject_manifest.json') as f:
    print(len(json.load(f)))
")

echo "Number of subjects: $N"

sbatch --array=0-$((N-1)) step3b_array.sh
