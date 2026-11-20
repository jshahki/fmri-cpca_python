mkdir -p logs

python step3a.py --create-manifest

N=$(python -c "
import json
with open('step3_run_manifest.json') as f:
    print(len(json.load(f)))
")

sbatch --array=0-$((N-1)) step3a_array.sh
