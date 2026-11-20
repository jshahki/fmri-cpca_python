import argparse
import json
import re
import sys
import time
import traceback

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

G_DIR = Path("/scratch/st-toddwood-1/jshahki/fmri-cpca_python/fir_output")

GZ_DIR = Path("/scratch/st-toddwood-1/jshahki/fmri-cpca_python/step3_output")

OUTPUT_DIR = Path("/scratch/st-toddwood-1/jshahki/fmri-cpca_python/step3b_output")

MANIFEST_FILE = Path("/scratch/st-toddwood-1/jshahki/fmri-cpca_python/step3b_subject_manifest.json")

DTYPE = np.float64

# Number of voxels processed per GC multiplication.
# Reduce this if memory is limited.
CHUNK_SIZE = 8192

# Numerical tolerance for matrix validation.
TOLERANCE = 1e-8


# ============================================================
# 1. NATURAL SORTING
# ============================================================

def natural_sort(items):

    def sort_key(item):

        return [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(item))
        ]

    return sorted(items, key=sort_key)


# ============================================================
# 2. SUBJECT DISCOVERY
# ============================================================

def discover_subjects():

    if not G_DIR.exists():

        raise FileNotFoundError(
            f"G directory not found: {G_DIR}"
        )

    if not GZ_DIR.exists():

        raise FileNotFoundError(
            f"GZ directory not found: {GZ_DIR}"
        )

    subjects = natural_sort([
        p.name
        for p in G_DIR.iterdir()
        if p.is_dir()
    ])

    if not subjects:

        raise ValueError(
            f"No subject directories found in {G_DIR}"
        )

    return subjects


# ============================================================
# 3. FIND SUBJECT-LEVEL G AND gg FILES
# ============================================================

def find_subject_file(subject_dir, suffix):

    candidates = [
        subject_dir / f"{subject_dir.name}_{suffix}.npy",
        subject_dir / f"{suffix}.npy"
    ]

    for candidate in candidates:

        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find subject-level {suffix} file "
        f"for {subject_dir.name}.\n"
        f"Checked:\n" +
        "\n".join(str(p) for p in candidates)
    )


def load_subject_G(subject_dir):

    G_file = find_subject_file(
        subject_dir,
        "G"
    )

    gg_file = find_subject_file(
        subject_dir,
        "gg"
    )

    G = np.load(
        G_file,
        mmap_mode="r"
    )

    gg = np.load(
        gg_file
    )

    if G.ndim != 2:

        raise ValueError(
            f"Subject G must be 2D: {G_file}"
        )

    if gg.ndim != 2:

        raise ValueError(
            f"Subject gg must be 2D: {gg_file}"
        )

    if G.shape[1] != gg.shape[0]:

        raise ValueError(
            f"Incompatible G and gg dimensions for "
            f"{subject_dir.name}: "
            f"G={G.shape}, gg={gg.shape}"
        )

    print(f"\nSubject G: {G.shape}")
    print(f"Subject gg: {gg.shape}")

    return G, gg, G_file, gg_file



# ============================================================
# 4. FIND GZ FILES AND EXTRACT BIDS ENTITIES
# ============================================================

def extract_bids_entities(filename):

    name = Path(filename).stem

    entities = {}

    for entity in ["sub", "ses", "task", "run"]:

        match = re.search(
            rf"(?:^|_){entity}-([^_]+)",
            name
        )

        entities[entity] = (
            match.group(1)
            if match
            else None
        )

    return entities


def find_GZ_files(subject):

    subject_dir = GZ_DIR / subject

    if not subject_dir.exists():

        raise FileNotFoundError(
            f"GZ subject directory not found: {subject_dir}"
        )

    files = natural_sort(
        list(subject_dir.glob("*_GZ.npy"))
    )

    if not files:

        raise FileNotFoundError(
            f"No GZ matrices found for {subject}"
        )

    print(
        f"\nFound {len(files)} GZ matrices for {subject}"
    )

    for f in files:

        entities = extract_bids_entities(f.name)

        if entities["sub"] != subject.replace("sub-", ""):

            raise ValueError(
                f"Subject mismatch: {f.name}"
            )

        print(f"  {f.name}")

    return files


def extract_run_suffix(filename):

    entities = extract_bids_entities(filename)

    task_suffix = (
        f"task-{entities['task']}"
        if entities["task"] is not None
        else None
    )

    # Preserve the original filename.
    # Use run 1 internally when no run label exists.

    run_number = (
        int(entities["run"])
        if entities["run"] is not None
        else 1
    )

    return task_suffix, run_number




# ============================================================
# 5. LOAD MATCHING RUN-LEVEL G MATRIX
# ============================================================

def find_run_G(
    subject_dir,
    task_suffix,
    run_number,
    gz_file=None
):

    if gz_file is None:

        raise ValueError(
            "GZ filename is required for matching G."
        )

    gz_entities = extract_bids_entities(
        gz_file.name
    )

    candidates = natural_sort(
        list(subject_dir.glob("*_G.npy"))
    )

    matches = []

    for candidate in candidates:

        g_entities = extract_bids_entities(
            candidate.name
        )

        # Match only entities that exist in the GZ filename.
        #
        # Missing entities must also be missing in G.
        # No labels are added or inferred.

        if all(
            g_entities[key] == gz_entities[key]
            for key in ["sub", "ses", "task", "run"]
        ):

            matches.append(candidate)

    if len(matches) == 1:

        return matches[0]

    if len(matches) == 0:

        raise FileNotFoundError(
            f"No matching G matrix found for:\n"
            f"{gz_file.name}"
        )

    raise ValueError(
        f"Multiple matching G matrices found:\n"
        + "\n".join(str(f) for f in matches)
    )



# ============================================================
# 6. LOAD GZ MATRIX
# ============================================================

def load_GZ(filename):

    GZ = np.load(
        filename,
        mmap_mode="r"
    )

    if GZ.ndim != 2:

        raise ValueError(
            f"GZ must be 2D: {filename}"
        )

    if not np.all(np.isfinite(GZ)):

        raise ValueError(
            f"GZ contains non-finite values: {filename}"
        )

    return GZ


# ============================================================
# 7. MATRIX MULTIPLICATION: B AND C
# ============================================================

def calculate_B_C(gg, GZ):

    # --------------------------------------------------------
    # MATRIX OPERATION 1:
    #
    # B_R = gg @ GZ_R
    #
    # MATLAB:
    # B_R = gg * GZ_R
    # --------------------------------------------------------

    B = gg @ GZ

    if not np.all(np.isfinite(B)):

        raise ValueError(
            "B matrix contains non-finite values."
        )

    # --------------------------------------------------------
    # MATRIX OPERATION 2:
    #
    # C_R = gg @ B_R
    #
    # MATLAB:
    # C_R = gg * B_R
    # --------------------------------------------------------

    C = gg @ B

    if not np.all(np.isfinite(C)):

        raise ValueError(
            "C matrix contains non-finite values."
        )

    return B, C


# ============================================================
# 8. SAVE NUMPY MATRIX
# ============================================================

def save_matrix(filename, matrix):

    np.save(
        filename,
        np.asarray(matrix)
    )


# ============================================================
# 9. INITIALIZE SUBJECT PROCESSING
# ============================================================

def initialize_subject(subject):

    subject_dir = G_DIR / subject

    output_subject_dir = OUTPUT_DIR / subject

    output_subject_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    G, gg, G_file, gg_file = load_subject_G(
        subject_dir
    )

    GZ_files = find_GZ_files(subject)

    print("\n")
    print("=" * 65)
    print(f"PROCESSING SUBJECT: {subject}")
    print("=" * 65)

    print(f"Number of runs: {len(GZ_files)}")

    return {
        "subject": subject,
        "subject_dir": subject_dir,
        "output_dir": output_subject_dir,
        "G": G,
        "gg": gg,
        "G_file": G_file,
        "gg_file": gg_file,
        "GZ_files": GZ_files
    }


# ============================================================
# 10. PROCESS RUN-LEVEL B AND C
# ============================================================

def process_run_B_C(
    subject,
    run_number,
    task_suffix,
    GZ_file,
    gg,
    output_dir
):

    print(
        f"\nProcessing run {run_number}"
    )

    print(
        f"GZ file: {GZ_file.name}"
    )

    GZ = load_GZ(GZ_file)

    # --------------------------------------------------------
    # Calculate B_R and C_R
    # --------------------------------------------------------

    B, C = calculate_B_C(
        gg,
        GZ
    )

    prefix = (
        f"{subject}_{task_suffix}_run-{run_number}"
        if task_suffix
        else f"{subject}_run-{run_number}"
    )

    B_file = output_dir / f"{prefix}_B_R.npy"

    C_file = output_dir / f"{prefix}_C_R.npy"

    save_matrix(
        B_file,
        B
    )

    save_matrix(
        C_file,
        C
    )

    print(f"B dimensions: {B.shape}")
    print(f"C dimensions: {C.shape}")

    print(f"Saved: {B_file.name}")
    print(f"Saved: {C_file.name}")

    result = {
        "run_number": run_number,
        "task_suffix": task_suffix,
        "GZ_file": str(GZ_file),
        "B_file": str(B_file),
        "C_file": str(C_file),
        "B": B,
        "C": C,
        "GZ_shape": GZ.shape
    }

    del GZ

    return result


# ============================================================
# 11. ACCUMULATE SUBJECT-LEVEL B AND C
# ============================================================

def accumulate_subject_B_C(
    subject,
    run_results,
    output_dir
):

    print("\n")
    print("=" * 65)
    print("PASS 1: SUBJECT-LEVEL B AND C ACCUMULATION")
    print("=" * 65)

    B_subject = None
    C_subject = None

    for result in run_results:

        B = result["B"]
        C = result["C"]

        # ----------------------------------------------------
        # MATRIX OPERATION 3:
        #
        # B_S = sum(B_R)
        #
        # MATLAB:
        # B_S = B_S + B_R
        # ----------------------------------------------------

        if B_subject is None:

            B_subject = B.copy()

        else:

            if B_subject.shape != B.shape:

                raise ValueError(
                    f"Incompatible B dimensions for {subject}"
                )

            B_subject += B

        # ----------------------------------------------------
        # MATRIX OPERATION 4:
        #
        # C_S = sum(C_R)
        #
        # MATLAB:
        # C_S = C_S + C_R
        # ----------------------------------------------------

        if C_subject is None:

            C_subject = C.copy()

        else:

            if C_subject.shape != C.shape:

                raise ValueError(
                    f"Incompatible C dimensions for {subject}"
                )

            C_subject += C

    if B_subject is None or C_subject is None:

        raise ValueError(
            f"No valid runs found for {subject}"
        )

    B_file = output_dir / f"{subject}_B.npy"

    C_file = output_dir / f"{subject}_C.npy"

    save_matrix(
        B_file,
        B_subject
    )

    save_matrix(
        C_file,
        C_subject
    )

    print(f"\nSubject B: {B_subject.shape}")
    print(f"Subject C: {C_subject.shape}")

    print(f"Saved: {B_file.name}")
    print(f"Saved: {C_file.name}")

    return B_subject, C_subject


# ============================================================
# 12. CHUNKED GC CALCULATION
# ============================================================

def calculate_GC_chunked(
    subject,
    G_run,
    C_subject,
    output_file,
    chunk_size=CHUNK_SIZE
):

    """
    Calculate:

        GC_R = G_R @ C_subject

    MATLAB equivalent:

        GC_R = G_R * C_S

    The calculation is performed in voxel chunks.

    The complete GC matrix is stored on disk using
    NumPy memory mapping.
    """

    nscans = G_run.shape[0]

    ncolumns = G_run.shape[1]

    nvoxels = C_subject.shape[1]

    if ncolumns != C_subject.shape[0]:

        raise ValueError(
            f"Incompatible G and C dimensions for {subject}: "
            f"G={G_run.shape}, C={C_subject.shape}"
        )

    print(
        f"\nGC dimensions: ({nscans}, {nvoxels})"
    )

    print(
        f"Voxel chunk size: {chunk_size}"
    )

    # --------------------------------------------------------
    # Create disk-backed GC matrix
    # --------------------------------------------------------

    GC = np.lib.format.open_memmap(
        output_file,
        mode="w+",
        dtype=DTYPE,
        shape=(nscans, nvoxels)
    )

    run_sos = 0.0

    # --------------------------------------------------------
    # MATRIX OPERATION 5:
    #
    # GC_R = G_R @ C_subject
    #
    # Perform multiplication in voxel chunks.
    # --------------------------------------------------------

    for start in range(
        0,
        nvoxels,
        chunk_size
    ):

        end = min(
            start + chunk_size,
            nvoxels
        )

        G_chunk = np.asarray(
            G_run,
            dtype=DTYPE
        )

        C_chunk = np.asarray(
            C_subject[:, start:end],
            dtype=DTYPE
        )

        GC_chunk = G_chunk @ C_chunk

        if not np.all(np.isfinite(GC_chunk)):

            raise ValueError(
                f"Non-finite GC values for {subject}, "
                f"voxels {start}:{end}"
            )

        GC[:, start:end] = GC_chunk

        # ----------------------------------------------------
        # MATRIX OPERATION 6:
        #
        # SSQ_R = sum(GC_R ** 2)
        #
        # MATLAB equivalent:
        #
        # sum(diag(GC_R(:,v) * GC_R(:,v)'))
        # summed over all voxels.
        # ----------------------------------------------------

        run_sos += float(
            np.sum(
                GC_chunk ** 2,
                dtype=np.float64
            )
        )

        del G_chunk
        del C_chunk
        del GC_chunk

        print(
            f"Processed voxels: {end}/{nvoxels}",
            end="\r",
            flush=True
        )

    GC.flush()

    del GC

    print()

    return run_sos


# ============================================================
# 13. PROCESS RUN-LEVEL GC
# ============================================================

def process_run_GC(
    subject,
    run_results,
    C_subject,
    subject_dir
):

    print("\n")
    print("=" * 65)
    print("PASS 2: RUN-LEVEL GC CALCULATIONS")
    print("=" * 65)

    total_sos = 0.0

    run_sos_values = {}

    for result in run_results:

        run_number = result["run_number"]

        task_suffix = result["task_suffix"]

        print(
            f"\nProcessing GC for run {run_number}"
        )

        # ----------------------------------------------------
        # Locate the corresponding run-level G matrix.
        # ----------------------------------------------------

        G_file = find_run_G(
            G_DIR / subject,
            task_suffix,
            run_number,
            gz_file=Path(result["GZ_file"])
        )

        G_run = np.load(
            G_file,
            mmap_mode="r"
        )

        if G_run.ndim != 2:

            raise ValueError(
                f"Run G must be 2D: {G_file}"
            )

        prefix = (
            f"{subject}_{task_suffix}_run-{run_number}"
            if task_suffix
            else f"{subject}_run-{run_number}"
        )

        GC_file = (
            subject_dir /
            f"{prefix}_GC_R.npy"
        )

        # ----------------------------------------------------
        # MATRIX OPERATION 5:
        #
        # GC_R = G_R @ C_subject
        # ----------------------------------------------------

        run_sos = calculate_GC_chunked(
            subject,
            G_run,
            C_subject,
            GC_file
        )

        run_sos_values[prefix] = run_sos

        total_sos += run_sos

        result["G_file"] = str(G_file)

        result["GC_file"] = str(GC_file)

        result["run_sos"] = run_sos

        print(
            f"Run SSQ: {run_sos:.10e}"
        )

        print(
            f"Saved: {GC_file.name}"
        )

        del G_run

    return total_sos, run_sos_values


# ============================================================
# 14. SAVE SUBJECT SSQ
# ============================================================

def save_subject_SSQ(
    subject,
    total_sos,
    run_sos_values,
    output_dir
):

    """
    Save the total subject-level sum of squares.

    MATLAB equivalent:

        subject_GCsd = sum over runs and voxels

        SSQ.sd = total subject sum of squares
    """

    SSQ_file = (
        output_dir /
        f"{subject}_SSQ.npy"
    )

    np.save(
        SSQ_file,
        np.array(total_sos, dtype=np.float64)
    )

    print(
        f"\nSubject SSQ: {total_sos:.10e}"
    )

    print(
        f"Saved: {SSQ_file.name}"
    )

    run_SSQ_file = (
        output_dir /
        f"{subject}_run_SSQ.csv"
    )

    pd.DataFrame([
        {
            "run": run,
            "SSQ": value
        }
        for run, value in run_sos_values.items()
    ]).to_csv(
        run_SSQ_file,
        index=False
    )

    return SSQ_file


# ============================================================
# 15. SAVE SUBJECT METADATA
# ============================================================

def save_subject_metadata(
    subject,
    run_results,
    B_subject,
    C_subject,
    total_sos,
    output_dir
):

    metadata = {

        "subject": subject,

        "number_of_runs": len(run_results),

        "B_subject_shape": list(
            B_subject.shape
        ),

        "C_subject_shape": list(
            C_subject.shape
        ),

        "total_SSQ": float(total_sos),

        "runs": []

    }

    for result in run_results:

        metadata["runs"].append({

            "run_number": result["run_number"],

            "task": result["task_suffix"],

            "GZ_file": result["GZ_file"],

            "G_file": result.get("G_file"),

            "B_file": result["B_file"],

            "C_file": result["C_file"],

            "GC_file": result.get("GC_file"),

            "GZ_shape": list(
                result["GZ_shape"]
            ),

            "run_SSQ": float(
                result["run_sos"]
            )

        })

    metadata_file = (
        output_dir /
        f"{subject}_step3b_metadata.json"
    )

    with open(
        metadata_file,
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    print(
        f"\nSaved metadata: {metadata_file.name}"
    )


# ============================================================
# 16. PROCESS ONE SUBJECT
# ============================================================

def process_subject(subject):

    start_time = time.time()

    # --------------------------------------------------------
    # Initialize subject
    # --------------------------------------------------------

    data = initialize_subject(subject)

    subject_dir = data["output_dir"]

    gg = data["gg"]

    GZ_files = data["GZ_files"]

    # --------------------------------------------------------
    # PASS 1:
    #
    # Calculate run-level B and C.
    # Accumulate subject-level B and C.
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("PASS 1: RUN-LEVEL B AND C")
    print("=" * 65)

    run_results = []

    seen_runs = set()

    for GZ_file in GZ_files:

        task_suffix, run_number = extract_run_suffix(
            GZ_file.name
        )

        run_key = (task_suffix, run_number)

        if run_key in seen_runs:

            raise ValueError(
                f"Duplicate GZ run detected: {run_key}"
            )

        seen_runs.add(run_key)

        result = process_run_B_C(
            subject,
            run_number,
            task_suffix,
            GZ_file,
            gg,
            subject_dir
        )

        run_results.append(result)

    # --------------------------------------------------------
    # Accumulate B_S and C_S
    # --------------------------------------------------------

    B_subject, C_subject = accumulate_subject_B_C(
        subject,
        run_results,
        subject_dir
    )

    # --------------------------------------------------------
    # Release unnecessary subject G and gg memory
    # --------------------------------------------------------

    del data["G"]
    del data["gg"]

    del gg

    # --------------------------------------------------------
    # PASS 2:
    #
    # Calculate run-level GC.
    # Accumulate subject-level SSQ.
    # --------------------------------------------------------

    total_sos, run_sos_values = process_run_GC(
        subject,
        run_results,
        C_subject,
        subject_dir
    )

    # --------------------------------------------------------
    # Save SSQ
    # --------------------------------------------------------

    save_subject_SSQ(
        subject,
        total_sos,
        run_sos_values,
        subject_dir
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    save_subject_metadata(
        subject,
        run_results,
        B_subject,
        C_subject,
        total_sos,
        subject_dir
    )

    # --------------------------------------------------------
    # Release memory
    # --------------------------------------------------------

    del B_subject
    del C_subject
    del run_results

    elapsed = time.time() - start_time

    print("\n")
    print("=" * 65)
    print(f"SUBJECT COMPLETED: {subject}")
    print("=" * 65)

    print(
        f"Total SSQ: {total_sos:.10e}"
    )

    print(
        f"Elapsed time: {elapsed / 60:.2f} minutes"
    )

    return total_sos


# ============================================================
# 17. CREATE SUBJECT MANIFEST
# ============================================================

def create_manifest():

    subjects = discover_subjects()

    manifest = []

    for subject in subjects:

        manifest.append({

            "subject": subject,

            "G_dir": str(
                G_DIR / subject
            ),

            "GZ_dir": str(
                GZ_DIR / subject
            )

        })

    with open(
        MANIFEST_FILE,
        "w"
    ) as f:

        json.dump(
            manifest,
            f,
            indent=4
        )

    print(
        f"\nCreated manifest: {MANIFEST_FILE}"
    )

    print(
        f"Subjects: {len(manifest)}"
    )

    return manifest


# ============================================================
# 18. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Step 3b: Subject-level FIR regression"
    )

    parser.add_argument(
        "--create-manifest",
        action="store_true"
    )

    parser.add_argument(
        "--index",
        type=int,
        default=None
    )

    parser.add_argument(
        "--subject",
        type=str,
        default=None
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Create manifest
    # --------------------------------------------------------

    if args.create_manifest:

        create_manifest()

        return

    if not MANIFEST_FILE.exists():

        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_FILE}\n"
            "Run: python step3b.py --create-manifest"
        )

    with open(
        MANIFEST_FILE,
        "r"
    ) as f:

        manifest = json.load(f)

    # --------------------------------------------------------
    # Select subject
    # --------------------------------------------------------

    if args.subject is not None:

        selected = [
            item
            for item in manifest
            if item["subject"] == args.subject
        ]

        if not selected:

            raise ValueError(
                f"Subject not found in manifest: {args.subject}"
            )

        subject = selected[0]["subject"]

    elif args.index is not None:

        if not 0 <= args.index < len(manifest):

            raise IndexError(
                f"Invalid subject index: {args.index}. "
                f"Manifest contains {len(manifest)} subjects."
            )

        subject = manifest[args.index]["subject"]

    else:

        parser.error(
            "Specify either --index or --subject."
        )

    # --------------------------------------------------------
    # Process subject
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    try:

        process_subject(subject)

    except Exception:

        print(
            f"\nERROR processing {subject}",
            file=sys.stderr
        )

        traceback.print_exc()

        raise


# ============================================================
# 19. EXECUTION
# ============================================================

if __name__ == "__main__":

    main()
