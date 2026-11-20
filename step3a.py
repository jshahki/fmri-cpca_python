import argparse
import json
import re
from pathlib import Path

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

# CHANGE THESE PATHS TO MATCH YOUR STEP 1 AND STEP 2b OUTPUTS

Z_DIR = Path("step1_output")

G_DIR = Path("fir_output")

OUTPUT_DIR = Path("step3_output")

MANIFEST_FILE = Path("step3_run_manifest.json")


# ============================================================
# 1. EXTRACT BIDS IDENTIFIERS
# ============================================================

def extract_entities(filename):
    """
    Extract the following identifiers when present:

        sub
        ses
        task
        run

    The order in the original filename does not matter.

    Example:

        sr_sub-c01_ses-A_task-semanticassociation_
        space-MNI152NLin2009cAsym_run-1_desc-preproc_bold_Z.npy

    Returns:

        sub-c01_ses-A_task-semanticassociation_run-1
    """

    name = Path(filename).name

    entities = {}

    for entity in ["sub", "ses", "task", "run"]:

        match = re.search(
            rf"(?:^|_){entity}-([^_]+)",
            name
        )

        if match:
            entities[entity] = match.group(1)

    if "sub" not in entities:
        raise ValueError(
            f"No subject identifier found: {filename}"
        )

    # Construct standardized identifier

    identifier = "_".join(
        f"{entity}-{entities[entity]}"
        for entity in ["sub", "ses", "task", "run"]
        if entity in entities
    )

    return identifier


# ============================================================
# 2. FIND INPUT MATRICES
# ============================================================

def find_input_files(directory, suffix):
    """
    Find all NumPy files ending in the requested suffix.

    Examples:

        *_Z.npy
        *_G.npy
    """

    if not directory.exists():

        raise FileNotFoundError(
            f"Input directory not found: {directory}"
        )

    files = sorted(
        directory.rglob(f"*_{suffix}.npy")
    )

    if not files:

        raise FileNotFoundError(
            f"No *_{suffix}.npy files found in {directory}"
        )

    return files


# ============================================================
# 3. CREATE RUN-LEVEL MANIFEST
# ============================================================

def create_manifest():

    print("\nScanning input directories...")

    z_files = find_input_files(
        Z_DIR,
        "Z"
    )

    g_files = find_input_files(
        G_DIR,
        "G"
    )

    print(f"Found Z matrices: {len(z_files)}")
    print(f"Found G matrices: {len(g_files)}")

    # Index files using standardized identifiers

    z_index = {}
    g_index = {}

    for filename in z_files:

        identifier = extract_entities(filename)

        if identifier in z_index:

            raise ValueError(
                f"Duplicate Z identifier: {identifier}\n"
                f"{z_index[identifier]}\n"
                f"{filename}"
            )

        z_index[identifier] = filename

    for filename in g_files:

        identifier = extract_entities(filename)

        if identifier in g_index:

            raise ValueError(
                f"Duplicate G identifier: {identifier}\n"
                f"{g_index[identifier]}\n"
                f"{filename}"
            )

        g_index[identifier] = filename

    # Verify matching input files

    z_identifiers = set(z_index)
    g_identifiers = set(g_index)

    missing_g = z_identifiers - g_identifiers
    missing_z = g_identifiers - z_identifiers

    if missing_g:

        raise ValueError(
            "Missing G matrices for:\n" +
            "\n".join(sorted(missing_g))
        )

    if missing_z:

        raise ValueError(
            "Missing Z matrices for:\n" +
            "\n".join(sorted(missing_z))
        )

    # Create manifest

    manifest = []

    for identifier in sorted(z_identifiers):

        manifest.append({

            "identifier": identifier,

            "z_file": str(
                z_index[identifier].resolve()
            ),

            "g_file": str(
                g_index[identifier].resolve()
            )

        })

    with open(MANIFEST_FILE, "w") as f:

        json.dump(
            manifest,
            f,
            indent=4
        )

    print("\nManifest successfully created.")

    print(f"Total subject/run combinations: {len(manifest)}")

    print(f"Manifest: {MANIFEST_FILE}")

    return manifest


# ============================================================
# 4. LOAD MANIFEST
# ============================================================

def load_manifest():

    if not MANIFEST_FILE.exists():

        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_FILE}\n"
            "Run step3a.py --create-manifest first."
        )

    with open(MANIFEST_FILE, "r") as f:

        manifest = json.load(f)

    return manifest


# ============================================================
# 5. CALCULATE RUN-LEVEL GZ
# ============================================================

def calculate_GZ(
    g_file,
    z_file
):
    """
    MATRIX OPERATION 1:

        GZ = G.T @ Z

    Dimensions:

        G:  (number of scans, FIR columns)

        Z:  (number of scans, voxels)

        GZ: (FIR columns, voxels)
    """

    print("\nLoading matrices...")

    print(f"G: {g_file.name}")
    print(f"Z: {z_file.name}")

    # Memory-map input matrices

    G = np.load(
        g_file,
        mmap_mode="r"
    )

    Z = np.load(
        z_file,
        mmap_mode="r"
    )

    # Validate dimensions

    if G.ndim != 2:

        raise ValueError(
            f"G must be 2D: {g_file} {G.shape}"
        )

    if Z.ndim != 2:

        raise ValueError(
            f"Z must be 2D: {z_file} {Z.shape}"
        )

    if G.shape[0] != Z.shape[0]:

        raise ValueError(
            "G and Z have different numbers of scans:\n"
            f"G: {G.shape}\n"
            f"Z: {Z.shape}"
        )

    print(f"\nG dimensions: {G.shape}")
    print(f"Z dimensions: {Z.shape}")

    # --------------------------------------------------------
    # MATRIX OPERATION 1
    # GZ = G.T @ Z
    # --------------------------------------------------------

    print("\nCalculating GZ = G.T @ Z ...")

    GZ = G.T @ Z

    print(f"GZ dimensions: {GZ.shape}")

    return GZ


# ============================================================
# 6. SAVE RUN-LEVEL GZ
# ============================================================

def save_GZ(
    identifier,
    GZ
):
    """
    Save each GZ matrix separately.

    Output structure:

        step3_output/
            sub-c01/
                sub-c01_task-rest_run-1_GZ.npy
    """

    # Extract subject identifier

    subject_match = re.search(
        r"(?:^|_)(sub-[^_]+)",
        identifier
    )

    subject = subject_match.group(1)

    # Create subject directory

    subject_dir = OUTPUT_DIR / subject

    subject_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Construct filename

    output_file = (
        subject_dir /
        f"{identifier}_GZ.npy"
    )

    # Save matrix

    np.save(
        output_file,
        GZ
    )

    print(f"\nSaved: {output_file}")

    print(f"Output dimensions: {GZ.shape}")


# ============================================================
# 7. PROCESS ONE ARRAY JOB
# ============================================================

def process_run(index):

    manifest = load_manifest()

    if index < 0 or index >= len(manifest):

        raise IndexError(
            f"Invalid array index: {index}\n"
            f"Valid indices: 0 to {len(manifest) - 1}"
        )

    run = manifest[index]

    identifier = run["identifier"]

    g_file = Path(run["g_file"])

    z_file = Path(run["z_file"])

    print("\n")
    print("=" * 65)
    print("STEP 3a: RUN-LEVEL GZ CALCULATION")
    print("=" * 65)

    print(f"\nArray index: {index}")

    print(f"Run identifier: {identifier}")

    # Calculate GZ

    GZ = calculate_GZ(
        g_file,
        z_file
    )

    # Save GZ

    save_GZ(
        identifier,
        GZ
    )

    print("\n")
    print("=" * 65)
    print("RUN-LEVEL CALCULATION COMPLETED")
    print("=" * 65)


# ============================================================
# 8. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Stage 1: Run-level GZ calculations"
    )

    parser.add_argument(
        "--create-manifest",
        action="store_true",
        help="Scan inputs and create the run manifest"
    )

    parser.add_argument(
        "--index",
        type=int,
        help="Process one manifest entry using a zero-based index"
    )

    args = parser.parse_args()

    if args.create_manifest:

        create_manifest()

    elif args.index is not None:

        process_run(args.index)

    else:

        parser.error(
            "Specify --create-manifest or --index"
        )


if __name__ == "__main__":

    main()
