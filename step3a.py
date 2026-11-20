import argparse
import json
import re
from pathlib import Path

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

# CHANGE THESE PATHS TO MATCH YOUR STEP 1 AND STEP 2b OUTPUTS

Z_DIR = Path("/scratch/st-toddwood-1/jshahki/fmri-cpca_python/vectorized_output")

G_DIR = Path("/scratch/st-toddwood-1/jshahki/fmri-cpca_python/fir_output")

OUTPUT_DIR = Path("/scratch/st-toddwood-1/jshahki/fmri-cpca_python/step3_output")

MANIFEST_FILE = Path("/scratch/st-toddwood-1/jshahki/fmri-cpca_python/step3_run_manifest.json")


# ============================================================
# 1. EXTRACT BIDS IDENTIFIERS
# ============================================================


def extract_entities(filename):
    """
    Extract BIDS-style entities from a matrix filename.

    The suffix (_Z or _G) is removed before parsing.

    Examples:
        sub-c01_task-semanticassociation_Z.npy
        sub-c01_task-semanticassociation_G.npy

        sub-c01_ses-01_task-rest_run-01_Z.npy
        sub-c01_ses-01_task-rest_run-01_G.npy

    Missing entities remain absent.
    """

    stem = Path(filename).stem

    # Remove matrix suffix.
    stem = re.sub(r"_(Z|G)$", "", stem)

    entities = {}

    for entity in stem.split("_"):

        if "-" not in entity:
            continue

        key, value = entity.split("-", 1)

        if key in ("sub", "ses", "task", "run", "acq", "rec", "dir", "echo"):
            entities[key] = value

    if "sub" not in entities:
        raise ValueError(
            f"No subject identifier found: {filename}"
        )

    # Include only entities actually present.
    # This preserves the distinction between missing labels
    # and explicitly provided labels.

    ordered_keys = [
        "sub",
        "ses",
        "task",
        "acq",
        "rec",
        "dir",
        "run",
        "echo"
    ]

    return tuple(
        (key, entities[key])
        for key in ordered_keys
        if key in entities
    )



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

    z_files = find_input_files(Z_DIR, "Z")
    g_files = [
        f for f in find_input_files(G_DIR, "G")
        if len(extract_entities(f)) > 1
    ]

    print(f"Found Z matrices: {len(z_files)}")
    print(f"Found G matrices: {len(g_files)}")

    z_index = {}
    g_index = {}

    # --------------------------------------------------------
    # INDEX Z MATRICES
    # --------------------------------------------------------

    for filename in z_files:

        identifier = extract_entities(filename)

        if identifier in z_index:
            raise ValueError(
                f"Duplicate Z identifier: {identifier}\n"
                f"{z_index[identifier]}\n{filename}"
            )

        z_index[identifier] = filename

    # --------------------------------------------------------
    # INDEX TASK/RUN-LEVEL G MATRICES
    #
    # Exclude subject-level G files such as:
    # sub-c01_G.npy
    #
    # These belong to Step 2b and must not be matched here.
    # --------------------------------------------------------

    for filename in g_files:

        identifier = extract_entities(filename)

        if identifier in g_index:
            raise ValueError(
                f"Duplicate G identifier: {identifier}\n"
                f"{g_index[identifier]}\n{filename}"
            )

        g_index[identifier] = filename

    # --------------------------------------------------------
    # VERIFY ONE-TO-ONE MATCHING
    # --------------------------------------------------------

    z_identifiers = set(z_index)
    g_identifiers = set(g_index)

    missing_g = z_identifiers - g_identifiers
    missing_z = g_identifiers - z_identifiers

    if missing_g:
        raise ValueError(
            "Missing task/run-level G matrices for:\n" +
            "\n".join(map(str, sorted(missing_g)))
        )

    if missing_z:
        raise ValueError(
            "G matrices without matching Z matrices:\n" +
            "\n".join(map(str, sorted(missing_z)))
        )

    # --------------------------------------------------------
    # CREATE MANIFEST
    # --------------------------------------------------------

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

    print(
        f"Total subject/task/run combinations: {len(manifest)}"
    )

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

def save_GZ(identifier, GZ):
    """
    Save each GZ matrix separately.

    Preserve the original Z filename and all existing BIDS labels.
    """

    # Ensure identifier is a string.
    if not isinstance(identifier, str):
        raise TypeError(
            f"Expected string identifier, got {type(identifier)}"
        )

    # Extract subject identifier.
    subject_match = re.search(
        r"(?:^|_)(sub-[^_]+)",
        identifier
    )

    if subject_match is None:
        raise ValueError(
            f"Could not identify subject in: {identifier}"
        )

    subject = subject_match.group(1)

    # Create subject directory.
    subject_dir = OUTPUT_DIR / subject

    subject_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Construct output filename.
    output_file = (
        subject_dir /
        f"{identifier}_GZ.npy"
    )

    # Save matrix.
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

    # Recover the original filename from the Z matrix.
    # Remove only the final _Z suffix.
    identifier = Path(run["z_file"]).stem

    if identifier.endswith("_Z"):
        identifier = identifier[:-2]

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
