import argparse
import json
import re
import sys
import time
import traceback

from pathlib import Path

import numpy as np

from scipy.sparse.linalg import eigsh, LinearOperator


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    "/scratch/st-toddwood-1/jshahki/fmri-cpca_python"
)

INPUT_DIR = BASE_DIR / "step3b_output"

OUTPUT_DIR = BASE_DIR / "step3c_output"

DTYPE = np.float64

# Number of B-matrix rows processed at a time.
# Reduce this if memory is limited.
CHUNK_SIZE = 256

# Number of leading eigenvalues to calculate.
# MATLAB equivalent: constant_define('EIG_COUNT')
EIG_COUNT = 15

# Numerical tolerance.
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
# 2. DISCOVER SUBJECTS
# ============================================================

def discover_subjects():

    if not INPUT_DIR.exists():

        raise FileNotFoundError(
            f"Step 3b directory not found: {INPUT_DIR}"
        )

    subjects = []

    for subject_dir in INPUT_DIR.iterdir():

        if not subject_dir.is_dir():
            continue

        B_file = (
            subject_dir /
            f"{subject_dir.name}_B.npy"
        )

        if B_file.exists():

            subjects.append(subject_dir.name)

    subjects = natural_sort(subjects)

    if not subjects:

        raise FileNotFoundError(
            f"No subject-level B matrices found in {INPUT_DIR}"
        )

    print("\nDiscovered subjects:")

    for subject in subjects:
        print(f"  {subject}")

    print(f"\nTotal subjects: {len(subjects)}")

    return subjects


# ============================================================
# 3. LOAD SUBJECT B MATRIX
# ============================================================

def load_subject_B(subject):

    B_file = (
        INPUT_DIR /
        subject /
        f"{subject}_B.npy"
    )

    if not B_file.exists():

        raise FileNotFoundError(
            f"Missing subject B matrix: {B_file}"
        )

    B = np.load(
        B_file,
        mmap_mode="r"
    )

    if B.ndim != 2:

        raise ValueError(
            f"B must be a 2D matrix: {B_file}"
        )

    if B.shape[0] == 0 or B.shape[1] == 0:

        raise ValueError(
            f"B matrix is empty: {B_file}"
        )

    print(
        f"{subject}: B dimensions = {B.shape}"
    )

    return B, B_file


# ============================================================
# 4. VALIDATE SUBJECT MATRICES
# ============================================================

def validate_subjects(subjects):

    print("\n")
    print("=" * 65)
    print("VALIDATING SUBJECT B MATRICES")
    print("=" * 65)

    metadata = []

    reference_columns = None

    for subject in subjects:

        B, B_file = load_subject_B(subject)

        # All subjects must have the same voxel dimensions.
        if reference_columns is None:

            reference_columns = B.shape[1]

        elif B.shape[1] != reference_columns:

            raise ValueError(
                f"Inconsistent voxel dimensions:\n"
                f"{subject}: {B.shape[1]}\n"
                f"Expected: {reference_columns}"
            )

        # ----------------------------------------------------
        # Validate finite values in manageable chunks.
        # ----------------------------------------------------

        for start in range(
            0,
            B.shape[0],
            CHUNK_SIZE
        ):

            block = np.asarray(
                B[start:start + CHUNK_SIZE, :]
            )

            if not np.all(np.isfinite(block)):

                raise ValueError(
                    f"Non-finite values found in {B_file}"
                )

            del block

        metadata.append({

            "subject": subject,

            "B_file": str(B_file),

            "rows": int(B.shape[0]),

            "columns": int(B.shape[1])

        })

        del B

    print("\nValidation successful.")

    print(
        f"Common voxel dimension: {reference_columns}"
    )

    return metadata


# ============================================================
# 5. CALCULATE SUBJECT POSITIONS IN GROUP MATRIX
# ============================================================

def calculate_subject_offsets(metadata):

    """
    Determine the row positions occupied by each subject
    in the concatenated group-level B matrix.
    """

    offset = 0

    for item in metadata:

        item["start"] = offset

        item["end"] = (
            offset + item["rows"]
        )

        offset = item["end"]

    total_rows = offset

    print("\nSubject positions in group matrix:")

    for item in metadata:

        print(
            f"{item['subject']}: "
            f"{item['start']}:{item['end']}"
        )

    print(
        f"\nTotal group-level rows: {total_rows}"
    )

    return metadata, total_rows


# ============================================================
# 6. CREATE GROUP-LEVEL CC MATRIX
# ============================================================

def create_CC_matrix(metadata, total_rows):

    """
    Construct:

        CC = B @ B.T

    where B is the vertically concatenated subject B matrices.

    MATLAB equivalent:

        CC = B * B'

    The CC matrix is saved as a disk-backed NumPy array.

    Only one subject-pair block and one row chunk
    are calculated at a time.
    """

    print("\n")
    print("=" * 65)
    print("CREATING GROUP-LEVEL CC MATRIX")
    print("=" * 65)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CC_file = OUTPUT_DIR / "GCC.npy"

    # --------------------------------------------------------
    # Estimate required disk space.
    # --------------------------------------------------------

    required_bytes = (
        total_rows *
        total_rows *
        np.dtype(DTYPE).itemsize
    )

    required_gb = (
        required_bytes /
        (1024 ** 3)
    )

    print(
        f"\nCC dimensions: ({total_rows}, {total_rows})"
    )

    print(
        f"Estimated CC disk space: {required_gb:.2f} GB"
    )

    # --------------------------------------------------------
    # Create disk-backed CC matrix.
    # --------------------------------------------------------

    CC = np.lib.format.open_memmap(
        CC_file,
        mode="w+",
        dtype=DTYPE,
        shape=(total_rows, total_rows)
    )

    number_subjects = len(metadata)

    # --------------------------------------------------------
    # Calculate each subject-pair block.
    # --------------------------------------------------------

    for i in range(number_subjects):

        primary = metadata[i]

        subject_i = primary["subject"]

        B_i = np.load(
            primary["B_file"],
            mmap_mode="r"
        )

        print(
            f"\nPrimary subject: {subject_i}"
        )

        for j in range(i, number_subjects):

            secondary = metadata[j]

            subject_j = secondary["subject"]

            B_j = np.load(
                secondary["B_file"],
                mmap_mode="r"
            )

            print(
                f"  Calculating block: "
                f"{subject_i} x {subject_j}"
            )

            # ------------------------------------------------
            # MATRIX OPERATION 1:
            #
            # CC_ij = B_i @ B_j.T
            #
            # MATLAB equivalent:
            #
            # CC(primary_rows, secondary_rows) =
            #     Primary * Secondary'
            # ------------------------------------------------

            for start in range(
                0,
                B_i.shape[0],
                CHUNK_SIZE
            ):

                end = min(
                    start + CHUNK_SIZE,
                    B_i.shape[0]
                )

                B_chunk = np.asarray(
                    B_i[start:end, :],
                    dtype=DTYPE
                )

                CC_block = B_chunk @ B_j.T

                if not np.all(np.isfinite(CC_block)):

                    raise ValueError(
                        f"Non-finite CC values in "
                        f"{subject_i} x {subject_j}"
                    )

                row_start = (
                    primary["start"] + start
                )

                row_end = (
                    primary["start"] + end
                )

                col_start = secondary["start"]

                col_end = secondary["end"]

                # Write upper block.
                CC[
                    row_start:row_end,
                    col_start:col_end
                ] = CC_block

                # Write corresponding lower block.
                # Avoid duplicate work on diagonal blocks.
                if i != j:

                    CC[
                        col_start:col_end,
                        row_start:row_end
                    ] = CC_block.T

                del B_chunk
                del CC_block

                print(
                    f"    Rows processed: {end}/{B_i.shape[0]}",
                    end="\r",
                    flush=True
                )

            print()

            del B_j

        del B_i

        CC.flush()

    # --------------------------------------------------------
    # Finish writing CC.
    # --------------------------------------------------------

    CC.flush()

    del CC

    print("\n")
    print("=" * 65)
    print("CC MATRIX COMPLETED")
    print("=" * 65)

    print(f"Saved: {CC_file}")

    print(
        f"Dimensions: ({total_rows}, {total_rows})"
    )

    return CC_file

  # ============================================================
# 7. MEMORY-EFFICIENT EIGENVALUE CALCULATION
# ============================================================

def calculate_eigenvalues(CC_file, total_rows):

    """
    Calculate the leading eigenvalues of CC.

    MATLAB equivalent:

        eig(CC)

    Uses scipy.sparse.linalg.eigsh with a disk-backed
    matrix-vector multiplication.

    This avoids loading the entire CC matrix into RAM.
    """

    print("\n")
    print("=" * 65)
    print("CALCULATING GROUP-LEVEL EIGENVALUES")
    print("=" * 65)

    print(
        f"\nCC dimensions: ({total_rows}, {total_rows})"
    )

    CC = np.load(
        CC_file,
        mmap_mode="r"
    )

    # --------------------------------------------------------
    # Special case: one-dimensional matrix.
    # --------------------------------------------------------

    if total_rows == 1:

        eigenvalues = np.array(
            [float(CC[0, 0])],
            dtype=DTYPE
        )

    else:

        k = min(
            EIG_COUNT,
            total_rows - 1
        )

        print(
            f"Calculating {k} leading eigenvalues."
        )

        # ----------------------------------------------------
        # Define disk-backed matrix-vector multiplication.
        # ----------------------------------------------------

        def matvec(x):

            result = np.zeros(
                total_rows,
                dtype=DTYPE
            )

            # ------------------------------------------------
            # MATRIX OPERATION 2:
            #
            # y = CC @ x
            #
            # The operation is performed in row chunks.
            # ------------------------------------------------

            for start in range(
                0,
                total_rows,
                CHUNK_SIZE
            ):

                end = min(
                    start + CHUNK_SIZE,
                    total_rows
                )

                result[start:end] = (
                    CC[start:end, :] @ x
                )

            return result

        operator = LinearOperator(
            shape=(total_rows, total_rows),
            matvec=matvec,
            dtype=DTYPE
        )

        # ----------------------------------------------------
        # MATRIX OPERATION 3:
        #
        # Calculate the largest eigenvalues of CC.
        #
        # MATLAB equivalent:
        #
        # sort(eig(CC), 'descend')
        #
        # Uses an iterative Lanczos eigensolver.
        # ----------------------------------------------------

        if total_rows <= EIG_COUNT:

            # Small matrix: calculate all eigenvalues.
            eigenvalues = np.linalg.eigvalsh(
                np.asarray(CC)
            )

            eigenvalues = eigenvalues[::-1]

        else:

            eigenvalues = eigsh(
                operator,
                k=k,
                which="LA",
                return_eigenvectors=False,
                tol=TOLERANCE,
                maxiter=max(1000, total_rows * 10)
            )

            eigenvalues = np.sort(
                eigenvalues
            )[::-1]

    del CC

    # --------------------------------------------------------
    # Save eigenvalues.
    # --------------------------------------------------------

    eigen_file = (
        OUTPUT_DIR /
        "GC_Eigenvalues.npy"
    )

    np.save(
        eigen_file,
        eigenvalues
    )

    print("\nEigenvalues calculated:")

    for i, value in enumerate(
        eigenvalues,
        start=1
    ):

        print(
            f"  Eigenvalue {i}: {value:.10e}"
        )

    print(
        f"\nSaved: {eigen_file}"
    )

    return eigenvalues


# ============================================================
# 8. ACCUMULATE GROUP-LEVEL SUM OF SQUARES
# ============================================================

def calculate_group_SSQ(subjects):

    """
    Accumulate the subject-level SSQ values.

    MATLAB equivalent:

        ts = ts + A.SSQ.sd
    """

    print("\n")
    print("=" * 65)
    print("ACCUMULATING GROUP-LEVEL SSQ")
    print("=" * 65)

    group_SSQ = 0.0

    subject_values = []

    for subject in subjects:

        SSQ_file = (
            INPUT_DIR /
            subject /
            f"{subject}_SSQ.npy"
        )

        if not SSQ_file.exists():

            raise FileNotFoundError(
                f"Missing subject SSQ: {SSQ_file}"
            )

        subject_SSQ = float(
            np.load(SSQ_file)
        )

        group_SSQ += subject_SSQ

        subject_values.append({

            "subject": subject,

            "SSQ": subject_SSQ

        })

        print(
            f"{subject}: {subject_SSQ:.10e}"
        )

    # --------------------------------------------------------
    # Save group SSQ.
    # --------------------------------------------------------

    group_SSQ_file = (
        OUTPUT_DIR /
        "G_group_SSQ.npy"
    )

    np.save(
        group_SSQ_file,
        np.array(group_SSQ, dtype=DTYPE)
    )

    print("\nGroup SSQ:")

    print(
        f"{group_SSQ:.10e}"
    )

    print(
        f"Saved: {group_SSQ_file}"
    )

    return group_SSQ, subject_values


# ============================================================
# 9. SAVE GROUP METADATA
# ============================================================

def save_metadata(
    metadata,
    total_rows,
    CC_file,
    eigenvalues,
    group_SSQ
):

    output = {

        "model": "G",

        "analysis": "whole_brain",

        "number_of_subjects": len(metadata),

        "total_group_rows": total_rows,

        "voxel_dimensions": metadata[0]["columns"],

        "CC_file": str(CC_file),

        "CC_shape": [
            total_rows,
            total_rows
        ],

        "eigenvalue_file": str(
            OUTPUT_DIR / "GC_Eigenvalues.npy"
        ),

        "number_of_eigenvalues": len(eigenvalues),

        "eigenvalues": eigenvalues.tolist(),

        "group_SSQ": float(group_SSQ),

        "subjects": metadata

    }

    metadata_file = (
        OUTPUT_DIR /
        "step3c_metadata.json"
    )

    with open(
        metadata_file,
        "w"
    ) as f:

        json.dump(
            output,
            f,
            indent=4
        )

    print(
        f"\nSaved metadata: {metadata_file}"
    )


# ============================================================
# 10. MAIN
# ============================================================

def main():

    start_time = time.time()

    parser = argparse.ArgumentParser(
        description=(
            "Step 3c: Whole-brain group-level "
            "CC matrix and eigenvalues"
        )
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE
    )

    parser.add_argument(
        "--eigen-count",
        type=int,
        default=EIG_COUNT
    )

    args = parser.parse_args()

    if args.chunk_size < 1:

        parser.error(
            "--chunk-size must be positive"
        )

    if args.eigen_count < 1:

        parser.error(
            "--eigen-count must be positive"
        )

    # --------------------------------------------------------
    # Apply command-line configuration.
    # --------------------------------------------------------

    global CHUNK_SIZE, EIG_COUNT

    CHUNK_SIZE = args.chunk_size

    EIG_COUNT = args.eigen_count

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n")
    print("=" * 65)
    print("STEP 3C: WHOLE-BRAIN GROUP-LEVEL PROCESSING")
    print("=" * 65)

    print(f"Input directory: {INPUT_DIR}")

    print(f"Output directory: {OUTPUT_DIR}")

    print(f"Chunk size: {CHUNK_SIZE}")

    print(f"Eigenvalue count: {EIG_COUNT}")

    # --------------------------------------------------------
    # STEP 1: Discover subjects.
    # --------------------------------------------------------

    subjects = discover_subjects()

    # --------------------------------------------------------
    # STEP 2: Validate B matrices.
    # --------------------------------------------------------

    metadata = validate_subjects(
        subjects
    )

    # --------------------------------------------------------
    # STEP 3: Calculate subject positions.
    # --------------------------------------------------------

    metadata, total_rows = calculate_subject_offsets(
        metadata
    )

    # --------------------------------------------------------
    # STEP 4: Create group-level CC.
    # --------------------------------------------------------

    CC_file = create_CC_matrix(
        metadata,
        total_rows
    )

    # --------------------------------------------------------
    # STEP 5: Calculate eigenvalues.
    # --------------------------------------------------------

    eigenvalues = calculate_eigenvalues(
        CC_file,
        total_rows
    )

    # --------------------------------------------------------
    # STEP 6: Accumulate group SSQ.
    # --------------------------------------------------------

    group_SSQ, subject_values = calculate_group_SSQ(
        subjects
    )

    # --------------------------------------------------------
    # STEP 7: Save metadata.
    # --------------------------------------------------------

    save_metadata(
        metadata,
        total_rows,
        CC_file,
        eigenvalues,
        group_SSQ
    )

    # --------------------------------------------------------
    # FINISH.
    # --------------------------------------------------------

    elapsed = time.time() - start_time

    print("\n")
    print("=" * 65)
    print("STEP 3C COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print(
        f"Subjects processed: {len(subjects)}"
    )

    print(
        f"CC dimensions: {total_rows} x {total_rows}"
    )

    print(
        f"Eigenvalues calculated: {len(eigenvalues)}"
    )

    print(
        f"Group SSQ: {group_SSQ:.10e}"
    )

    print(
        f"Elapsed time: {elapsed / 60:.2f} minutes"
    )

    print(
        f"\nOutput directory: {OUTPUT_DIR}"
    )


# ============================================================
# 11. EXECUTION
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception:

        print(
            "\nERROR DURING STEP 3C",
            file=sys.stderr
        )

        traceback.print_exc()

        sys.exit(1)
