import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data")
CONDITIONS_FILE = Path("conditions.txt")
RUN_CONDITIONS_FILE = Path("run_conditions.txt")
TIMING_FILE = Path("timing_onsets.txt")

OUTPUT_DIR = Path("fir_output")
METADATA_FILE = Path("fir_metadata.txt")


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
# 2. EXTRACT TASK AND RUN IDENTIFIERS
# ============================================================

def extract_run_id(filename):

    name = Path(filename).name

    task_match = re.search(
        r"(?:^|_)task-([^_]+)",
        name
    )

    if not task_match:
        raise ValueError(
            f"No task identifier found: {name}"
        )

    task_name = task_match.group(1)

    run_match = re.search(
        r"(?:^|_)run-([^_]+)",
        name
    )

    if run_match:
        run_number = run_match.group(1)
        return f"task-{task_name}_run-{run_number}"

    return f"task-{task_name}"


# ============================================================
# 3. READ GLOBAL CONDITIONS
# ============================================================

def read_conditions(filename):

    if not filename.exists():
        raise FileNotFoundError(filename)

    conditions = {}

    with open(filename, "r") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid condition at line {line_number}: {line}"
                )

            name = parts[0]
            bins = int(parts[1])

            if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
                raise ValueError(f"Invalid condition name: {name}")

            if bins < 1:
                raise ValueError(f"Invalid FIR bin count: {name}")

            if name in conditions:
                raise ValueError(f"Duplicate condition: {name}")

            conditions[name] = bins

    if not conditions:
        raise ValueError("No conditions found.")

    return conditions


# ============================================================
# 4. INSPECT NIFTI DATA
# ============================================================

def inspect_data(data_dir):

    if not data_dir.exists():
        raise FileNotFoundError(data_dir)

    subjects = {}

    subject_dirs = natural_sort([
        p for p in data_dir.iterdir()
        if p.is_dir()
    ])

    for subject_dir in subject_dirs:

        nii_files = natural_sort(
            list(subject_dir.glob("*.nii.gz"))
        )

        if not nii_files:
            continue

        runs = {}

        for nii_file in nii_files:

            print(
                f"Inspecting {subject_dir.name}: "
                f"{nii_file.name}"
            )

            run_id = extract_run_id(nii_file.name)

            if run_id in runs:
                raise ValueError(
                    f"Duplicate task/run identifier: "
                    f"{subject_dir.name} {run_id}"
                )

            img = nib.load(str(nii_file))
            shape = img.shape

            if len(shape) == 4:
                nscans = shape[3]

            elif len(shape) == 3:
                nscans = 1

                print(
                    f"WARNING: {nii_file.name} is 3D "
                    "and will be treated as one scan."
                )

            else:
                raise ValueError(
                    f"Unexpected NIfTI dimensions: {shape}"
                )

            task_name = run_id.split("_run-")[0]

            runs[run_id] = {
                "file": nii_file,
                "nscans": nscans,
                "shape": shape,
                "task": task_name,
                "run_id": run_id
            }

        subjects[subject_dir.name] = runs

    if not subjects:
        raise ValueError("No NIfTI files found.")

    return subjects


# ============================================================
# 5. READ TASK/RUN-SPECIFIC CONDITIONS
# ============================================================

def read_run_conditions(filename, subjects, conditions):

    if not filename.exists():
        raise FileNotFoundError(filename)

    run_conditions = {}

    with open(filename, "r") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 3:
                raise ValueError(
                    f"Invalid definition at line {line_number}: {line}"
                )

            subject = parts[0]
            run_id = parts[1]
            active_conditions = parts[2:]

            key = (subject, run_id)

            if key in run_conditions:
                raise ValueError(f"Duplicate definition: {key}")

            if subject not in subjects:
                raise ValueError(f"Unknown subject: {subject}")

            if run_id not in subjects[subject]:
                raise ValueError(f"Unknown task/run: {key}")

            if len(active_conditions) != len(set(active_conditions)):
                raise ValueError(f"Duplicate conditions: {key}")

            for condition in active_conditions:

                if condition not in conditions:
                    raise ValueError(
                        f"Unknown condition: {condition}"
                    )

            run_conditions[key] = active_conditions

    expected = {
        (subject, run_id)
        for subject, runs in subjects.items()
        for run_id in runs
    }

    actual = set(run_conditions.keys())

    missing = expected - actual
    extra = actual - expected

    if missing:
        raise ValueError(
            "Missing task/run definitions:\n" +
            "\n".join(f"{s} {r}" for s, r in sorted(missing))
        )

    if extra:
        raise ValueError(
            "Unexpected task/run definitions:\n" +
            "\n".join(f"{s} {r}" for s, r in sorted(extra))
        )

    return run_conditions



# ============================================================
# 6. READ ONSET FILE
# ============================================================

def read_onsets(filename):

    if not filename.exists():
        raise FileNotFoundError(
            f"Timing file not found: {filename}"
        )

    onsets = {}

    with open(filename, "r") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            # Allow an optional semicolon after ]
            match = re.match(
                r"^([^\s=]+)\s*=\s*\[(.*?)\]\s*;?\s*$",
                line
            )

            if not match:
                raise ValueError(
                    f"Invalid onset definition at line "
                    f"{line_number}: {line}"
                )

            name = match.group(1)
            values = match.group(2).strip()

            if name in onsets:
                raise ValueError(
                    f"Duplicate onset definition: {name}"
                )

            if not values:
                onsets[name] = []
                continue

            tokens = re.split(r"[\s,]+", values)

            try:
                event_onsets = [
                    float(token) for token in tokens
                ]

            except ValueError:
                raise ValueError(
                    f"Invalid onset values at line "
                    f"{line_number}: {values}"
                )

            if any(
                not np.isfinite(value) or value < 0
                for value in event_onsets
            ):
                raise ValueError(
                    f"Invalid onset values for {name}"
                )

            onsets[name] = event_onsets

    print(f"\nLoaded {len(onsets)} onset definitions.")

    return onsets



# ============================================================
# 7. VALIDATE ONSET DEFINITIONS
# ============================================================

def validate_onsets(subjects, run_conditions, onsets):

    expected_keys = {
        f"{subject}_{run_id}_{condition}"
        for subject, runs in subjects.items()
        for run_id in runs
        for condition in run_conditions[(subject, run_id)]
    }

    actual_keys = set(onsets.keys())

    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys

    if missing:
        raise ValueError(
            "Missing onset definitions:\n" +
            "\n".join(sorted(missing))
        )

    if extra:
        raise ValueError(
            "Unexpected onset definitions:\n" +
            "\n".join(sorted(extra))
        )

    for subject, runs in subjects.items():

        for run_id, info in runs.items():

            nscans = info["nscans"]

            for condition in run_conditions[(subject, run_id)]:

                key = f"{subject}_{run_id}_{condition}"

                for onset in onsets[key]:

                    if onset >= nscans:
                        raise ValueError(
                            f"Onset {onset} exceeds {nscans} scans "
                            f"for {key}"
                        )

    print("\nOnset validation successful.")


# ============================================================
# 8. CREATE GLOBAL FIR COLUMN NAMES
# ============================================================

def create_column_names(conditions):

    column_names = []
    condition_columns = {}

    column = 0

    for condition, bins in conditions.items():

        first_col = column

        for bin_index in range(bins):

            column_names.append(
                f"{condition}_bin{bin_index + 1}"
            )

            column += 1

        condition_columns[condition] = (
            first_col,
            column
        )

    return column_names, condition_columns


# ============================================================
# 9. CREATE FIR MATRICES
# ============================================================

def create_fir_matrices(
    subjects,
    conditions,
    run_conditions,
    onsets,
    column_names,
    condition_columns
):

    subject_matrices = {}
    subject_scan_info = {}

    total_columns = len(column_names)

    for subject, runs in subjects.items():

        print(f"\nGenerating FIR matrix for {subject}")

        run_matrices = []
        scan_info_rows = []

        for run_id, info in runs.items():

            nscans = info["nscans"]
            task_name = info["task"]

            active_conditions = run_conditions[
                (subject, run_id)
            ]

            print(f"\nProcessing: {run_id}")
            print(f"Scans: {nscans}")
            print(f"Conditions: {active_conditions}")

            # Initialize run matrix

            G_run = np.zeros(
                (nscans, total_columns),
                dtype=np.uint8
            )

            # Populate active conditions

            for condition in active_conditions:

                bins = conditions[condition]

                key = f"{subject}_{run_id}_{condition}"

                event_onsets = onsets[key]

                first_col, last_col = (
                    condition_columns[condition]
                )

                for onset in event_onsets:

                    # Convert 1-based scan to 0-based index

                    onset_index = int(round(onset)) - 1

                    for bin_index in range(bins):

                        scan_index = onset_index + bin_index

                        # Do not cross run boundaries

                        if scan_index >= nscans:
                            break

                        G_run[
                            scan_index,
                            first_col + bin_index
                        ] = 1

            run_matrices.append(G_run)

            # Record scan information

            for scan in range(nscans):

                scan_info_rows.append({

                    "row": len(scan_info_rows),

                    "subject": subject,

                    "task": task_name,

                    "run_id": run_id,

                    "scan": scan + 1,

                    "nscans_in_run": nscans,

                    "nifti_file": info["file"].name

                })

        # Concatenate all runs vertically

        G_subject = np.vstack(run_matrices)

        subject_matrices[subject] = G_subject

        subject_scan_info[subject] = pd.DataFrame(
            scan_info_rows
        )

        print(
            f"Final matrix for {subject}: "
            f"{G_subject.shape}"
        )

    return subject_matrices, subject_scan_info


# ============================================================
# 10. SAVE INDIVIDUAL SUBJECT MATRICES
# ============================================================

def save_subject_matrices(
    subject_matrices,
    subject_scan_info,
    column_names
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for subject, G in subject_matrices.items():

        subject_dir = OUTPUT_DIR / subject

        subject_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Save NumPy matrix

        np.save(
            subject_dir / "fir_matrix.npy",
            G
        )

        # Save CSV matrix

        df = pd.DataFrame(
            G,
            columns=column_names
        )

        df.to_csv(
            subject_dir / "fir_matrix.csv",
            index=False
        )

        # Save scan information

        subject_scan_info[subject].to_csv(
            subject_dir / "scan_info.csv",
            index=False
        )

        print(f"Saved {subject}: {G.shape}")


# ============================================================
# 11. SAVE METADATA
# ============================================================

def save_metadata(
    subjects,
    conditions,
    run_conditions,
    subject_matrices
):

    total_runs = sum(
        len(runs)
        for runs in subjects.values()
    )

    total_scans = sum(
        info["nscans"]
        for runs in subjects.values()
        for info in runs.values()
    )

    total_columns = sum(conditions.values())

    with open(METADATA_FILE, "w") as f:

        f.write("FIR MODEL METADATA\n")
        f.write("=" * 55 + "\n\n")

        f.write(f"Number of subjects: {len(subjects)}\n")
        f.write(f"Total task/runs: {total_runs}\n")
        f.write(f"Total scans: {total_scans}\n")
        f.write(f"Global conditions: {len(conditions)}\n")
        f.write(f"Total FIR columns: {total_columns}\n\n")

        f.write("GLOBAL FIR COLUMN DEFINITIONS\n")
        f.write("-" * 55 + "\n")

        column = 0

        for condition, bins in conditions.items():

            f.write(
                f"{condition}: columns "
                f"{column + 1}-{column + bins}\n"
            )

            column += bins

        f.write("\nTASK/RUN-SPECIFIC CONDITIONS\n")
        f.write("-" * 55 + "\n")

        for subject, runs in subjects.items():

            for run_id, info in runs.items():

                active = run_conditions[
                    (subject, run_id)
                ]

                f.write(f"\n{subject} {run_id}\n")
                f.write(f"  Task: {info['task']}\n")
                f.write(f"  Scans: {info['nscans']}\n")
                f.write(f"  Conditions: {', '.join(active)}\n")
                f.write(f"  File: {info['file'].name}\n")

        f.write("\nSUBJECT FIR MATRIX DIMENSIONS\n")
        f.write("-" * 55 + "\n")

        for subject, G in subject_matrices.items():

            f.write(
                f"{subject}: "
                f"{G.shape[0]} x {G.shape[1]}\n"
            )


# ============================================================
# 12. MAIN
# ============================================================

def main():

    print("\nStarting FIR generation...")

    # Read global conditions

    conditions = read_conditions(
        CONDITIONS_FILE
    )

    # Inspect subjects and task/runs

    subjects = inspect_data(
        DATA_DIR
    )

    # Read task/run-specific conditions

    run_conditions = read_run_conditions(
        RUN_CONDITIONS_FILE,
        subjects,
        conditions
    )

    # Read onset definitions

    onsets = read_onsets(
        TIMING_FILE
    )

    # Validate onset definitions

    validate_onsets(
        subjects,
        run_conditions,
        onsets
    )

    # Create global FIR columns

    column_names, condition_columns = (
        create_column_names(conditions)
    )

    # Generate subject matrices

    (
        subject_matrices,
        subject_scan_info
    ) = create_fir_matrices(
        subjects,
        conditions,
        run_conditions,
        onsets,
        column_names,
        condition_columns
    )

    # Save individual subject matrices

    save_subject_matrices(
        subject_matrices,
        subject_scan_info,
        column_names
    )

    # Save metadata

    save_metadata(
        subjects,
        conditions,
        run_conditions,
        subject_matrices
    )

    print("\n")
    print("=" * 60)
    print("FIR GENERATION COMPLETED")
    print("=" * 60)

    for subject, G in subject_matrices.items():

        print(f"{subject}: {G.shape}")

    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":

    main()
