import re
from pathlib import Path

import nibabel as nib


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data")

CONDITIONS_FILE = Path("conditions.txt")

RUN_CONDITIONS_FILE = Path("run_conditions.txt")

TEMPLATE_FILE = Path("timing_onsets_template.txt")

INFO_FILE = Path("fir_experiment_info.txt")


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
# 2. EXTRACT TASK AND RUN IDENTIFIERS FROM FILENAME
# ============================================================

def extract_run_id(filename):

    """
    Extract task and optional run identifiers.

    Examples:

    sub-01_task-workingmemory_run-1_bold.nii.gz
        -> task-workingmemory_run-1

    sub-01_task-emotionreg_bold.nii.gz
        -> task-emotionreg

    sub-01_task-emotionreg_run-2_bold.nii.gz
        -> task-emotionreg_run-2
    """

    name = Path(filename).name

    task_match = re.search(
        r"(?:^|_)task-([^_]+)",
        name
    )

    if not task_match:

        raise ValueError(
            f"No task identifier found in filename: {name}\n"
            "Expected a filename containing task-<name>."
        )

    task_name = task_match.group(1)

    run_match = re.search(
        r"(?:^|_)run-([^_]+)",
        name
    )

    if run_match:

        run_number = run_match.group(1)

        run_id = (
            f"task-{task_name}_run-{run_number}"
        )

    else:

        run_id = f"task-{task_name}"

    return run_id


# ============================================================
# 3. READ GLOBAL CONDITIONS
# ============================================================

def read_conditions(filename):

    conditions = {}

    if not filename.exists():

        raise FileNotFoundError(
            f"Condition file not found: {filename}"
        )

    with open(filename, "r") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) != 2:

                raise ValueError(
                    f"Invalid condition definition at "
                    f"line {line_number}: {line}"
                )

            name = parts[0]

            if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):

                raise ValueError(
                    f"Invalid condition name: {name}"
                )

            bins = int(parts[1])

            if bins < 1:

                raise ValueError(
                    f"FIR bins must be positive: {name}"
                )

            if name in conditions:

                raise ValueError(
                    f"Duplicate condition: {name}"
                )

            conditions[name] = bins

    if not conditions:

        raise ValueError("No conditions found.")

    return conditions


# ============================================================
# 4. INSPECT DATA DIRECTORY
# ============================================================

def inspect_data(data_dir):

    if not data_dir.exists():

        raise FileNotFoundError(
            f"Data directory not found: {data_dir}"
        )

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

            # Extract task and run identifiers

            run_id = extract_run_id(nii_file.name)

            # Prevent duplicate task/run identifiers

            if run_id in runs:

                raise ValueError(
                    f"Duplicate run identifier for "
                    f"{subject_dir.name}: {run_id}\n"
                    f"Files:\n"
                    f"{runs[run_id]['file']}\n"
                    f"{nii_file}\n"
                    "Each task/run identifier must correspond "
                    "to exactly one NIfTI file."
                )

            # Read NIfTI dimensions

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
                    f"Unexpected NIfTI dimensions: "
                    f"{nii_file}: {shape}"
                )

            runs[run_id] = {

                "file": nii_file,

                "nscans": nscans,

                "shape": shape,

                "task": run_id.split("_run-")[0],

                "run_id": run_id
            }

        subjects[subject_dir.name] = runs

    if not subjects:

        raise ValueError(
            "No subjects containing .nii.gz files found."
        )

    return subjects


# ============================================================
# 5. READ TASK/RUN-SPECIFIC CONDITIONS
# ============================================================

def read_run_conditions(
    filename,
    subjects,
    conditions
):

    if not filename.exists():

        raise FileNotFoundError(
            f"Run condition file not found: {filename}"
        )

    run_conditions = {}

    with open(filename, "r") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 3:

                raise ValueError(
                    f"Invalid run condition definition "
                    f"at line {line_number}: {line}"
                )

            subject = parts[0]

            run_id = parts[1]

            condition_list = parts[2:]

            key = (subject, run_id)

            if key in run_conditions:

                raise ValueError(
                    f"Duplicate run definition: "
                    f"{subject} {run_id}"
                )

            if subject not in subjects:

                raise ValueError(
                    f"Unknown subject: {subject}"
                )

            if run_id not in subjects[subject]:

                raise ValueError(
                    f"Unknown task/run: "
                    f"{subject} {run_id}"
                )

            if len(condition_list) != len(set(condition_list)):

                raise ValueError(
                    f"Duplicate conditions for "
                    f"{subject} {run_id}"
                )

            for condition in condition_list:

                if condition not in conditions:

                    raise ValueError(
                        f"Unknown condition: {condition}"
                    )

            run_conditions[key] = condition_list

    # Check that every detected run has a definition

    expected_runs = {

        (subject, run_id)

        for subject, runs in subjects.items()

        for run_id in runs
    }

    actual_runs = set(run_conditions.keys())

    missing = expected_runs - actual_runs

    extra = actual_runs - expected_runs

    if missing:

        raise ValueError(
            "Missing task/run condition definitions:\n" +
            "\n".join(
                f"{s} {r}" for s, r in sorted(missing)
            )
        )

    if extra:

        raise ValueError(
            "Unexpected task/run definitions:\n" +
            "\n".join(
                f"{s} {r}" for s, r in sorted(extra)
            )
        )

    return run_conditions


# ============================================================
# 6. PRINT EXPERIMENT INFORMATION
# ============================================================

def print_information(
    subjects,
    conditions,
    run_conditions
):

    total_runs = 0

    total_scans = 0

    print("\n")
    print("=" * 60)
    print("EXPERIMENT INFORMATION")
    print("=" * 60)

    for subject, runs in subjects.items():

        print(f"\nSubject: {subject}")

        for run_id, info in runs.items():

            active_conditions = run_conditions[
                (subject, run_id)
            ]

            print(
                f"\n  Run: {run_id}"
            )

            print(
                f"  File: {info['file'].name}"
            )

            print(
                f"  Scans: {info['nscans']}"
            )

            print(
                f"  Conditions: "
                f"{', '.join(active_conditions)}"
            )

            total_runs += 1

            total_scans += info["nscans"]

    total_bins = sum(conditions.values())

    print("\n")
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"Subjects: {len(subjects)}")

    print(f"Total task/runs: {total_runs}")

    print(f"Total scans: {total_scans}")

    print(f"Global conditions: {len(conditions)}")

    print(f"Total FIR columns: {total_bins}")


# ============================================================
# 7. CREATE TASK-SPECIFIC ONSET TEMPLATE
# ============================================================

def create_template(
    subjects,
    run_conditions,
    filename
):

    with open(filename, "w") as f:

        f.write(
            "# ==================================================\n"
        )

        f.write(
            "# FIR TIMING ONSET TEMPLATE\n"
        )

        f.write(
            "# ==================================================\n\n"
        )

        f.write(
            "# Onsets must be specified in scan units.\n"
        )

        f.write(
            "# Scan numbering starts at 1.\n"
        )

        f.write(
            "# Spaces and commas are both accepted.\n"
        )

        f.write(
            "# Example: [10 20 30 40]\n"
        )

        f.write(
            "# Example: [10, 20, 30, 40]\n\n"
        )

        f.write(
            "# Only conditions assigned to each task/run "
            "are included.\n\n"
        )

        for subject, runs in subjects.items():

            f.write("\n")
            f.write(f"# SUBJECT: {subject}\n")

            for run_id in runs:

                active_conditions = run_conditions[
                    (subject, run_id)
                ]

                f.write("\n")
                f.write(f"# {run_id}\n")

                for condition in active_conditions:

                    variable_name = (
                        f"{subject}_{run_id}_{condition}"
                    )

                    f.write(
                        f"{variable_name} = []\n"
                    )

    print(
        f"\nCreated onset template: {filename}"
    )


# ============================================================
# 8. SAVE EXPERIMENT METADATA
# ============================================================

def save_experiment_info(
    subjects,
    conditions,
    run_conditions,
    filename
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

    with open(filename, "w") as f:

        f.write("FIR EXPERIMENT INFORMATION\n")

        f.write("=" * 55 + "\n\n")

        f.write(
            f"Subjects: {len(subjects)}\n"
        )

        f.write(
            f"Task/runs: {total_runs}\n"
        )

        f.write(
            f"Total scans: {total_scans}\n"
        )

        f.write(
            f"Global conditions: {len(conditions)}\n"
        )

        f.write(
            f"Total FIR columns: {total_columns}\n\n"
        )

        f.write("GLOBAL CONDITIONS\n")

        f.write("-" * 55 + "\n")

        for condition, bins in conditions.items():

            f.write(
                f"{condition}: {bins} bins\n"
            )

        f.write("\nTASK/RUN-SPECIFIC CONDITIONS\n")

        f.write("-" * 55 + "\n")

        for subject, runs in subjects.items():

            for run_id, info in runs.items():

                active = run_conditions[
                    (subject, run_id)
                ]

                f.write(
                    f"{subject} {run_id}\n"
                )

                f.write(
                    f"  Conditions: "
                    f"{', '.join(active)}\n"
                )

                f.write(
                    f"  Scans: {info['nscans']}\n"
                )

                f.write(
                    f"  File: {info['file'].name}\n\n"
                )


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print("\nStarting FIR preparation...")

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

    # Display experiment information

    print_information(
        subjects,
        conditions,
        run_conditions
    )

    # Generate onset template

    create_template(
        subjects,
        run_conditions,
        TEMPLATE_FILE
    )

    # Save metadata

    save_experiment_info(
        subjects,
        conditions,
        run_conditions,
        INFO_FILE
    )

    print("\n")
    print("=" * 60)
    print("FIR PREPARATION COMPLETED")
    print("=" * 60)

    print(f"Template: {TEMPLATE_FILE}")

    print(f"Metadata: {INFO_FILE}")

    print(
        "\nComplete timing_onsets_template.txt "
        "and save it as timing_onsets.txt."
    )


if __name__ == "__main__":

    main()
