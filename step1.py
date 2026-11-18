#!/usr/bin/env python3

import re
from pathlib import Path
import argparse

import numpy as np
import nibabel as nib


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("vectorized_output")


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
# 2. LOAD OPTIONAL BRAIN MASK
# ============================================================

def load_mask(mask_file):

    if mask_file is None:
        print("\nNo brain mask provided.")
        return None

    print(f"\nLoading brain mask: {mask_file}")

    mask_img = nib.load(str(mask_file))

    if len(mask_img.shape) != 3:
        raise ValueError(
            f"Expected a 3D brain mask, got {mask_img.shape}"
        )

    mask = mask_img.get_fdata() > 0

    print(f"Mask dimensions: {mask.shape}")
    print(f"Masked voxels: {mask.sum()}")

    return {
        "mask": mask,
        "affine": mask_img.affine
    }


# ============================================================
# 3. VECTORIZE AND NORMALIZE ONE FMRI RUN
# ============================================================

def vectorize_fmri(
    input_file,
    output_dir,
    mask_info=None
):

    input_file = Path(input_file)

    print("\n" + "=" * 60)
    print(f"Processing: {input_file.name}")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD NIFTI IMAGE
    # --------------------------------------------------------

    img = nib.load(str(input_file))

    if len(img.shape) != 4:
        raise ValueError(
            f"Expected a 4D NIfTI file: "
            f"{input_file.name}, got {img.shape}"
        )

    x, y, z, t = img.shape

    print(f"Original dimensions: {img.shape}")

    # --------------------------------------------------------
    # LOAD FMRI DATA
    # --------------------------------------------------------

    data = img.get_fdata(dtype=np.float32)

    # Reshape into timepoints x voxels
    data_2d = data.reshape(-1, t).T

    # --------------------------------------------------------
    # APPLY OPTIONAL BRAIN MASK
    # --------------------------------------------------------

    if mask_info is not None:

        mask = mask_info["mask"]

        if mask.shape != (x, y, z):
            raise ValueError(
                f"Mask dimensions {mask.shape} do not match "
                f"fMRI dimensions {(x, y, z)}"
            )

        # Check spatial alignment
        if not np.allclose(
            img.affine,
            mask_info["affine"],
            atol=1e-4
        ):
            raise ValueError(
                f"Mask affine does not match fMRI affine: "
                f"{input_file.name}"
            )

        mask_vector = mask.reshape(-1)

        data_2d = data_2d[:, mask_vector]

        mask_indices = np.flatnonzero(mask_vector)

        print(
            f"Number of masked voxels: "
            f"{mask_vector.sum()}"
        )

    else:

        print("Keeping all voxels.")

        mask_indices = np.arange(x * y * z)

    print(f"Original vectorized dimensions: {data_2d.shape}")

    # --------------------------------------------------------
    # NORMALIZE EACH VOXEL INDEPENDENTLY
    # --------------------------------------------------------

    print("\nPerforming run-specific voxel-wise normalization...")

    # Calculate mean across timepoints for each voxel
    voxel_mean = np.mean(
        data_2d,
        axis=0,
        keepdims=True
    )

    # Calculate standard deviation across timepoints
    voxel_std = np.std(
        data_2d,
        axis=0,
        keepdims=True
    )

    # Avoid division by zero for constant voxels
    zero_std_voxels = voxel_std == 0

    voxel_std[zero_std_voxels] = 1.0

    # Z-score normalization
    normalized_data = (
        data_2d - voxel_mean
    ) / voxel_std

    normalized_data = normalized_data.astype(np.float32)

    print(f"Normalized dimensions: {normalized_data.shape}")

    print(
        f"Normalized mean: {normalized_data.mean():.6f}"
    )

    print(
        f"Normalized standard deviation: "
        f"{normalized_data.std():.6f}"
    )

    print(f"Constant voxels: {zero_std_voxels.sum()}")

    # --------------------------------------------------------
    # CREATE OUTPUT FILENAMES
    # --------------------------------------------------------

    filename = input_file.name

    if filename.endswith(".nii.gz"):
        filename = filename[:-7]
    else:
        filename = input_file.stem

    output_file = (
        output_dir /
        f"{filename}_vectorized_normalized.npy"
    )

    metadata_file = (
        output_dir /
        f"{filename}_metadata.npz"
    )

    normalization_file = (
        output_dir /
        f"{filename}_normalization_parameters.npz"
    )

    # --------------------------------------------------------
    # SAVE NORMALIZED MATRIX
    # --------------------------------------------------------

    np.save(
        output_file,
        normalized_data
    )

    print(f"\nSaved normalized matrix: {output_file}")

    # --------------------------------------------------------
    # SAVE RECONSTRUCTION METADATA
    # --------------------------------------------------------

    header_bytes = img.header.binaryblock

    np.savez_compressed(
        metadata_file,

        # Original image dimensions
        original_shape=np.array(img.shape),

        # Spatial transformation
        affine=img.affine,

        # Original voxel dimensions
        voxel_size=np.array(
            img.header.get_zooms()[:3]
        ),

        # Temporal information
        tr=float(img.header.get_zooms()[3]),
        n_timepoints=t,
        n_voxels=normalized_data.shape[1],

        # Mask information
        mask_indices=mask_indices,
        mask_applied=mask_info is not None,

        # Original NIfTI header
        header_bytes=header_bytes,

        # Original filename
        original_filename=input_file.name
    )

    print(f"Saved reconstruction metadata: {metadata_file}")

    # --------------------------------------------------------
    # SAVE NORMALIZATION PARAMETERS
    # --------------------------------------------------------

    np.savez_compressed(
        normalization_file,

        voxel_mean=voxel_mean.astype(np.float32),
        voxel_std=voxel_std.astype(np.float32),
        zero_std_voxels=zero_std_voxels,
        n_timepoints=t,
        n_voxels=normalized_data.shape[1]
    )

    print(
        f"Saved normalization parameters: "
        f"{normalization_file}"
    )

    print("\nRun completed successfully.")

    return normalized_data.shape


# ============================================================
# 4. PROCESS ALL SUBJECTS AND RUNS
# ============================================================

def process_dataset(
    data_dir,
    output_dir,
    mask_file=None,
    single_file=None
):

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    if not data_dir.exists():

        raise FileNotFoundError(
            f"Data directory not found: {data_dir}"
        )

    # Load optional mask once

    mask_info = load_mask(mask_file)

    total_files = 0

    # ========================================================
    # MODE 1: PROCESS ONE SPECIFIC RUN (SLURM ARRAY)
    # ========================================================

    if single_file is not None:

        single_file = Path(single_file)

        if not single_file.is_file():

            raise FileNotFoundError(
                f"NIfTI file not found: {single_file}"
            )

        # Identify subject from parent directory

        subject = single_file.parent.name

        print("\n")
        print("#" * 60)
        print(f"SUBJECT: {subject}")
        print("#" * 60)

        print(f"Processing single run: {single_file.name}")

        # Create subject output directory

        subject_output_dir = output_dir / subject

        subject_output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Process only the assigned run

        vectorize_fmri(
            single_file,
            subject_output_dir,
            mask_info
        )

        total_files = 1

        n_subjects = 1

    # ========================================================
    # MODE 2: PROCESS ALL SUBJECTS AND RUNS
    # ========================================================

    else:

        # Find subjects

        subject_dirs = natural_sort([

            p for p in data_dir.iterdir()

            if p.is_dir()

        ])

        if not subject_dirs:

            raise ValueError(
                "No subject directories found."
            )

        n_subjects = len(subject_dirs)

        # Process each subject

        for subject_dir in subject_dirs:

            subject = subject_dir.name

            print("\n")
            print("#" * 60)
            print(f"SUBJECT: {subject}")
            print("#" * 60)

            # Find NIfTI files

            nii_files = natural_sort(

                list(subject_dir.glob("*.nii.gz"))

            )

            if not nii_files:

                print(
                    f"No NIfTI files found for {subject}. "
                    "Skipping."
                )

                continue

            # Create subject output directory

            subject_output_dir = output_dir / subject

            subject_output_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            # Process each run independently

            for nii_file in nii_files:

                vectorize_fmri(
                    nii_file,
                    subject_output_dir,
                    mask_info
                )

                total_files += 1

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")

    print("=" * 60)

    print("VECTORIZATION AND NORMALIZATION COMPLETED")

    print("=" * 60)

    print(f"Subjects processed: {n_subjects}")

    print(f"NIfTI files processed: {total_files}")

    print(f"Output directory: {output_dir}")


# ============================================================
# 5. RECONSTRUCT NORMALIZED MATRIX INTO NIFTI
# ============================================================

def reconstruct_nifti(
    matrix_file,
    metadata_file,
    output_file
):

    matrix_file = Path(matrix_file)
    metadata_file = Path(metadata_file)
    output_file = Path(output_file)

    print("\nReconstructing NIfTI image...")

    # --------------------------------------------------------
    # LOAD MATRIX AND METADATA
    # --------------------------------------------------------

    matrix = np.load(matrix_file)
    metadata = np.load(metadata_file)

    original_shape = tuple(
        metadata["original_shape"]
    )

    affine = metadata["affine"]

    mask_indices = metadata["mask_indices"]

    mask_applied = bool(
        metadata["mask_applied"]
    )

    # --------------------------------------------------------
    # CHECK DIMENSIONS
    # --------------------------------------------------------

    n_timepoints = original_shape[3]

    n_spatial_voxels = int(
        np.prod(original_shape[:3])
    )

    if matrix.shape[0] != n_timepoints:
        raise ValueError(
            "Matrix time dimension does not match metadata."
        )

    if matrix.shape[1] != len(mask_indices):
        raise ValueError(
            "Matrix voxel dimension does not match mask indices."
        )

    # --------------------------------------------------------
    # RECONSTRUCT SPATIAL DATA
    # --------------------------------------------------------

    reconstructed = np.zeros(
        (n_spatial_voxels, n_timepoints),
        dtype=np.float32
    )

    reconstructed[mask_indices, :] = matrix.T

    # Reshape back to X x Y x Z x T
    reconstructed = reconstructed.reshape(
        original_shape,
        order="C"
    )

    # --------------------------------------------------------
    # RESTORE ORIGINAL NIFTI HEADER
    # --------------------------------------------------------

    header = nib.Nifti1Header()

    header.binaryblock = metadata["header_bytes"].tobytes()

    # Ensure the output datatype supports normalized values
    header.set_data_dtype(np.float32)

    # --------------------------------------------------------
    # CREATE AND SAVE NIFTI IMAGE
    # --------------------------------------------------------

    output_img = nib.Nifti1Image(
        reconstructed,
        affine,
        header=header
    )

    # Ensure affine is correctly represented
    output_img.set_sform(affine)
    output_img.set_qform(affine)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    nib.save(
        output_img,
        str(output_file)
    )

    print(f"Reconstructed dimensions: {reconstructed.shape}")
    print(f"Saved NIfTI: {output_file}")
    print("Reconstruction completed successfully.")


# ============================================================
# 6. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Vectorize and independently normalize "
            "all 4D fMRI NIfTI runs. "
            "Also supports NIfTI reconstruction."
        )
    )

    parser.add_argument(
        "--data",
        default=str(DATA_DIR),
        help="Input data directory."
    )

    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR),
        help="Output directory."
    )

    parser.add_argument(
        "--mask",
        default=None,
        help="Optional 3D brain mask."
    )

    parser.add_argument(
        "--single-file",
        default=None,
        help="Process only this specific NIfTI file."
    )

    # Optional reconstruction arguments

    parser.add_argument(
        "--reconstruct",
        action="store_true",
        help="Reconstruct a normalized matrix into NIfTI."
    )

    parser.add_argument(
        "--matrix",
        default=None,
        help="Input normalized .npy matrix."
    )

    parser.add_argument(
        "--metadata",
        default=None,
        help="Corresponding reconstruction metadata .npz."
    )

    parser.add_argument(
        "--reconstructed-output",
        default=None,
        help="Output reconstructed .nii.gz filename."
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # RECONSTRUCTION MODE
    # --------------------------------------------------------

    if args.reconstruct:

        if not all([
            args.matrix,
            args.metadata,
            args.reconstructed_output
        ]):

            parser.error(
                "--reconstruct requires "
                "--matrix, --metadata, "
                "and --reconstructed-output."
            )

        reconstruct_nifti(
            args.matrix,
            args.metadata,
            args.reconstructed_output
        )

    # --------------------------------------------------------
    # VECTORIZATION MODE
    # --------------------------------------------------------

    else:

        process_dataset(
            data_dir=Path(args.data),
            output_dir=Path(args.output),

            mask_file=(
                Path(args.mask)
                if args.mask
                else None
            ),

            single_file=(
                Path(args.single_file)
                if args.single_file
                else None
            )
        )


if __name__ == "__main__":
    main()
