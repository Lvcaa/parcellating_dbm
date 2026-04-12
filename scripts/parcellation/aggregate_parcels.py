import argparse
from concurrent.futures import ProcessPoolExecutor
import nibabel as nib
import os
from glob import glob
import numpy as np


def load_parcel_chunk(chunk_id, parcel_files, expected_shape, expected_affine):
    print(
        f"[worker {chunk_id}] Starting chunk with {len(parcel_files)} parcel files",
        flush=True,
    )
    chunk_sum = np.zeros(expected_shape, dtype=np.uint16)

    for file_index, parcel_file in enumerate(parcel_files, start=1):
        parcel_nifti = nib.load(parcel_file)
        if parcel_nifti.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for {parcel_file}: expected {expected_shape}, got {parcel_nifti.shape}"
            )
        if not np.allclose(parcel_nifti.affine, expected_affine):
            raise ValueError(f"Affine mismatch for {parcel_file}")

        parcel_img = parcel_nifti.get_fdata()
        chunk_sum += (parcel_img > 0).astype(np.uint8)
        print(
            f"[worker {chunk_id}] Loaded {file_index}/{len(parcel_files)}: {os.path.basename(parcel_file)}",
            flush=True,
        )

    print(f"[worker {chunk_id}] Finished chunk", flush=True)
    return chunk_sum


def chunk_files(parcel_files, n_chunks):
    chunk_size = max(1, int(np.ceil(len(parcel_files) / n_chunks)))
    return [
        parcel_files[start_index : start_index + chunk_size]
        for start_index in range(0, len(parcel_files), chunk_size)
    ]


def aggregate_parcels(parcels_dir, output_path, workers=1):
    # Get a list of all parcel files in the directory
    parcel_files = sorted(glob(os.path.join(parcels_dir, "roi_*.nii.gz")))

    if not parcel_files:
        raise ValueError(f"No parcel files found in {parcels_dir}")

    print(f"Found {len(parcel_files)} parcel files in {parcels_dir}", flush=True)
    print(f"Output will be written to {output_path}", flush=True)

    # Load the first parcel to get the shape and affine
    first_parcel = nib.load(parcel_files[0])
    template_img = first_parcel.get_fdata()
    affine = first_parcel.affine
    header = first_parcel.header
    print(f"Template shape: {template_img.shape}", flush=True)

    # Create an empty image to hold the aggregated parcels
    aggregated_image = np.zeros(template_img.shape, dtype=np.uint16)

    if workers <= 1 or len(parcel_files) == 1:
        print("Running aggregation in serial mode", flush=True)
        aggregated_image = load_parcel_chunk(1, parcel_files, template_img.shape, affine)
    else:
        n_workers = min(workers, len(parcel_files))
        file_chunks = chunk_files(parcel_files, n_workers)
        print(
            f"Running aggregation in parallel with {n_workers} workers across {len(file_chunks)} chunks",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(load_parcel_chunk, chunk_id, chunk, template_img.shape, affine)
                for chunk_id, chunk in enumerate(file_chunks, start=1)
            ]
            for future_index, future in enumerate(futures, start=1):
                aggregated_image += future.result()
                print(
                    f"Merged completed chunk {future_index}/{len(futures)} into aggregate volume",
                    flush=True,
                )

    # Create a NIfTI image from the aggregated data and save it
    print("Writing aggregated NIfTI image", flush=True)
    aggregated_nii = nib.Nifti1Image(aggregated_image, affine=affine, header=header)
    aggregated_nii.to_filename(output_path)
    print(f"Aggregated parcels saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate ROI parcel masks into a single NIfTI volume."
    )
    parser.add_argument(
        "parcels_dir",
        help="Directory containing roi_*.nii.gz parcel files.",
    )
    parser.add_argument(
        "output_path",
        help="Path for the aggregated NIfTI output.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes to use while loading parcels.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    aggregate_parcels(
        parcels_dir=args.parcels_dir,
        output_path=args.output_path,
        workers=args.workers,
    )
