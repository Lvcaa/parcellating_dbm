FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir nibabel scikit-learn

COPY . .

ENV PYTHONUNBUFFERED=1
ENV DEFAULT_SEGMENTATION=/app/data/reference/MNI152_T1_1mm_seg.nii.gz

ENTRYPOINT ["/bin/bash", "-lc", "\
set -eu\n\
export OPENBLAS_NUM_THREADS=\"${OPENBLAS_NUM_THREADS:-64}\"\n\
workdir=\"${PWD:-/app}\"\n\
output_root=\"${SUB_PARCELS_OUTPUT_ROOT:-${workdir}/outputs/rois}\"\n\
mkdir -p \"${output_root}\"\n\
if [ -n \"${SUB_PARCELS_SEGMENTATION:-}\" ]; then\n\
    segmentation=\"${SUB_PARCELS_SEGMENTATION}\"\n\
elif [ -f \"${workdir}/MNI152_T1_1mm_seg.nii.gz\" ]; then\n\
    segmentation=\"${workdir}/MNI152_T1_1mm_seg.nii.gz\"\n\
else\n\
    segmentation=\"${DEFAULT_SEGMENTATION}\"\n\
fi\n\
exec python /app/scripts/parcellation/sub_parcels_equal_size.py --segmentation \"${segmentation}\" --output-root \"${output_root}\" \"$@\"", "--"]
