FROM python:3.11-slim

ARG UID=1000
ARG GID=1000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && groupadd --gid "${GID}" appuser \
    && useradd --uid "${UID}" --gid "${GID}" --create-home --shell /bin/bash appuser \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir nibabel scikit-learn

COPY . .

RUN chown -R appuser:appuser /app

ENV PYTHONUNBUFFERED=1
ENV DEFAULT_SEGMENTATION=/app/data/reference/MNI152_T1_1mm_seg.nii.gz
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV HOME=/home/appuser

USER appuser

ENTRYPOINT ["/bin/bash", "-lc", "\
set -eu\n\
workdir=\"${PWD:-/app}\"\n\
segmentation=\"${RUN_ALLOWED_SEGMENTATION:-${DEFAULT_SEGMENTATION}}\"\n\
output_root=\"${RUN_ALLOWED_OUTPUT_ROOT:-${workdir}/outputs/rois}\"\n\
aggregated_output_root=\"${RUN_ALLOWED_AGGREGATED_OUTPUT_ROOT:-${workdir}/outputs/aggregated_parcellations}\"\n\
mkdir -p \"${output_root}\"\n\
mkdir -p \"${aggregated_output_root}\"\n\
exec python /app/scripts/parcellation/run_allowed_parcellations.py --segmentation \"${segmentation}\" --output-root \"${output_root}\" --aggregated-output-root \"${aggregated_output_root}\" \"$@\"", "--"]
