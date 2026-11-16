ARG BASE_IMAGE=python:3.12@sha256:17b9724493d916f56565c131f389faff2597f46dec6ce09516f95125bc9f519f
FROM ${BASE_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
ENV NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=/candidate/src:/candidate
# Start with exactly the supplied snapshot. Offline ensurepip later supplies
# the installer from the pinned Python base without resolving dependencies.
RUN rm -rf /usr/local/lib/python3.12/site-packages && mkdir -p /usr/local/lib/python3.12/site-packages
COPY site-packages/ /usr/local/lib/python3.12/site-packages/
COPY snapshot.files.json runtime.snapshot.lock.json runtime.observed.requirements.txt verify_ml_snapshot.py verify_ml_runtime.py /opt/passage-image-inputs/
COPY core-wheels/ /opt/passage-image-inputs/core-wheels/
RUN python /opt/passage-image-inputs/verify_ml_snapshot.py \
 && python -m ensurepip --default-pip \
 && python -m pip install --no-index --no-deps --force-reinstall /opt/passage-image-inputs/core-wheels/*.whl \
 && CUDA_VISIBLE_DEVICES="" python /opt/passage-image-inputs/verify_ml_runtime.py
WORKDIR /candidate
ENTRYPOINT ["python", "-m", "passage_eval.campaign.candidate.worker"]
CMD ["--manifest", "/candidate/candidate.json", "--inputs", "/inputs/scenes.npz", "--output", "/output", "--device", "cpu"]
