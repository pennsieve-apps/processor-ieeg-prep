FROM --platform=linux/amd64 python:3.12.16-slim-bookworm

WORKDIR /processor

# Pinned Python dependencies, binary wheels only (no compiler, no apt packages).
COPY requirements.txt /processor/requirements.txt
RUN pip install --no-cache-dir --only-binary=:all: -r /processor/requirements.txt \
    && find /usr/local/lib/python3.12/site-packages -depth -type d \( -name __pycache__ -o -name tests \) \
            -exec rm -rf {} + \
    && rm -rf /usr/local/lib/python3.12/site-packages/matplotlib/mpl-data/sample_data

COPY processor/ /processor/processor/
COPY VERSION /processor/VERSION

ENV PYTHONPATH="/processor" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    MNE_DONTWRITE_HOME=true \
    MNE_LOGGING_LEVEL=ERROR \
    HOME=/tmp \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2

# Runs as uid 1000 on the platform; nothing here needs root.
USER 1000

ENTRYPOINT ["python3", "-m", "processor.main"]
