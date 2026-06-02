# The reproducible benchmark bundle: the suite + every baseline's deps + the real
# PageIndex repo, in one image you can ship to a VM and run unattended.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VLBENCH_PAGEINDEX_PATH=/opt/PageIndex

# git: to vendor PageIndex. build-essential: some wheels (psycopg, tiktoken).
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the suite + all public baseline deps. vectorless-sdk is best-effort:
# if it isn't published to your PyPI, the build still succeeds and you mount it
# or set PYTHONPATH at run time (only the `vectorless` system needs it).
COPY . /app
RUN pip install ".[llm,vector,bm25,data,viz]" \
    && (pip install "vectorless-sdk>=0.1" \
        || echo "WARN: vectorless-sdk not installed — mount it for the vectorless system")

# Vendor PageIndex's actual repo (not on PyPI) and install its requirements, so
# the pageindex baseline runs their real tree builder. Best-effort: upstream's
# pinned requirements.txt sometimes has internal conflicts (e.g. litellm vs
# python-dotenv → ResolutionImpossible). The clone is cheap and only the
# `pageindex` baseline needs these deps, so a failed install must NOT break the
# image for every other system. Retry without the broken pins, else skip.
RUN git clone --depth 1 https://github.com/VectifyAI/PageIndex.git /opt/PageIndex || true; \
    pip install -r /opt/PageIndex/requirements.txt \
      || pip install litellm pymupdf PyPDF2 python-dotenv pyyaml tiktoken openai \
      || echo "WARN: PageIndex deps skipped — the 'pageindex' baseline will be unavailable"

# results land here; mount a volume so they survive the container
VOLUME ["/results"]

ENTRYPOINT ["vlbench"]
CMD ["run", "--config", "configs/financebench.yaml", "--out", "/results"]
