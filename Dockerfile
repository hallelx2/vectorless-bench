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
RUN pip install ".[llm,vector,bm25,data,viz]"

# The vectorless SDK isn't on PyPI. Prefer the monorepo source staged into the
# build context by deploy/vendor_sdk.sh (./vendor/vectorless-sdk); fall back to
# PyPI best-effort so the image still builds for baseline-only runs.
RUN if [ -d vendor/vectorless-sdk ]; then \
        pip install ./vendor/vectorless-sdk; \
    else \
        pip install "vectorless-sdk>=0.1" || echo "WARN: vectorless-sdk unavailable — the vectorless system will be skipped"; \
    fi

# Vendor PageIndex's actual repo (not on PyPI) so the pageindex baseline can run
# their real tree builder. Their requirements.txt self-conflicts (pins
# python-dotenv==1.2.2 while litellm needs 1.0.1), so drop that pin and let pip
# resolve it. The whole step is best-effort: pageindex isn't needed for every run
# (e.g. Gemini-only), and a failure here must not block the build.
RUN git clone --depth 1 https://github.com/VectifyAI/PageIndex.git /opt/PageIndex \
    && grep -v '^python-dotenv' /opt/PageIndex/requirements.txt > /tmp/pi-req.txt \
    && pip install -r /tmp/pi-req.txt \
    || echo "WARN: PageIndex install failed; the pageindex system will be unavailable"

# results land here; mount a volume so they survive the container
VOLUME ["/results"]

ENTRYPOINT ["vlbench"]
CMD ["run", "--config", "configs/financebench.yaml", "--out", "/results"]
