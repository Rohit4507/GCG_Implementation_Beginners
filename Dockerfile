# Reproducible harness for the GCG / PAIR implementations in this repo.
#
# CPU-only by design: the offline smoke test and tiny-model experiments run
# anywhere (this is how CI and reviewers can verify the pipeline without a GPU
# or Hugging Face access). For real-model runs, build with GPU support or run
# the image on a CUDA host; the experiment entry points are unchanged.
#
#   docker build -t gcg-scratch .
#   docker run --rm gcg-scratch                       # runs the offline smoke test
#   docker run --rm -it gcg-scratch bash              # poke around
#   docker run --rm gcg-scratch python -m experiments.run_algorithm1 --tiny --benchmark
#
# Plain `docker run gcg-scratch` is the "does this actually work?" check: it
# runs the same 7-check pipeline validation recorded in results/reports/.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU wheels keep the image ~1.5 GB instead of pulling the CUDA stack.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    || pip install torch
RUN pip install -r requirements.txt

COPY . .

# Fail the build if the pipeline is broken: 7 offline checks, seconds on CPU.
RUN python -m experiments.run_smoke \
    && python -m unittest discover -s tests

CMD ["python", "-m", "experiments.run_smoke"]
