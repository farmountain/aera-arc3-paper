# ARC-AGI-3 AERA Agent — reproducible evaluation environment
#
# Build:
#   docker build -t aera-arc3 .
#
# Run EXP-002 (AERA with explore) on 5 public games:
#   docker run --rm \
#     -v ~/models:/models:ro \
#     -v $(pwd)/runs:/out \
#     aera-arc3 python -m run_eval \
#       --model /models/qwen2.5-7b-instruct-q4_k_m.gguf \
#       --games sb26 ft09 cd82 tu93 r11l \
#       --out-dir /out
#
# Run EXP-001 (no-explore baseline):
#   docker run --rm ... python -m run_eval ... --no-explore

FROM python:3.11-slim

# System dependencies for llama-cpp-python CPU build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (pinned — see requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent source (model weights NOT included — mount at runtime)
COPY agent.py env_wrapper.py llm_backend.py run_eval.py ./

# Sanity check: confirm imports work
RUN python -c "from run_eval import main; print('imports OK')"

# Default: show help
CMD ["python", "-m", "run_eval", "--help"]
