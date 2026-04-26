FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    tmux \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.4 /uv /uvx /bin/
