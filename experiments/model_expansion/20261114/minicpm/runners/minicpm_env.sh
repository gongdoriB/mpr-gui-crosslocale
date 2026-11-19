#!/usr/bin/env bash
# Source from any directory. Repo-local caches only; no credentials saved.
MINICPM_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export XDG_CACHE_HOME="${MINICPM_REPO_ROOT}/.cache"
export HF_HOME="${MINICPM_REPO_ROOT}/.cache/huggingface"
export TORCH_HOME="${MINICPM_REPO_ROOT}/.cache/torch"
export PIP_CACHE_DIR="${MINICPM_REPO_ROOT}/.cache/pip"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
export TOKENIZERS_PARALLELISM=false
export MINICPM_MODEL_ID=openbmb/MiniCPM-V-4_5
export MINICPM_REVISION=14a2004908140699317e2109e90573c1e6323c36
export MODEL_CONFIG="01_code/configs/models/minicpm_v4_5_8b_4090.yaml"
export CFG_SHA=298953fe68ae62368e1456b9a61766988058617f9cdab22c4eec296c236bb775
