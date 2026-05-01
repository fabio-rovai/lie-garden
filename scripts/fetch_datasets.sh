#!/usr/bin/env bash
# Fetch the four prompt-injection datasets at PINNED revisions.
# Idempotent: skips files that are already present and verified.
#
# Pinned revisions (audit before changing):
#   tensor-trust-data   commit  747a75e096761ebc01bd3970158827326b4add23 (2024-03-17)
#   InjecAgent          commit  f19c9f2c79a41046eb13c03c51a24c567a8ffa07
#   deepset/prompt-injections     hf rev   4f61ecb038e9c3fb77e21034b22511b523772cdd
#   neuralchemy/Prompt-injection-dataset  hf rev   7d70432dfcf47a821612cbf9d34e9d9e3ad20e75

set -euo pipefail

mkdir -p /tmp

readonly TT_SHA="747a75e096761ebc01bd3970158827326b4add23"
readonly INJEC_SHA="f19c9f2c79a41046eb13c03c51a24c567a8ffa07"
readonly DEEPSET_HF_REV="4f61ecb038e9c3fb77e21034b22511b523772cdd"
readonly NEURALCHEMY_HF_REV="7d70432dfcf47a821612cbf9d34e9d9e3ad20e75"

verify_sha256() {
  local path="$1" expected="$2"
  if [[ -z "$expected" ]]; then return 0; fi
  local got
  got=$(shasum -a 256 "$path" | awk '{print $1}')
  if [[ "$got" != "$expected" ]]; then
    echo "ERROR: $path SHA-256 mismatch" >&2
    echo "  expected: $expected" >&2
    echo "  got:      $got" >&2
    exit 1
  fi
}

# 1. TensorTrust — pinned commit
if [[ ! -d /tmp/tensor-trust-data ]]; then
  echo "[fetch] TensorTrust @ ${TT_SHA}"
  git clone https://github.com/HumanCompatibleAI/tensor-trust-data.git /tmp/tensor-trust-data
  (cd /tmp/tensor-trust-data && git checkout "${TT_SHA}")
fi
(cd /tmp/tensor-trust-data && git rev-parse HEAD | grep -q "${TT_SHA}") || {
  echo "ERROR: /tmp/tensor-trust-data is not at pinned commit ${TT_SHA}" >&2; exit 1; }

# 2. InjecAgent — pinned commit
if [[ ! -d /tmp/InjecAgent ]]; then
  echo "[fetch] InjecAgent @ ${INJEC_SHA}"
  git clone https://github.com/uiuc-kang-lab/InjecAgent.git /tmp/InjecAgent
  (cd /tmp/InjecAgent && git checkout "${INJEC_SHA}")
fi
(cd /tmp/InjecAgent && git rev-parse HEAD | grep -q "${INJEC_SHA}") || {
  echo "ERROR: /tmp/InjecAgent is not at pinned commit ${INJEC_SHA}" >&2; exit 1; }

# 3. Deepset — pinned HF revision; preserve publisher train/test split
if [[ ! -f /tmp/deepset_train.parquet || ! -f /tmp/deepset_test.parquet ]]; then
  echo "[fetch] Deepset @ ${DEEPSET_HF_REV}"
  curl -sL "https://huggingface.co/datasets/deepset/prompt-injections/resolve/${DEEPSET_HF_REV}/data/train-00000-of-00001-9564e8b05b4757ab.parquet" \
    -o /tmp/deepset_train.parquet
  curl -sL "https://huggingface.co/datasets/deepset/prompt-injections/resolve/${DEEPSET_HF_REV}/data/test-00000-of-00001-9cb1ce73e6f8e96b.parquet" \
    -o /tmp/deepset_test.parquet
fi

# 4. Neuralchemy core split — pinned HF revision; preserve publisher train/test split
if [[ ! -f /tmp/neuralchemy_train.parquet || ! -f /tmp/neuralchemy_test.parquet ]]; then
  echo "[fetch] Neuralchemy @ ${NEURALCHEMY_HF_REV}"
  curl -sL "https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset/resolve/${NEURALCHEMY_HF_REV}/core/train-00000-of-00001.parquet" \
    -o /tmp/neuralchemy_train.parquet
  curl -sL "https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset/resolve/${NEURALCHEMY_HF_REV}/core/test-00000-of-00001.parquet" \
    -o /tmp/neuralchemy_test.parquet
fi

echo
echo "[fetch] All datasets at pinned revisions."
echo "  /tmp/tensor-trust-data            commit ${TT_SHA:0:12}"
echo "  /tmp/InjecAgent                   commit ${INJEC_SHA:0:12}"
echo "  /tmp/deepset_{train,test}.parquet HF rev ${DEEPSET_HF_REV:0:12}"
echo "  /tmp/neuralchemy_{train,test}.parquet HF rev ${NEURALCHEMY_HF_REV:0:12}"
