#!/usr/bin/env bash
# One-shot environment setup for a RunPod GPU pod (official PyTorch template).
# Idempotent: safe to re-run. Ends by running the step-zero verification.
#
# Assumes: torch with CUDA already installed (any RunPod pytorch template),
# nvcc available for the flash-attn source-build fallback (-devel images).
set -uo pipefail

WORKDIR="${WORKDIR:-/workspace}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

step() { echo; echo "=== $* ==="; }
die() { echo "FATAL: $*" >&2; exit 1; }

step "GPU visibility"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || die "nvidia-smi failed — no GPU on this pod?"

step "torch sanity"
python - <<'EOF' || die "torch missing or CPU-only. Use a RunPod PyTorch template (torch>=2.4, CUDA 12.x)."
import torch
assert torch.cuda.is_available(), "torch installed but CUDA unavailable"
print(f"torch {torch.__version__} cu{torch.version.cuda} on {torch.cuda.get_device_name(0)}")
EOF

# Version pins discovered the hard way (2026-07-03, see README caveats):
#  * NSA's vendored fla needs torch>=2.5 (torch.distributed.tensor.DeviceMesh)
#  * fla vendored at the NSA repo's pinned commit targets triton 3.0/3.1;
#    PyPI flash-linear-attention 0.5.x needs triton>=3.3 and upgrades torch —
#    never install it. Expose the vendored copy via a .pth instead.
#  * transformers must predate its own 'bitnet' arch (fla registers one): 4.47.1
#  * every pip install that could touch torch runs with --no-deps
TORCH_PIN="2.5.1"
TV_PIN="0.20.1"
CU_INDEX="https://download.pytorch.org/whl/cu124"

step "build tooling + python deps"
pip install -q -U pip ninja packaging wheel
pip install -q einops "transformers==4.47.1" pandas matplotlib

step "torch ${TORCH_PIN}+cu124 (pinned)"
python - <<EOF || pip install --index-url "$CU_INDEX" "torch==${TORCH_PIN}" "torchvision==${TV_PIN}"
import sys, torch
sys.exit(0 if torch.__version__.startswith("${TORCH_PIN}") else 1)
EOF

step "flash-attn (FlashAttention-2 dense baseline)"
if python -c "import flash_attn" 2>/dev/null; then
    echo "flash-attn already installed and importable: $(python -c 'import flash_attn; print(flash_attn.__version__)')"
else
    # --no-deps: flash-attn's metadata must never upgrade torch underneath us
    MAX_JOBS=8 pip install flash-attn --no-build-isolation --no-deps || die "flash-attn install failed — check torch/CUDA combo, see verify_env.py hints"
fi

step "fla-org/native-sparse-attention (NSA Triton kernels)"
NSA_DIR="$WORKDIR/native-sparse-attention"
if [ ! -d "$NSA_DIR/.git" ]; then
    git clone https://github.com/fla-org/native-sparse-attention.git "$NSA_DIR" || die "clone failed"
fi
git -C "$NSA_DIR" submodule update --init --recursive
pip install -e "$NSA_DIR" --no-deps || die "NSA install failed"
# expose the repo root so the VENDORED fla (pinned, triton-3.1-compatible) is
# importable; remove any leftover PyPI fla that would shadow it
SITE_DIR="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
rm -rf "$SITE_DIR/fla" "$SITE_DIR"/fla-* "$SITE_DIR"/flash_linear_attention*
echo "$NSA_DIR" > "$SITE_DIR/nsa_repo.pth"
python -c "import fla, os; assert os.path.realpath(fla.__file__).startswith(os.path.realpath('$NSA_DIR')), fla.__file__" || die "fla resolves to the wrong package"

step "guard: torch untouched by later installs"
python -c "import torch; assert torch.__version__.startswith('${TORCH_PIN}'), f'torch drifted to {torch.__version__}'" || die "torch version drifted during setup"

step "record installed versions"
mkdir -p "$REPO_DIR/results"
pip freeze > "$REPO_DIR/results/pip_freeze.txt"
echo "wrote results/pip_freeze.txt"

step "step-zero verification (imports + trivial forward passes)"
python "$REPO_DIR/setup/verify_env.py"
