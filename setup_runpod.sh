#!/usr/bin/env bash
# Prepare a RunPod container (no Docker) to run this project.
# Only /workspace survives a pod restart, so re-run this script after every restart; it is safe to re-run.
#   bash setup_runpod.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-/workspace/cifar_data}"   # CIFAR is stored here so it is downloaded only once

# point link_path at target; refuse to replace a real file or directory
link() {
    local target="$1" link_path="$2"
    if [ -L "$link_path" ]; then
        ln -sfn "$target" "$link_path"
    elif [ -e "$link_path" ]; then
        echo "error: $link_path exists and is not a symlink; move it away and re-run" >&2
        exit 1
    else
        ln -s "$target" "$link_path"
    fi
    echo "  $link_path -> $target"
}

echo "[1/5] Python packages"
python -c "import easydict" 2>/dev/null || pip install -q easydict
python -c "import scipy" 2>/dev/null || pip install -q scipy   # torchvision reads the ImageNet devkit with scipy
echo "  easydict, scipy ok"

echo "[2/5] /app -> repo (default path in all configs)"
link "$REPO_DIR" /app

echo "[3/5] CIFAR data -> $DATA_DIR"
mkdir -p "$DATA_DIR" /tmp/public_dataset
link "$DATA_DIR" /tmp/public_dataset/pytorch

echo "[4/5] log directories"
mkdir -p "$REPO_DIR/log/resnet20/cifar10" "$REPO_DIR/log/resnet20/cifar100"
echo "  $REPO_DIR/log/resnet20/{cifar10,cifar100}"

echo "[5/5] NeuroSIM (make only rebuilds what changed)"
NEUROSIM_DIR="$REPO_DIR/NeuroSim/Inference_pytorch/NeuroSIM"
command -v g++ >/dev/null && command -v make >/dev/null || { echo "error: g++ and make are required to build NeuroSIM" >&2; exit 1; }
make -s -C "$NEUROSIM_DIR" -j"$(nproc)"
echo "  $NEUROSIM_DIR/main"

python - <<'EOF'
import sys, torch
print(f"\nPython {sys.version.split()[0]}, PyTorch {torch.__version__}, "
      f"CUDA {'available: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT available'}")
EOF
echo "Ready: cd /app && python convert_pretrained.py --net resnet20 --dataset cifar10"
