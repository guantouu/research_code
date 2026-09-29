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

echo "[1/4] Python packages"
python -c "import easydict" 2>/dev/null || pip install -q easydict
echo "  easydict ok"

echo "[2/4] /app -> repo (default path in all configs)"
link "$REPO_DIR" /app

echo "[3/4] CIFAR data -> $DATA_DIR"
mkdir -p "$DATA_DIR" /tmp/public_dataset
link "$DATA_DIR" /tmp/public_dataset/pytorch

echo "[4/4] log directories"
mkdir -p "$REPO_DIR/log/resnet20/cifar10" "$REPO_DIR/log/resnet20/cifar100"
echo "  $REPO_DIR/log/resnet20/{cifar10,cifar100}"

python - <<'EOF'
import sys, torch
print(f"\nPython {sys.version.split()[0]}, PyTorch {torch.__version__}, "
      f"CUDA {'available: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT available'}")
EOF
echo "Ready: cd /app && python convert_pretrained.py --dataset cifar10"
