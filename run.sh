#!/bin/bash
#SBATCH --job-name=iso-attention
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=256GB
#SBATCH --time=4:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# Activate environment
source $SCRATCH/miniconda3/bin/activate
conda activate iso-attention
cd $SCRATCH/projects/iso-attention

# Setup cache directory for nanochat
export NANOCHAT_BASE_DIR="$SCRATCH/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Install Rust if needed
if ! command -v rustc &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
# Only source cargo env if it exists
if [ -f "$SCRATCH/.cargo/env" ]; then
    source "$SCRATCH/.cargo/env"
fi

# Install Python dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e .
pip install maturin
maturin develop --release --manifest-path rustbpe/Cargo.toml

# Download dataset shards for tokenizer training (16 shards = ~4GB)
echo "Downloading dataset shards for tokenizer training..."
python -m nanochat.dataset -n 16

# Train tokenizer (only if not already done)
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    echo "Training tokenizer..."
    python -m scripts.tok_train --max_chars=100000000 --vocab_size=65536
else
    echo "Tokenizer already exists, skipping training"
fi

# Fix distributed training networking issues
# Force IPv4 and get the actual IP address instead of hostname
export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=29500

# Disable IPv6 for PyTorch distributed
export NCCL_SOCKET_IFNAME=^docker,lo
export GLOO_SOCKET_IFNAME=$(ip -4 route get 8.8.8.8 | grep -oP 'dev \K\S+')

# Optional: Enable detailed debugging if needed
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_DEBUG=INFO

echo "Starting training on $(hostname) with MASTER_ADDR=$MASTER_ADDR:$MASTER_PORT"
echo "Using network interface: $GLOO_SOCKET_IFNAME"

# Run distributed training
# Reduced device_batch_size to 4 to fit in 40GB A100 memory
# Gradient accumulation will automatically handle reaching total_batch_size
torchrun \
    --standalone \
    --nproc_per_node=1 \
    -m scripts.base_train --depth=12 --device_batch_size=4

echo "Job finished at $(date)"
