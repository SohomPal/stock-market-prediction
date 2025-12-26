#!/bin/bash
#SBATCH -A m4431_g
#SBATCH -C gpu                  # use &hbm80g ONLY if you need >40GB
#SBATCH -q regular
#SBATCH -t 02:00:00

#SBATCH -N 1                    # single node
#SBATCH --ntasks=1              # single process
#SBATCH --gpus-per-task=1       # single GPU
#SBATCH --cpus-per-task=8       # CPU workers for DataLoader

#SBATCH -J lstm_train_1gpu
#SBATCH -o logs/lstm_train_%j.out
#SBATCH -e logs/lstm_train_%j.err

# ============================
# Environment
# ============================
module load python/3.12
source activate nersc-python

cd $HOME/StockMarketPrediction

# ============================
# Performance knobs
# ============================
export OMP_NUM_THREADS=8
export PYTHONFAULTHANDLER=1
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDA_VISIBLE_DEVICES=0

# ============================
# Run training
# ============================
python train.py \
  --data_dir $PSCRATCH/StockPrediction/LSTM_Data \
  --outdir $PSCRATCH/StockPrediction/models \
  --batch_size 16384 \
  --val_batch_size 4096 \
  --epochs 10 \
  --hidden_dim 512 \
  --num_layers 4 \
  --num_workers 8 \
  --lr 1e-4
