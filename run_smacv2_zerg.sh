#!/bin/bash

# ==============================================================================
# HARL-MAGPO: 4-GPU Concurrent Training Script for SMACv2 (zerg_5_vs_5)
# ==============================================================================

# 1. 默认参数设置 (Default Arguments)
SEED=1
LOG_DIR="./results/smacv2_zerg_5_vs_5"
SAVE_FREQ=10

# 2. 解析命令行传参
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --seed) SEED="$2"; shift ;;
        --log_dir) LOG_DIR="$2"; shift ;;
        --save_freq) SAVE_FREQ="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "=================================================================="
echo "🚀 Starting 4 Concurrent Experiments on SMACv2 (zerg_5_vs_5)"
echo "▶ SEED      : $SEED"
echo "▶ LOG_DIR   : $LOG_DIR"
echo "▶ SAVE_FREQ : $SAVE_FREQ"
echo "=================================================================="

# 3. 配置文件路径
CONFIG_HAPPO="tuned_configs/smacv2/zerg_5_vs_5/happo/config.json"
CONFIG_HAPPO_DEEP="tuned_configs/smacv2/zerg_5_vs_5/happo_deep/config.json"
CONFIG_MAGPO="tuned_configs/smacv2/zerg_5_vs_5/magpo/config.json"
CONFIG_PRL="tuned_configs/smacv2/zerg_5_vs_5/prl/config.json"

# 创建总日志目录
mkdir -p $LOG_DIR

# 4. 在 4 张卡上并发启动实验 (使用 nohup 放至后台)

# [GPU 0] - Vanilla HAPPO (Lower Bound)
echo "🔥 [GPU 0] Launching Vanilla HAPPO..."
CUDA_VISIBLE_DEVICES=0 nohup python examples/train.py \
    --load_config $CONFIG_HAPPO \
    --seed $SEED \
    --log_dir $LOG_DIR/happo_seed$SEED \
    --save_backup_interval $SAVE_FREQ \
    > $LOG_DIR/happo_run.log 2>&1 &

# [GPU 1] - HAPPO Deep (Architecture Ablation)
echo "🔥 [GPU 1] Launching HAPPO Deep..."
CUDA_VISIBLE_DEVICES=1 nohup python examples/train.py \
    --load_config $CONFIG_HAPPO_DEEP \
    --seed $SEED \
    --log_dir $LOG_DIR/happo_deep_seed$SEED \
    --save_backup_interval $SAVE_FREQ \
    > $LOG_DIR/happo_deep_run.log 2>&1 &

# [GPU 2] - MAGPO (Explicit Distillation Baseline)
echo "🔥 [GPU 2] Launching MAGPO..."
CUDA_VISIBLE_DEVICES=2 nohup python examples/train.py \
    --load_config $CONFIG_MAGPO \
    --seed $SEED \
    --log_dir $LOG_DIR/magpo_seed$SEED \
    --save_backup_interval $SAVE_FREQ \
    > $LOG_DIR/magpo_run.log 2>&1 &

# [GPU 3] - PRL (Ours - Proposed Representation Learning)
echo "🔥[GPU 3] Launching PRL (Ours)..."
CUDA_VISIBLE_DEVICES=3 nohup python examples/train.py \
    --load_config $CONFIG_PRL \
    --seed $SEED \
    --log_dir $LOG_DIR/prl_ours_seed$SEED \
    --save_backup_interval $SAVE_FREQ \
    > $LOG_DIR/prl_run.log 2>&1 &

echo "=================================================================="
echo "✅ All 4 experiments successfully launched in the background!"
echo "📄 You can check the output logs in: $LOG_DIR/"
echo "💡 Tip: Use 'tail -f $LOG_DIR/prl_run.log' to monitor training."
echo "=================================================================="