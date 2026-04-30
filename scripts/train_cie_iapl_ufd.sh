#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH=${DATASET_PATH:-/data2/caiguoqing/Datasets/UniversalFakeDetect}
CLIP_PATH=${CLIP_PATH:-/data2/caiguoqing/clip_weights/ViT-L-14.pt}
IAPL_CKPT=${IAPL_CKPT:-/data2/caiguoqing/modelscope_cache/yihengli/IAPL_pretrain/checkpoint_best_acc_progan.pth}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_PORT=${MASTER_PORT:-29580}
EPOCH=${EPOCH:-5}
OUTPUT_DIR=${OUTPUT_DIR:-results}
MODEL_NAME=${MODEL_NAME:-cie_iapl_ufd}

export CUDA_VISIBLE_DEVICES

python -m torch.distributed.launch \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port "${MASTER_PORT}" \
    main.py \
    --model_variant cie_iapl \
    --batchsize 16 \
    --evalbatchsize 32 \
    --dataset_path "${DATASET_PATH}" \
    --clip_path "${CLIP_PATH}" \
    --train_selected_subsets car cat chair horse \
    --test_selected_subsets biggan crn cyclegan deepfake gaugan imle progan san seeingdark stargan stylegan \
    --lr 0.00005 \
    --model_name "${MODEL_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --epoch "${EPOCH}" \
    --lr_drop 10 \
    --gate True \
    --condition True \
    --smooth True \
    --cie_init_from_iapl_ckpt "${IAPL_CKPT}" \
    --cie_eval_mode final \
    --cie_num_specialists 3 \
    --cie_warmup_epochs 2 \
    --cie_gate_warmup_epochs 1 \
    "$@"
