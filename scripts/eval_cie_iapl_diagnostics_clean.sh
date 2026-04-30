#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH=${DATASET_PATH:-/data2/caiguoqing/Datasets/UniversalFakeDetect}
CLIP_PATH=${CLIP_PATH:-/data2/caiguoqing/clip_weights/ViT-L-14.pt}
PRETRAINED_MODEL=${PRETRAINED_MODEL:-results/cie_iapl_v11_ufd/checkpoint_best_acc.pth}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
OUTPUT_DIR=${OUTPUT_DIR:-results}
MODEL_NAME=${MODEL_NAME:-cie_iapl_v11_diag_clean}

export CUDA_VISIBLE_DEVICES

python tools/eval_cie_iapl_diagnostics.py \
    --model_variant cie_iapl \
    --eval \
    --batchsize 16 \
    --evalbatchsize 32 \
    --dataset_path "${DATASET_PATH}" \
    --clip_path "${CLIP_PATH}" \
    --pretrained_model "${PRETRAINED_MODEL}" \
    --train_selected_subsets car cat chair horse \
    --test_selected_subsets biggan crn cyclegan deepfake gaugan imle progan san seeingdark stargan stylegan \
    --lr 0.00005 \
    --epoch 5 \
    --lr_drop 10 \
    --gate True \
    --condition True \
    --smooth True \
    --output_dir "${OUTPUT_DIR}" \
    --model_name "${MODEL_NAME}" \
    "$@"
