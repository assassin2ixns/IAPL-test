#!/usr/bin/env bash
set -euo pipefail

JPEG75_PATH=${JPEG75_PATH:-/data2/caiguoqing/Datasets_B/UniversalFakeDetect_jpeg75}
CLIP_PATH=${CLIP_PATH:-/data2/caiguoqing/clip_weights/ViT-L-14.pt}
PRETRAINED_MODEL=${PRETRAINED_MODEL:-results/cie_iapl_ufd/checkpoint_best_acc.pth}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_PORT=${MASTER_PORT:-29582}

export CUDA_VISIBLE_DEVICES

python -m torch.distributed.launch \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port "${MASTER_PORT}" \
    main.py \
    --model_variant cie_iapl \
    --eval \
    --batchsize 16 \
    --evalbatchsize 32 \
    --dataset_path "${JPEG75_PATH}" \
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
    --cie_eval_mode final \
    "$@"
