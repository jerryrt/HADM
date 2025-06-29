#!/bin/sh

echo "Activate the conda environment \"hadm\" before running this script"
echo "Evaluating HADM-L ..."
echo "Using pretrained model: HADM-L_0249999.pth"

python tools/lazyconfig_train_net.py --num-gpus 1 --eval-only \
    --config-file projects/ViTDet/configs/eva2_o365_to_coco/eva02_large_local.py \
    train.output_dir=./outputs/eva02_large_local/250k_on_all_val \
    train.init_checkpoint=pretrained_models/HADM-L_0249999.pth \
    dataloader.evaluator.output_dir=cache/large_local_human_artifact_ALL_val/250k_on_all_val \
    dataloader.evaluator.dataset_name=local_human_artifact_val_ALL \
    dataloader.test.dataset.names=local_human_artifact_val_ALL \
    dataloader.train.total_batch_size=1 \
    train.model_ema.enabled=True \
    train.model_ema.use_ema_weights_for_eval_only=True 