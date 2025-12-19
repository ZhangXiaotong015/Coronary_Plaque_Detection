#!/bin/bash
HOST_OUTPUT_DIR="/home/xzhang/inference_project/PlaqueDetectionMaskRCNN/output"
data_test_polarCT="/home/xzhang/inference_project/PlaqueDetectionMaskRCNN/data/polarCT"
data_test_polarCT_25D="/home/xzhang/inference_project/PlaqueDetectionMaskRCNN/data/polarCT_25D"
mkdir -p "$HOST_OUTPUT_DIR"
mkdir -p "$data_test_polarCT"
mkdir -p "$data_test_polarCT_25D"
docker run --rm --gpus "device=0" \
    -v "$HOST_OUTPUT_DIR":/output \
    -v /mnt/e/xiaotong/WSL/TestData/PlaqueDet/oriCT/deeplearning:/data_test_CT \
    -v "$data_test_polarCT":/data_test_polarCT \
    -v "$data_test_polarCT_25D":/data_test_polarCT_25D \
    -e MODEL_PATH=/app/model/40_checkpoint.npz \
    -e SPARSE_MATRIX_PATH=/app/model/lumenCenter_sampleLine_sparseMetrix.npy \
    -e SAMPLE_LINE_PATH=/app/model/lumenCenter_sampleLine.npy \
    plaque_det_maskrcnn:20251119 \
    --output_dir /output \
    --data_test_CT /data_test_CT \
    --data_test_polarCT /data_test_polarCT \
    --data_test_polarCT_25D /data_test_polarCT_25D \
    --model_type denseUNet \
    --eval_batch_size 4 
    # --pred_box_viz

## Interactive
# output="/home/xzhang/inference_project/PlaqueDetectionMaskRCNN/output"
# data_test_polarCT="/home/xzhang/inference_project/PlaqueDetectionMaskRCNN/data/polarCT"
# data_test_polarCT_25D="/home/xzhang/inference_project/PlaqueDetectionMaskRCNN/data/polarCT_25D"
# mkdir -p "$output"
# mkdir -p "$data_test_polarCT"
# mkdir -p "$data_test_polarCT_25D"
# docker run --rm -it --gpus "device=0" \
#     -v "$output":/output \
#     -v /mnt/e/xiaotong/WSL/TestData/PlaqueDet/oriCT/deeplearning:/data_test_CT \
#     -v "$data_test_polarCT":/data_test_polarCT \
#     -v "$data_test_polarCT_25D":/data_test_polarCT_25D \
#     -e MODEL_PATH=/app/model/40_checkpoint.npz \
#     -e SPARSE_MATRIX_PATH=/app/model/lumenCenter_sampleLine_sparseMetrix.npy \
#     -e SAMPLE_LINE_PATH=/app/model/lumenCenter_sampleLine.npy \
#     plaque_det_maskrcnn:20251119 \
#     bash

##### All mounted paths must be exist!!!