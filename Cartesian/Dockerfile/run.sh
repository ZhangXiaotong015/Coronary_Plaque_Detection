#!/bin/bash
OUTDIR="/home/xzhang/inference_project/PlaqueDetection/output"
OUTDIR_Chemo="/home/xzhang/inference_project/PlaqueDetection/output_chemogram"
Data_test_25D="/home/xzhang/inference_project/PlaqueDetection/data/CT_25D"
mkdir -p "$OUTDIR"
mkdir -p "$OUTDIR_Chemo"
mkdir -p "$Data_test_25D"
docker run --rm --gpus "device=0" \
    -v "$OUTDIR":/output \
    -v "$OUTDIR_Chemo":/output_chemo \
    -v /mnt/e/xiaotong/WSL/TestData/PlaqueDet/oriCT/deeplearning:/data_test_CT \
    -v "$Data_test_25D":/data_test_CT_25D \
    -e MODEL_PATH=/app/model/model_041.hdf5 \
    -e SPARSE_MATRIX_PATH=/app/model/lumenCenter_sampleLine_sparseMetrix.npy \
    -e SAMPLE_LINE_PATH=/app/model/lumenCenter_sampleLine.npy \
    plaque_det_denseunet:20251119 \
    --test_save_path /output \
    --chemogram_save_path /output_chemo \
    --path_test_CT /data_test_CT \
    --path_test /data_test_CT_25D \
    --model_type denseUNet \
    --batch_size 4

## Interactive 
# OUTDIR="/home/xzhang/inference_project/PlaqueDetection/output"
# OUTDIR_Chemo="/home/xzhang/inference_project/PlaqueDetection/output_chemogram"
# Data_test_25D="/home/xzhang/inference_project/PlaqueDetection/data/CT_25D"
# mkdir -p "$OUTDIR"
# mkdir -p "$OUTDIR_Chemo"
# mkdir -p "$Data_test_25D"
# docker run --rm -it --gpus "device=0" \
#     -v "$OUTDIR":/output \
#     -v "$OUTDIR_Chemo":/output_chemo \
#     -v /mnt/e/xiaotong/WSL/TestData/PlaqueDet/oriCT/deeplearning:/data_test_CT \
#     -v "$Data_test_25D":/data_test_CT_25D \
#     -e MODEL_PATH=/app/model/model_041.hdf5 \
#     -e SPARSE_MATRIX_PATH=/app/model/lumenCenter_sampleLine_sparseMetrix.npy \
#     -e SAMPLE_LINE_PATH=/app/model/lumenCenter_sampleLine.npy \
#     plaque_det:20251119 \
#     bash

##### All mounted paths must be exist!!!