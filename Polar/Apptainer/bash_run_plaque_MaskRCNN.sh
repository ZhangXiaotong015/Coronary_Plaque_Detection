echo "Using GPUs:"
nvidia-smi || echo "No GPU visible"

# --- Ensure module system is available ---
source /etc/profile.d/modules.sh

# --- Load Apptainer module ---
module load container/apptainer/1.4.1

# --- Base directory for all Apptainer data ---
BASE=/exports/lkeb-hpc/xzhang/docker_archive
mkdir -p $BASE/slurm_logs

# --- 1. Set Apptainer tmp & cache to your large persistent directory ---
export APPTAINER_TMPDIR=$BASE/apptainer_tmp
export APPTAINER_CACHEDIR=$BASE/apptainer_cache

mkdir -p "$APPTAINER_TMPDIR"
mkdir -p "$APPTAINER_CACHEDIR"
chmod 700 "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

# --- 2. Where to store your .sif ---
export IMAGEDIR=$BASE/apptainer_images
mkdir -p "$IMAGEDIR"
chmod 700 "$IMAGEDIR"

# --- 3. Path to your Docker archive (.tar) ---
TARFILE=$BASE/plaque_det_maskrcnn.tar

# --- 4. Build SIF only if not already present ---
if [ ! -f "$IMAGEDIR/plaque_det_maskrcnn.sif" ]; then
    echo "Building SIF from $TARFILE ..."
    apptainer build "$IMAGEDIR/plaque_det_maskrcnn.sif" docker-archive://$TARFILE
fi

# --- 5. Prepare output and data directories ---
echo "Running container..."

OUT_BASE=/exports/lkeb-hpc/xzhang/docker_archive/PlaqueDetection_MaskRCNN

HOST_OUTPUT_DIR="$OUT_BASE/output"
data_test_polarCT="$OUT_BASE/data/polarCT"
data_test_polarCT_25D="$OUT_BASE/data/polarCT_25D"

mkdir -p "$HOST_OUTPUT_DIR" "$data_test_polarCT" "$data_test_polarCT_25D"

DATA_CT=/exports/lkeb-hpc/xzhang/docker_archive/data/PlaqueDet/oriCT/deeplearning

# --- 6. Run MaskRCNN container on GPU ---
apptainer run --nv \
    --bind "$HOST_OUTPUT_DIR:/output" \
    --bind "$DATA_CT:/data_test_CT" \
    --bind "$data_test_polarCT:/data_test_polarCT" \
    --bind "$data_test_polarCT_25D:/data_test_polarCT_25D" \
    --bind "$data_test_polarCT:/data_test_CTPolar_rcnn" \
    --env MODEL_PATH=/app/model/40_checkpoint.npz \
    --env SPARSE_MATRIX_PATH=/app/model/lumenCenter_sampleLine_sparseMetrix.npy \
    --env SAMPLE_LINE_PATH=/app/model/lumenCenter_sampleLine.npy \
    "$IMAGEDIR/plaque_det_maskrcnn.sif" \
        --output_dir /output \
        --data_test_CT /data_test_CT \
        --data_test_polarCT /data_test_polarCT \
        --data_test_polarCT_25D /data_test_polarCT_25D \
        --model_type denseUNet \
        --eval_batch_size 4 \
        --pred_box_viz  # optional
