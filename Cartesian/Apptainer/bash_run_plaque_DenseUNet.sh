echo "GPU availability check:"
nvidia-smi || echo "No GPU detected."

# --- Ensure module system is available ---
source /etc/profile.d/modules.sh

# --- Load Apptainer module ---
module load container/apptainer/1.4.1

# --- Base directory for all Apptainer data ---
BASE=/exports/lkeb-hpc/xzhang/docker_archive

# --- 1. Set Apptainer tmp & cache to your large persistent directory ---
export APPTAINER_TMPDIR=$BASE/apptainer_tmp
export APPTAINER_CACHEDIR=$BASE/apptainer_cache

mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
chmod 700 "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

# --- 2. Where to store your .sif ---
export IMAGEDIR=$BASE/apptainer_images
mkdir -p "$IMAGEDIR"
chmod 700 "$IMAGEDIR"

# --- 3. Path to your Docker archive (.tar) ---
TARFILE=$BASE/plaque_det_denseunet.tar

# --- 4. Build SIF only if not already present ---
if [ ! -f "$IMAGEDIR/plaque_det_denseunet.sif" ]; then
    echo "Building SIF from $TARFILE ..."
    apptainer build "$IMAGEDIR/plaque_det_denseunet.sif" docker-archive://"$TARFILE"
fi

# --- 5. Prepare output and data directories ---
echo "Running DenseUNet container..."

OUT_BASE=/exports/lkeb-hpc/xzhang/docker_archive/PlaqueDetection_DenseUNet

OUTDIR=$OUT_BASE/output
OUTDIR_Chemo=$OUT_BASE/output_chemogram
Data_test_25D=$OUT_BASE/data/CT_25D

mkdir -p "$OUTDIR" "$OUTDIR_Chemo" "$Data_test_25D"

DATA_CT=/exports/lkeb-hpc/xzhang/docker_archive/data/PlaqueDet/oriCT/deeplearning

# --- 6. Run DenseUNet container ---
apptainer run --nv \
    --bind "$OUTDIR:/output" \
    --bind "$OUTDIR_Chemo:/output_chemo" \
    --bind "$DATA_CT:/data_test_CT:ro" \
    --bind "$Data_test_25D:/data_test_CT_25D" \
    "$IMAGEDIR/plaque_det_denseunet.sif" \
        --test_save_path /output \
        --chemogram_save_path /output_chemo \
        --path_test_CT /data_test_CT \
        --path_test /data_test_CT_25D \
        --model_type denseUNet \
        --batch_size 4
