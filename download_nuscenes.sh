#!/bin/bash
set -e

# === Configuration ===
TARGET_DIR="./nuscenes_data"
START_INDEX=1
END_INDEX=10  # Change to 10 for full dataset

# Create directory
mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}"

BASE_URL="https://motional-nuscenes.s3.amazonaws.com/public/v1.0"

# --- STEP 1: Download Metadata (CRITICAL) ---
if [ ! -d "v1.0-trainval" ]; then
    echo "⬇️  Downloading Metadata (v1.0-trainval_meta.tgz)..."
    wget -c "${BASE_URL}/v1.0-trainval_meta.tgz" -O v1.0-trainval_meta.tgz
    
    echo "📦 Extracting Metadata..."
    tar -xf v1.0-trainval_meta.tgz
    rm v1.0-trainval_meta.tgz
else
    echo "✅ Metadata already exists."
fi

# --- STEP 2: Download & Extract Blobs ---
echo "⬇️  Downloading Blobs ${START_INDEX} to ${END_INDEX}..."

for ((i=START_INDEX; i<=END_INDEX; i++)); do
    # Force 2-digit zero padding (1 -> 01)
    PADDED_I=$(printf "%02d" $i)
    
    FILE="v1.0-trainval${PADDED_I}_blobs.tgz"
    URL="${BASE_URL}/${FILE}"

    # Download
    if [ ! -f "${FILE}" ]; then
        echo "Downloading ${FILE}..."
        wget -c "${URL}" -O "${FILE}"
    else
        echo "${FILE} already downloaded. Skipping."
    fi

    # Extract
    echo "📦 Extracting ${FILE}..."
    tar -xf "${FILE}"
    
    # Optional: Delete tarball after extraction to save space
    rm "${FILE}" 
done

echo "🎉 Dataset Setup Complete in $(pwd)"