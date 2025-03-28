

#!/bin/bash
URL_TRAIN="https://www.repository.cam.ac.uk/bitstreams/3e871fb5-0cb9-4376-913d-5f5e5006a5c4/download"

DEST_TRAIN="./data/reference"
mkdir -p "$DEST_TRAIN"

TEMP_DIR="/tmp/si_hdr"
mkdir -p "$TEMP_DIR"

wget -O "$TEMP_DIR/reference.zip" "$URL_TRAIN"

unzip "$TEMP_DIR/reference.zip" -d "$TEMP_DIR"

mv "$TEMP_DIR/train"/DIV2K_train_HR/* "$DEST_TRAIN/"

rm -rf "$TEMP_DIR"
