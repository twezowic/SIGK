# LAB 1

## Instruction:
```sh
# Download High Resolution Images from: https://data.vision.ee.ethz.ch/cvl/DIV2K/ 
# and place them in /data/raw/test and /data/raw/valid or execute script below
sudo chmod +x download.sh
./download.sh

# Download required dependencies
pip install -r requirements.txt

# Prepare data for scalling and inpainting tasks
python preparing_data.py
```