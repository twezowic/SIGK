# LAB 2

## Pobieranie danych:
```sh
# Download HDR Images from: https://www.repository.cam.ac.uk/bitstreams/3e871fb5-0cb9-4376-913d-5f5e5006a5c4/download
# and place them in /data/raw or execute script below
sudo chmod +x download.sh
./download.sh

# Download required dependencies
pip install -r ./../requirements.txt

# Prepare data for scalling and inpainting tasks
python preparing_data.py
```

## Zadanie

Wejście to obrazy HDR o rozdzielczości 256 x 256, zaś wyjście to obrazy przetworzone w celu uzyskania odpowiedniego odwzorowania tonalnego.

Wyniki znajdują się w pliku [note.ipynb](note.ipynb).
