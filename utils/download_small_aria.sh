#!/bin/bash

# exit immediately if a command exits with non-zero status
set -e

echo "activating venv"
# check if venv exists before trying to use as source
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "venv activated. yippee"
else
    echo "error: venv not found.\nMAKE SURE TO USE uv sync FROM THE csce585-midi DIRECTORY!!!"
    exit 1
fi

# create 'data' directory if it doesn't exist and navigate into it
mkdir -p data/
cd data/

echo "downloading dataset..."
wget -O aria-midi-v1.tar.gz "https://huggingface.co/datasets/loubb/aria-midi/resolve/main/aria-midi-v1-unique-ext.tar.gz?download=true"

echo "extracting data :)"
tar -xvf aria-midi-v1.tar.gz

cd aria-midi-v1-unique-ext/data/

# check for arg saying if we should remove files
if [ "$1" == "shrink"]; then
    echo "shrinking arg detected. deleting all but 2 directories of aria dataset"

    ls | grep a. | head -n 26 | xargs rm -r
    ls | grep b. | head -n 26 | xargs rm -r
    ls | grep c. | head -n 26 | xargs rm -r
    ls | grep d. | head -n 24 | xargs rm -r
    echo "dataset reduction complete... two directories remaining:"
    ls
else
    echo "shrink arg not provided, keeping full dataset."
fi

echo "finished"

exit 0