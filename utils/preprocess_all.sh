#!/bin/bash

# Check if dataset path is provided
if [ -z "$1" ]; then
    echo "Error: Please provide the path to the dataset directory."
    echo "Usage: ./utils/preprocess_all.sh <path_to_dataset>"
    echo "Example: ./utils/preprocess_all.sh data/nottingham-dataset-master/MIDI"
    exit 1
fi

DATASET_PATH="$1"

# Check if dataset path exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: Dataset path '$DATASET_PATH' does not exist."
    exit 1
fi

# Infer dataset name from path
# If path ends in MIDI/midi/raw, take parent folder name, else take folder name
# Use name for naming the output files
BASENAME=$(basename "$DATASET_PATH")
if [[ "$BASENAME" == "MIDI" || "$BASENAME" == "midi" || "$BASENAME" == "raw" ]]; then
    PARENT=$(dirname "$DATASET_PATH")
    DATASET_NAME=$(basename "$PARENT")
else
    DATASET_NAME="$BASENAME"
fi

echo "Processing dataset: $DATASET_NAME"
echo "Source Path: $DATASET_PATH"

# output directories
NAIVE_OUTPUT="data/${DATASET_NAME}_naive"
MIDITOK_OUTPUT="data/${DATASET_NAME}_miditok"
MIDITOK_AUG_OUTPUT="data/${DATASET_NAME}_miditok_augmented"

# naive preprocessing
echo "Running Naive Preprocessing..."
python utils/preprocess_naive.py --dataset "$DATASET_PATH" --output_dir "$NAIVE_OUTPUT"

# miditok preprocessing
echo "Running Standard MIDITok Preprocessing..."
python utils/preprocess_miditok.py --dataset "$DATASET_PATH" --output_dir "$MIDITOK_OUTPUT"

# miditok augmented
echo "Running Augmented MIDITok Preprocessing..."
python utils/augment_miditok.py --dataset "$DATASET_PATH" --output_dir "$MIDITOK_AUG_OUTPUT"

echo "Preprocessing Complete!"
echo "Naive data saved to: $NAIVE_OUTPUT"
echo "Standard tokenized data saved to: $MIDITOK_OUTPUT"
echo "Augmented tokenized data saved to: $MIDITOK_AUG_OUTPUT"