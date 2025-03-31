#!/bin/bash

# Define dataset paths
dataset_a="/zhome/d4/a/214319/adlcv_project/data/part_A_final/train_data/images"
dataset_b="/zhome/d4/a/214319/adlcv_project/data/part_B_final/train_data/images"
output_dir="/zhome/d4/a/214319/adlcv_project/data/merged/train_data/images"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Copy files from dataset A as they are
cp "$dataset_a"/IMG_*.jpg "$output_dir"

# Get the highest index from dataset A
max_idx=$(ls "$dataset_a" | grep -oP 'IMG_\K[0-9]+' | sort -n | tail -1)
max_idx=${max_idx:-0}  # Default to 0 if no files exist

# Copy and rename files from dataset B
counter=$((max_idx + 1))
for file in "$dataset_b"/IMG_*.jpg; do
    cp "$file" "$output_dir/IMG_$counter.jpg"
    ((counter++))
done

echo "Merging complete!"
