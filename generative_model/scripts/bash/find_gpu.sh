#!/bin/bash

# Script to find the GPU with the least memory usage
# Adapted from: https://github.com/YanjieZe/3D-Diffusion-Policy

# User-configurable: List of GPU IDs to exclude (modify or set via env variable)
exclude_gpus=(${EXCLUDE_GPUS[@]})  # Supports environment variable EXCLUDE_GPUS

# Function to check if a value exists in an array (more efficient version)
containsElement() {
  local element="$1"
  shift
  [[ " ${@} " =~ " ${element} " ]] && return 0 || return 1
}

# Fetch GPU usage (index and memory used)
gpu_usage=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null)

# Validate if nvidia-smi executed successfully
if [[ -z "$gpu_usage" ]]; then
  echo "Error: Unable to fetch GPU usage. Ensure NVIDIA drivers and nvidia-smi are installed."
  exit 1
fi

# Initialize tracking variables
min_usage=999999
min_gpu=-1

# Parse GPU usage data
while IFS=, read -r gpu_id usage; do
    # Check if this GPU is in the exclusion list
    if containsElement "$gpu_id" "${exclude_gpus[@]}"; then
        continue
    fi

    # Update if this GPU has the least memory usage
    if (( usage < min_usage )); then
        min_usage=$usage
        min_gpu=$gpu_id
    fi
done <<< "$gpu_usage"

# Output the selected GPU index or an error message if no GPU is available
if [[ "$min_gpu" -eq -1 ]]; then
  echo "Error: No available GPU found."
  exit 1
else
  echo "$min_gpu"
fi