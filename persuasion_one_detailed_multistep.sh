#!/opt/homebrew/bin/bash

# --- 指定要使用的 GPU ID (例如 0, 1, 2, 3) ---
GPU_ID=3

# Define models (comment out the ones you don't want to use)
models=(
    "/workspace/models/Llama-3.1-8B-Instruct"
)

prompts_file_path="prompts/persuasion_knowledge_infusion.yaml"
method_type="pcot_one_detailed_multistep"

# Define datasets
declare -a datasets=(
    "data/CoAID/test.csv"
#    "data/ISOTFakeNews/test.csv"
#    "data/MultiDis/test.csv"
#    "data/EUDisinfo/test.csv"
#    "data/ECTF/test.csv"
)

# Loop through datasets
for model in "${models[@]}"; do
  for dataset_file in "${datasets[@]}"; do
    # Generate output file path from dataset file
    model_name_for_path=$(basename "$model")
    parent_dir=$(basename "$(dirname "$dataset_file")")
    output_file="results/$model_name_for_path/$parent_dir/PCoT_One_Detailed_MultiStep/persuasion_and_explanation.csv"

    echo "Processing: $dataset_file with prompt to generate persuasion analysis on model $model..."

    # Run the Python script
    CUDA_VISIBLE_DEVICES=$GPU_ID uv run src/simple_detection_and_persuasion_step.py \
        -dataset_file "$dataset_file" \
        -model "$model" \
        -output_file_path "$output_file" \
        -prompts_file_path "$prompts_file_path" \
        -method_type "$method_type"
  done
done

echo "All tasks completed."
