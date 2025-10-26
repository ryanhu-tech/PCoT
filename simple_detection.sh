#!/opt/homebrew/bin/bash

# --- 指定要使用的 GPU ID (例如 0, 1, 2, 3) ---
GPU_ID=${1:-3}



# --- 從命令列參數設定模型和資料集，若未提供則使用預設值 ---
MODEL_TO_RUN=${2:-"/workspace/models/Llama-3.1-8B-Instruct"}
DATASET_TO_RUN=${3} # 從第三個參數讀取，但不設定預設值

models=("$MODEL_TO_RUN")

declare -a all_datasets=(
    "data/CoAID/test.csv"
    "data/ISOTFakeNews/test.csv"
    "data/MultiDis/test.csv"
    "data/EUDisinfo/test.csv"
    "data/ECTF/test.csv"
)

prompts_file_path="prompts/simple_detection.yaml"
method_type="simple_detection"


# Define prompt types
declare -a prompt_types=("VaN" "Z-CoT" "DeF_Spec")

# Function to run the script
run_script() {
    local dataset_file=$1
    local prompt_type=$2
    local model=$3

    # Generate output file path
    model_name_for_path=$(basename "$model")
    local parent_dir
    parent_dir=$(basename "$(dirname "$dataset_file")")
    local output_file
    output_file="results/$model_name_for_path/$parent_dir/Simple_Detection/$prompt_type/simple_detection.csv"

    echo "Processing: $dataset_file with prompt type $prompt_type on model $model..."
    CUDA_VISIBLE_DEVICES=$GPU_ID uv run src/simple_detection_and_persuasion_step.py \
        -dataset_file "$dataset_file" \
        -model "$model" \
        -output_file_path "$output_file" \
        -prompts_file_path "$prompts_file_path" \
        -prompt_type "$prompt_type" \
        -method_type "$method_type"
}

# 決定要執行的資料集
if [ -n "$DATASET_TO_RUN" ]; then
    datasets=("$DATASET_TO_RUN")
else
    datasets=("${all_datasets[@]}")
fi

# Main loop to execute tasks
for model in "${models[@]}"; do
    for prompt_type in "${prompt_types[@]}"; do
        for dataset_file in "${datasets[@]}"; do
            run_script "$dataset_file" "$prompt_type" "$model"
        done
    done
done

echo "All tasks completed."
