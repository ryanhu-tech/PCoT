#!/opt/homebrew/bin/bash

# --- 使用說明 ---
# bash run_pcot_one_detailed_multistep.sh [GPU_ID] [MODEL_PATH] [DATASET_NAME]
#
# 範例:
# bash run_pcot_one_detailed_multistep.sh 0 "/workspace/models/Llama-3.1-8B-Instruct" "CoAID"
#
# 參數:
# GPU_ID:         要使用的 GPU ID (預設: 3)
# MODEL_PATH:     模型的完整路徑 (預設: "/workspace/models/Llama-3.1-8B-Instruct")
# DATASET_NAME:   資料集的名稱 (預設: "CoAID")
#                 可選: "CoAID", "ISOTFakeNews", "MultiDis", "EUDisinfo", "ECTF"
# -----------------

# --- 指定要使用的 GPU ID (例如 0, 1, 2, 3) ---
GPU_ID=${1:-3}

# --- 是否使用 vLLM (設為 true 來啟用) ---
USE_VLLM=true

# --- 從命令列參數設定模型和資料集，若未提供則使用預設值 ---
MODEL_TO_RUN=${2:-"/workspace/models/Llama-3.1-8B-Instruct"}
DATASET_NAME_TO_RUN=${3:-"CoAID"}

models=("$MODEL_TO_RUN")
dataset_names=("$DATASET_NAME_TO_RUN")

prompts_file_path="prompts/pcot_final_step.yaml"
method_type="pcot_one_detailed_multistep"

declare -a prompt_types=("VaN" "Z-CoT" "DeF_Spec")

# Function to run the script
run_script() {
    local dataset_file=$1
    local prompt_type=$2
    local model=$3

    # Generate output file path
    model_name_for_path=$(basename "$model")
    # Extract the dataset name (e.g., CoAID) from the input file path
    local dataset_name=$(echo "$dataset_file" | awk -F/ '{print $(NF-2)}')

    local output_file
    output_file="results/$model_name_for_path/$dataset_name/PCoT_One_Detailed_MultiStep/$prompt_type/final.csv"


    echo "Processing: $dataset_file with prompt type $prompt_type on model $model..."
    CUDA_VISIBLE_DEVICES=$GPU_ID uv run src/pcot_final.py \
        -dataset_file "$dataset_file" \
        -model "$model" \
        -output_file_path "$output_file" \
        -prompts_file_path "$prompts_file_path" \
        -prompt_type "$prompt_type" \
        -method_type "$method_type" \
        -use_vllm "$USE_VLLM"
}

# Loop through prompt types and datasets
for model in "${models[@]}"; do
    for dataset_name in "${dataset_names[@]}"; do
        model_name_for_path=$(basename "$model")
        dataset_file="results/$model_name_for_path/$dataset_name/PCoT_One_Detailed_MultiStep/persuasion_and_explanation.csv"
        for prompt_type in "${prompt_types[@]}"; do
            run_script "$dataset_file" "$prompt_type" "$model"
        done
    done
done
echo "All tasks completed."