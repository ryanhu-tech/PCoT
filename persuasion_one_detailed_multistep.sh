#!/opt/homebrew/bin/bash

# --- 使用說明 ---
# bash persuasion_one_detailed_multistep.sh [GPU_ID] [MODEL_PATH] [DATASET_PATH]
#
# 範例:
# bash persuasion_one_detailed_multistep.sh 0 "/workspace/models/Llama-3.1-8B-Instruct" "data/MultiDis/test.csv"
#
# 參數:
# GPU_ID:         要使用的 GPU ID (預設: 3)
# MODEL_PATH:     模型的完整路徑 (預設: "/workspace/models/Llama-3.1-8B-Instruct")
# DATASET_PATH:   資料集的完整路徑 (預設: "data/MultiDis/test.csv")
#                                     "data/CoAID/test.csv"
#                                     "data/ISOTFakeNews/test.csv"
#                                     "data/MultiDis/test.csv"
#                                     "data/EUDisinfo/test.csv"
#                                     "data/ECTF/test.csv"
# -----------------

# --- 指定要使用的 GPU ID (例如 0, 1, 2, 3) ---
GPU_ID=${1:-3}

# --- 從命令列參數設定模型和資料集，若未提供則使用預設值 ---
MODEL_TO_RUN=${2:-"/workspace/models/Llama-3.1-8B-Instruct"}
DATASET_TO_RUN=${3:-"data/MultiDis/test.csv"}

models=("$MODEL_TO_RUN")
datasets=("$DATASET_TO_RUN")

prompts_file_path="prompts/persuasion_knowledge_infusion.yaml"
method_type="pcot_one_detailed_multistep"

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
