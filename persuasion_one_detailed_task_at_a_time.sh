#!/bin/bash

# Define models (comment out the ones you don't want to use)
models=(
#    "gpt-4o-mini"
    "meta-llama/Llama-3.3-70B-Instruct-Turbo"
)

prompts_file_path="prompts/persuasion_knowledge_infusion.yaml"
method_type="pcot_one_detailed_task_at_a_time"

# Define datasets
declare -a datasets=(
#    "../data/CoAID/test.csv"
#    "../data/FakeNewsNet/test.csv"
#    "../data/ISOTFakeNews/test.csv"
#    "../data/F3_LLM/test.csv"
#    "../data/F3_LLM_Minor/test.csv"
#    "../data/F3_LLM_Major/test.csv"
#    "../data/F3_LLM_Critical/test.csv"
     "data/Infotester4Education/test.csv"
)

# Function to run the script
run_script() {
    local dataset_file=$1
    local model=$2

    # Generate output file path
    local parent_dir
    parent_dir=$(basename "$(dirname "$dataset_file")")

    local output_file

    if [[ "$model" == "meta-llama/Llama-3.3-70B-Instruct-Turbo" ]]; then
        output_file="Llama-3_3_70B_instruct_turbo/results/$parent_dir/PCoT_One_Detailed_Task_At_a_Time/persuasion_and_explanation_detailed.csv"
    else
        output_file="gpt4o-mini/results/$parent_dir/PCoT_One_Detailed_Task_At_a_Time/persuasion_and_explanation_detailed.csv"
    fi

    # Run the Python script
    echo "Processing: $dataset_file with prompt to generate persuasion analysis on model $model..."
    uv run src/simple_detection_and_persuasion_step.py \
        -dataset_file "$dataset_file" \
        -model "$model" \
        -output "$output_file" \
        -prompts_file_path "$prompts_file_path" \
        -method_type "$method_type"
}

# Loop through datasets
for model in "${models[@]}"; do
  for dataset_file in "${datasets[@]}"; do
    run_script "$dataset_file" "$model"
  done
done

echo "All tasks completed."
