#!/opt/homebrew/bin/bash

# Define models (comment out the ones you don't want to use)
models=(
    "gpt-4o-mini"
    "meta-llama/Llama-3.3-70B-Instruct-Turbo"
)

prompts_file_path="prompts/simple_detection.yaml"
method_type="simple_detection"

# Define common datasets
declare -a datasets=(
    "data/CoAID/test.csv"
    "data/FakeNewsNet/test.csv"
    "data/ISOTFakeNews/test.csv"
    "data/F3_LLM/test.csv"
    "data/F3_LLM_Minor/test.csv"
    "data/F3_LLM_Major/test.csv"
    "data/F3_LLM_Critical/test.csv"
     "data/Infotester4Education/test.csv"
)

# Define prompt types
declare -a prompt_types=("VaN" "Z-CoT" "DeF_Spec")


# Function to run the script
run_script() {
    local dataset_file=$1
    local prompt_type=$2
    local model=$3

    # Generate output file path
    local parent_dir
    parent_dir=$(basename "$(dirname "$dataset_file")")
    local output_file

    if [[ "$model" == "meta-llama/Llama-3.3-70B-Instruct-Turbo" ]]; then
        output_file="Llama-3_3_70B_instruct_turbo/results/$parent_dir/Simple_Detection/$prompt_type/simple_detection.csv"
    else
        output_file="gpt4o-mini/results/$parent_dir/Simple_Detection/$prompt_type/simple_detection.csv"
    fi

    echo "Processing: $dataset_file with prompt type $prompt_type on model $model..."
    uv run src/simple_detection_and_persuasion_step.py \
        -dataset_file "$dataset_file" \
        -model "$model" \
        -output "$output_file" \
        -prompts_file_path "$prompts_file_path" \
        -prompt_type "$prompt_type" \
        -method_type "$method_type"
}  # <-- Ensure this closing brace is here.

# Main loop to execute tasks
for model in "${models[@]}"; do
    for prompt_type in "${prompt_types[@]}"; do
        for dataset_file in "${datasets[@]}"; do
            run_script "$dataset_file" "$prompt_type" "$model"
        done
    done
done

echo "All tasks completed."
