#!/opt/homebrew/bin/bash

# Define models (comment out the ones you don't want to use)
models=(
#    "gpt-4o-mini"
#    "meta-llama/Llama-3.3-70B-Instruct-Turbo"
#    "gemini-1.5-flash"
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
#    "claude-3-haiku-20240307"
)

prompts_file_path="prompts/pcot_final_step.yaml"
method_type="pcot_one_detailed_multistep"

# Define dataset names (comment out the ones you don't want to use)
dataset_names=(
    "CoAID"
#    "ISOTFakeNews"
#    "MultiDis"
#    "EUDisinfo"
#    "ECTF"
)

declare -a prompt_types=("VaN" "Z-CoT" "DeF_Spec")

# Function to run the script
run_script() {
    local dataset_file=$1
    local prompt_type=$2
    local model=$3

    # Generate output file path
    local parent_dir
    parent_dir=$(dirname "$dataset_file")

    local output_file
    output_file="$parent_dir/$prompt_type/final.csv"


    echo "Processing: $dataset_file with prompt type $prompt_type on model $model..."
    uv run src/pcot_final.py \
        -dataset_file "$dataset_file" \
        -model "$model" \
        -output "$output_file" \
        -prompts_file_path "$prompts_file_path" \
        -prompt_type "$prompt_type" \
        -method_type "$method_type"
}

# Loop through prompt types and datasets
for model in "${models[@]}"; do
    for dataset_name in "${dataset_names[@]}"; do
        dataset_file="$model/results/$dataset_name/PCoT_One_Detailed_MultiStep/persuasion_and_explanation.csv"
        for prompt_type in "${prompt_types[@]}"; do
            run_script "$dataset_file" "$prompt_type" "$model"
        done
    done
done
echo "All tasks completed."