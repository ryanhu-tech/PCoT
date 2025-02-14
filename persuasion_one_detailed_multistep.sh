#!/opt/homebrew/bin/bash

# Define models (comment out the ones you don't want to use)
models=(
    "gpt-4o-mini"
    "gemini-1.5-flash"
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    "claude-3-haiku-20240307"
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

# Function to run the script
run_script() {
    local dataset_file=$1
    local model=$2

    # Generate output file path
    local parent_dir
    parent_dir=$(basename "$(dirname "$dataset_file")")

    local output_file

    output_file="$model/results/$parent_dir/PCoT_One_Detailed_MultiStep/persuasion_and_explanation.csv"


    echo "Processing: $dataset_file with prompt to generate persuasion analysis on model $model..."
    # Run the Python script
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
