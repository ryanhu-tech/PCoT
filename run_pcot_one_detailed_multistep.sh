#!/opt/homebrew/bin/bash

# Define models (comment out the ones you don't want to use)
models=(
#    "gpt-4o-mini"
#    "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    "gemini-1.5-flash"
#    "meta-llama/Meta-Llama-3.1-8B-Instruct"
#    "claude-3-haiku-20240307"
)

prompts_file_path="prompts/pcot_final_step.yaml"
method_type="pcot_one_detailed_multistep"


define_datasets() {
    local model=$1
    if [[ "$model" == "gpt-4o-mini" ]]; then
        declare -a datasets=(
            "gpt-4o-mini/results/CoAID/PCoT_One_Detailed_MultiStep/persuasion_and_explanation.csv"
#            "gpt-4o-mini/results/ISOTFakeNews/PCoT_One_Detailed_MultiStep/persuasion_and_explanation.csv"
#            "gpt-4o-mini/results/MultiDis/PCoT_One_Detailed_MultiStep/persuasion_and_explanation.csv"
#             "gpt-4o-mini/results/EUDisinfo/PCoT_One_Detailed_MultiStep/persuasion_and_explanation.csv"
#             "gpt-4o-mini/results/ECTF/PCoT_One_Detailed_MultiStep/persuasion_and_explanation.csv"
        )
    elif [[ "$model" == "meta-llama/Llama-3.3-70B-Instruct-Turbo" ]]; then
        declare -a datasets=(
            "meta-llama/Llama-3.3-70B-Instruct-Turbo/results/CoAID/PCoT_One_Detailed_MultiStep/persuasion_and_explanation.csv"
#            "meta-llama/Llama-3.3-70B-Instruct-Turbo/results/ISOTFakeNews/PCoT_One_Detailed_MultiStep/persuasion_and_explanation.csv"
#            "meta-llama/Llama-3.3-70B-Instruct-Turbo/results/MultiDis/PCoT_One_Detailed_MultiStep/persuasion_and_explanation.csv"
#            "meta-llama/Llama-3.3-70B-Instruct-Turbo/results/EUDisinfo/PCoT_One_Detailed_MultiStep/persuasion_and_explanation.csv"
#            "meta-llama/Llama-3.3-70B-Instruct-Turbo/results/ECTF/PCoT_One_Detailed_MultiStep/persuasion_and_explanation.csv"
        )
    elif [[ "$model" == "meta-llama/Meta-Llama-3.1-8B-Instruct" ]]; then
        declare -a datasets=(
            "meta-llama/Meta-Llama-3.1-8B-Instruct/results/CoAID/PCoT_One_Detailed_MultiStep/persuasion_and_explanation.csv"
#            "meta-llama/Meta-Llama-3.1-8B-Instruct/results/ISOTFakeNews/PCoT_One_Detailed_MultiStep/persuasion_and_explanation.csv"
#            "meta-llama/Meta-Llama-3.1-8B-Instruct/results/MultiDis/PCoT_One_Detailed_MultiStep/persuasion_and_explanation.csv"
#            "meta-llama/Meta-Llama-3.1-8B-Instruct/results/EUDisinfo/PCoT_One_Detailed_MultiStep/persuasion_and_explanation.csv"
#            "meta-llama/Meta-Llama-3.1-8B-Instruct/results/ECTF/PCoT_One_Detailed_MultiStep/persuasion_and_explanation.csv"
        )
    elif [[ "$model" == "gemini-1.5-flash" ]]; then
        declare -a datasets=(
            "gemini-1.5-flash/results/CoAID/PCoT_One_Detailed_MultiStep/persuasion_and_explanation.csv"
#            "gemini-1.5-flash/results/ISOTFakeNews/PCoT_One_Detailed_MultiStep/persuasion_and_explanation.csv"
#            "gemini-1.5-flash/results/MultiDis/PCoT_One_Detailed_MultiStep/persuasion_and_explanation.csv"
#            "gemini-1.5-flash/results/ECTF/PCoT_One_Detailed_MultiStep/persuasion_and_explanation.csv"
#            "gemini-1.5-flash/results/EUDisinfo/PCoT_One_Detailed_MultiStep/persuasion_and_explanation.csv"
        )
    elif [[ "$model" == "claude-3-haiku-20240307" ]]; then
        declare -a datasets=(
            "claude-3-haiku-20240307/results/CoAID/PCoT_One_Detailed_MultiStep/persuasion_and_explanation.csv"
#            "claude-3-haiku-20240307/results/ISOTFakeNews/PCoT_One_Detailed_MultiStep/persuasion_and_explanation.csv"
#            "claude-3-haiku-20240307/results/MultiDis/PCoT_One_Detailed_MultiStep/persuasion_and_explanation.csv"
#            "claude-3-haiku-20240307/results/ECTF/PCoT_One_Detailed_MultiStep/persuasion_and_explanation.csv"
#            "claude-3-haiku-20240307/results/EUDisinfo/PCoT_One_Detailed_MultiStep/persuasion_and_explanation.csv"
        )
    fi
    for dataset in "${datasets[@]}"; do
        echo "$dataset"
    done
}

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
    # Dynamically define datasets based on the model using mapfile
    datasets=()
    mapfile -t datasets < <(define_datasets "$model")

    for prompt_type in "${prompt_types[@]}"; do
        for dataset_file in "${datasets[@]}"; do
            run_script "$dataset_file" "$prompt_type" "$model"
        done
    done
done

echo "All tasks completed."