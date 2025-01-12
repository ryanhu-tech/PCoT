#!/opt/homebrew/bin/bash

# This is tested and it works
# Define common variables
models=(
    "gpt-4o-mini"
)
prompts_file_path="prompts/pcot_final_step.yaml"
method_type="pcot_ensemble_one_detailed_task_at_a_time"


define_datasets() {
    local model=$1
    if [[ "$model" == "gpt-4o-mini" ]]; then
        declare -a datasets=(
            "gpt4o-mini/results/CoAID/PCoT_One_Detailed_Task_At_a_Time/"
            "gpt4o-mini/results/FakeNewsNet/PCoT_One_Detailed_Task_At_a_Time/"
            "gpt4o-mini/results/ISOTFakeNews/PCoT_One_Detailed_Task_At_a_Time/"
            "gpt4o-mini/results/F3_LLM/PCoT_One_Detailed_Task_At_a_Time/"
            "gpt4o-mini/results/F3_LLM_Minor/PCoT_One_Detailed_Task_At_a_Time/"
            "gpt4o-mini/results/F3_LLM_Major/PCoT_One_Detailed_Task_At_a_Time/"
            "gpt4o-mini/results/F3_LLM_Critical/PCoT_One_Detailed_Task_At_a_Time/"
            "gpt4o-mini/results/Infotester4Education/PCoT_One_Detailed_Task_At_a_Time/"
        )
    elif [[ "$model" == "meta-llama/Llama-3.3-70B-Instruct-Turbo" ]]; then
        declare -a datasets=(

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

    if [[ "$model" == "meta-llama/Llama-3.3-70B-Instruct-Turbo" ]]; then
        output_file="$parent_dir/PCoT_Ensemble_One_Detailed_Task_At_a_Time/$prompt_type/final.csv"
    else
        output_file="$parent_dir/PCoT_Ensemble_One_Detailed_Task_At_a_Time/$prompt_type/final.csv"
    fi

    echo "Processing: $dataset_file with prompt type $prompt_type..."
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
