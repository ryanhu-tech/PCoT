#!/opt/homebrew/bin/bash

# Define models (comment out the ones you don't want to use)

models=(
    "gpt-4o-mini"
#    "meta-llama/Llama-3.3-70B-Instruct-Turbo"
)

prompts_file_path="prompts/pcot_final_step.yaml"
method_type="pcot_one_task_at_a_time"


define_datasets() {
    local model=$1
    if [[ "$model" == "gpt-4o-mini" ]]; then
        declare -a datasets=(
#            "gpt4o-mini/results/CoAID/PCoT_One_Task_At_a_Time/"
#            "gpt4o-mini/results/FakeNewsNet/PCoT_One_Task_At_a_Time/"
#            "gpt4o-mini/results/ISOTFakeNews/PCoT_One_Task_At_a_Time/"
#            "gpt4o-mini/results/F3_LLM/PCoT_One_Task_At_a_Time/"
#            "gpt4o-mini/results/F3_LLM_Minor/PCoT_One_Task_At_a_Time/"
#            "gpt4o-mini/results/F3_LLM_Major/PCoT_One_Task_At_a_Time/"
#            "gpt4o-mini/results/F3_LLM_Critical/PCoT_One_Task_At_a_Time/"
            "gpt4o-mini/results/Infotester4Education/PCoT_One_Task_At_a_Time/"
        )
    elif [[ "$model" == "meta-llama/Llama-3.3-70B-Instruct-Turbo" ]]; then
        declare -a datasets=(
#            "Llama-3_3_70B_instruct_turbo/results/CoAID/PCoT_One_Task_At_a_Time/"
#            "Llama-3_3_70B_instruct_turbo/results/FakeNewsNet/PCoT_One_Task_At_a_Time/"
#            "Llama-3_3_70B_instruct_turbo/results/ISOTFakeNews/PCoT_One_Task_At_a_Time/"
#            "Llama-3_3_70B_instruct_turbo/results/F3_LLM/PCoT_One_Task_At_a_Time/"
#            "Llama-3_3_70B_instruct_turbo/results/F3_LLM_Minor/PCoT_One_Task_At_a_Time/"
#            "Llama-3_3_70B_instruct_turbo/results/F3_LLM_Major/PCoT_One_Task_At_a_Time/"
#            "Llama-3_3_70B_instruct_turbo/results/F3_LLM_Critical/PCoT_One_Task_At_a_Time/"
#            "Llama-3_3_70B_instruct_turbo/results/Infotester4Education/PCoT_One_Task_At_a_Time/"
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

    # Determine output file path based on the model
    local parent_dir
    parent_dir=$(dirname "$dataset_file")
    local output_file

    if [[ "$model" == "meta-llama/Llama-3.3-70B-Instruct-Turbo" ]]; then
        output_file="$parent_dir/PCoT_One_Task_At_a_Time/$prompt_type/final.csv"
    else
        output_file="$parent_dir/PCoT_One_Task_At_a_Time/$prompt_type/extracted/final.csv"
    fi

    echo "Processing: $dataset_file with prompt type $prompt_type on model $model..."
    uv run src/pcot_final.py \
        -dataset_file "$dataset_file" \
        -model "$model" \
        -output "$output_file" \
        -prompts_file_path "$prompts_file_path" \
        -prompt_type "$prompt_type" \
        -method_type "$method_type"
}

# Main loop to execute tasks
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
