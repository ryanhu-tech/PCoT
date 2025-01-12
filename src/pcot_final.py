import pandas as pd
import argparse
import os
import logging
from datetime import datetime
from utils.utils import (
    parallel_pcot, load_prompts_pcot_one_multistep, load_high_level_persuasion_groups,
    process_csv_files, remove_csv_suffix, process_ensemble_with_named_columns
)

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process a file path with test dataset for the script.")
    parser.add_argument(
        '-dataset_file', type=str, required=True, help="Path to the test dataset file (CSV format).")
    parser.add_argument(
        '-model', type=str, required=True, help="Model name to use for processing.")
    parser.add_argument(
        '-output_file_path', type=str, required=True, help="Output file path to save the results.")
    parser.add_argument(
        '-prompts_file_path', type=str, required=True, help="Output file path to save the results.")
    parser.add_argument(
        '-method_type', type=str, required=True, help="Provide prompting method which you want to use")
    parser.add_argument(
        '-prompt_type', type=str, required=True, help="Output file path to save the results.")
    args = parser.parse_args()

    # Retrieve arguments
    dataset_file = args.dataset_file
    model = args.model
    output_file_path = args.output_file_path
    prompts_file_path = args.prompts_file_path
    method_type = args.method_type
    prompt_type = args.prompt_type


    # Configure logging
    if model == "meta-llama/Llama-3.3-70B-Instruct-Turbo":
        log_filename = f"Llama-3_3_70B_instruct_turbo/logging/{method_type}/logs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        log_dir = os.path.dirname(log_filename)
        os.makedirs(log_dir, exist_ok=True)
    elif model == "gpt-4o-mini":
        log_filename = f"gpt4o-mini/logging/{method_type}/logs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        log_dir = os.path.dirname(log_filename)
        os.makedirs(log_dir, exist_ok=True)
    else:
        raise ValueError(f"""
        Model '{model}' not tested for this experiment. Please choose between belows models: 
        - "gpt-4o-mini" 
        - "meta-llama/Llama-3.3-70B-Instruct-Turbo" 
        or modify scripts to run it on other models""")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename)
        ]
    )

    logging.info(f"Script started with the following parameters:")
    logging.info(f"Test file path: {dataset_file}")
    logging.info(f"Model: {model}")
    logging.info(f"Output file path: {output_file_path}")

    # Call the function with parallelization
    if method_type == "pcot_one_multistep" or method_type == "pcot_one_detailed_multistep" or method_type == "pcot_one_multistep_human_llm_knowledge_infusion" or method_type == "pcot_one_multistep_detailed_human_llm_knowledge_infusion":
        # Load prompts from the YAML file based on the task name
        system_prompt, user_prompt_1, user_prompt_2 = load_prompts_pcot_one_multistep(
            prompts_file_path=prompts_file_path,
            method_type=method_type,
            prompt_type=prompt_type
        )
        # Read the CSV file
        try:
            df = pd.read_csv(dataset_file)
            logging.info(f"Successfully read the file: {dataset_file}")
            logging.info(f"DataFrame shape: {df.shape}")
        except FileNotFoundError:
            logging.error(f"File not found: {dataset_file}")
            exit(1)  # Exit the script if the file is not found
        except Exception as e:
            logging.error("An error occurred while reading the CSV file.", exc_info=True)
            exit(1)  # Exit the script if there's any other error

        try:
            logging.info("Starting text processing with parallelization.")
            parallel_pcot(
                dataframe=df.copy(),
                method_type=method_type,
                col_with_content="content",
                generated_persuasion_analysis="generated_pred",
                column="final_pred",
                filename=output_file_path,
                model=model,
                system_prompt=system_prompt,
                user_part_1=user_prompt_1,
                user_part_2=user_prompt_2
            )
            logging.info(f"Processing completed successfully. Results saved to: {output_file_path}")
        except Exception as e:
            logging.error("An error occurred during text processing.", exc_info=True)
    elif method_type == "pcot_one_task_at_a_time" or method_type == "pcot_one_detailed_task_at_a_time":
        # Load prompts from the YAML file based on the task name
        system_prompt, user_prompt_1, user_prompt_2 = load_prompts_pcot_one_multistep(
            prompts_file_path=prompts_file_path,
            method_type=method_type,
            prompt_type=prompt_type
        )

        final_df = process_csv_files(dataset_file, config_file_path="persuasion_groups.yaml")

        try:
            logging.info("Starting text processing with parallelization.")
            parallel_pcot(
                dataframe=final_df.copy(),
                method_type=method_type,
                col_with_content="content",
                column="final_pred",
                filename=output_file_path,
                model=model,
                system_prompt=system_prompt,
                user_part_1=user_prompt_1,
                user_part_2=user_prompt_2
            )
            logging.info(f"Processing completed successfully. Results saved to: {output_file_path}")
        except Exception as e:
            logging.error("An error occurred during text processing.", exc_info=True)
    elif method_type == "pcot_ensemble_one_task_at_a_time" or method_type == "pcot_ensemble_one_detailed_task_at_a_time" or method_type == "pcot_ensemble_extracted_one_task_at_a_time" or method_type == "pcot_ensemble_extracted_one_detailed_task_at_a_time":
        high_level_persuasion_groups = load_high_level_persuasion_groups(config_file_path="persuasion_groups.yaml")
        if method_type == "pcot_ensemble_one_task_at_a_time" or method_type == "pcot_ensemble_one_detailed_task_at_a_time":
            final_df = process_csv_files(dataset_file, config_file_path="persuasion_groups.yaml")
        else:
            final_df = pd.read_csv(dataset_file)

        for persuasion_group in high_level_persuasion_groups:
            logging.info("Loading prompts using `load_prompts_persuasion_knowledge_infusion`.")
            # Load prompts from the YAML file based on the task name
            system_prompt, user_prompt_1, user_prompt_2 = load_prompts_pcot_one_multistep(
                prompts_file_path=prompts_file_path,
                method_type=method_type,
                prompt_type=prompt_type,
                persuasion_group=persuasion_group
            )
            try:
                logging.info("Starting text processing with parallelization.")
                column = "pred_with_" + persuasion_group
                parallel_pcot(
                    dataframe=final_df.copy(),
                    method_type=method_type,
                    col_with_content="content",
                    generated_persuasion_analysis=persuasion_group,
                    column=column,
                    filename=output_file_path.split(".")[0] + "_" + persuasion_group + ".csv",
                    model=model,
                    system_prompt=system_prompt,
                    user_part_1=user_prompt_1,
                    user_part_2=user_prompt_2
                )
                logging.info(f"Processing completed successfully. Results saved to: {output_file_path}")
            except Exception as e:
                logging.error("An error occurred during text processing.", exc_info=True)
        dir_path = remove_csv_suffix(output_file_path)
        if "persuasions/" in dir_path:
            dir_path = dir_path.replace("persuasions/", "")
        ensemble_df = process_csv_files(dir_path, config_file_path="persuasion_groups.yaml", ensemble=True)
        column_names = ["pred_with_" + persuasion_group for persuasion_group in high_level_persuasion_groups]
        process_ensemble_with_named_columns(
            df=ensemble_df.copy(),
            column_names=column_names,
            results_path=dir_path+"/final",
            output_filename="final.csv"
        )