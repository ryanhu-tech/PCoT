import pandas as pd
import argparse
import logging
from datetime import datetime
import os
from utils.utils import (
    parallel_text_processing, load_prompts_simple_detection, load_prompts_persuasion_knowledge_infusion,
    load_high_level_persuasion_groups
)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process a file path with test dataset for the script.")
    parser.add_argument(
        '-dataset_file', type=str, required=True, help="Path to the test dataset file (CSV format).")
    parser.add_argument(
        '-model', type=str, required=True, help="Model name to use for processing.")
    parser.add_argument(
        '-output_file_path', type=str, required=True, help="Output file path to save the results.")
    parser.add_argument(
        '-prompts_file_path', type=str, required=True, help="Path to the prompts file.")
    parser.add_argument(
        '-method_type', type=str, required=True, help="Provide prompting method which you want to use")
    parser.add_argument(
        '-prompt_type', type=str, required=False, default=None,
        help="Prompt type for `load_prompts`. If not provided, `load_prompts_persuasion_one_multistep` will be used.")

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
        log_filename = (
            f"Llama-3_3_70B_instruct_turbo/logging/simple_detection_and_persuasion_step/{dataset_file}/{method_type}/"
            f"logs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        )
        log_dir = os.path.dirname(log_filename)
        os.makedirs(log_dir, exist_ok=True)
    elif model == "gpt-4o-mini":
        log_filename = (
                f"gpt4o-mini/logging/simple_detection_and_persuasion_step/"
                f"logs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
            )
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

    # Load prompts based on the provided arguments
    try:
        if method_type == "simple_detection":
            logging.info(f"Loading prompts using `load_prompts` with type: {prompt_type}")
            system_prompt, user_prompt = load_prompts_simple_detection(
                prompts_file_path=prompts_file_path,
                prompt_type=prompt_type
            )
            # Call the function with parallelization
            try:
                logging.info("Starting text processing with parallelization.")
                parallel_text_processing(
                    dataframe=df.copy(),
                    col_with_content="content",
                    column="generated_pred",
                    filename=output_file_path,
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                )
                logging.info(f"Processing completed successfully. Results saved to: {output_file_path}")
            except Exception as e:
                logging.error("An error occurred during text processing.", exc_info=True)
        elif method_type == "pcot_one_multistep" or method_type == "pcot_one_detailed_multistep":
            logging.info("Loading prompts using `load_prompts_persuasion_one_multistep`.")
            system_prompt, user_prompt = load_prompts_persuasion_knowledge_infusion(
                prompts_file_path=prompts_file_path,
                method_type=method_type
            )
            # Call the function with parallelization
            try:
                logging.info("Starting text processing with parallelization.")
                parallel_text_processing(
                    dataframe=df.copy(),
                    col_with_content="content",
                    column="generated_pred",
                    filename=output_file_path,
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                )
                logging.info(f"Processing completed successfully. Results saved to: {output_file_path}")
            except Exception as e:
                logging.error("An error occurred during text processing.", exc_info=True)
        elif method_type == "pcot_one_task_at_a_time" or method_type == "pcot_one_detailed_task_at_a_time":
            high_level_persuasion_groups = load_high_level_persuasion_groups(config_file_path="persuasion_groups.yaml")

            for persuasion_group in high_level_persuasion_groups:
                logging.info("Loading prompts using `load_prompts_persuasion_knowledge_infusion`.")
                system_prompt, user_prompt = load_prompts_persuasion_knowledge_infusion(
                    prompts_file_path=prompts_file_path,
                    method_type=method_type,
                    persuasion_group=persuasion_group
                )
                try:
                    logging.info("Starting text processing with parallelization.")
                    parallel_text_processing(
                        dataframe=df.copy(),
                        col_with_content="content",
                        column=persuasion_group,
                        filename=output_file_path.rsplit('.', 1)[0] + "_" + persuasion_group + ".csv",
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt
                    )
                    logging.info(f"Processing completed successfully. Results saved to: {output_file_path}")
                except Exception as e:
                    logging.error("An error occurred during text processing.", exc_info=True)
    except Exception as e:
        logging.error("An error occurred while loading prompts.", exc_info=True)
        exit(1)


if __name__ == '__main__':
    main()
