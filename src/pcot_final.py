import argparse
import os
import logging
from datetime import datetime
from utils.utils import (
    parallel_pcot, load_prompts_pcot_one_multistep, sequential_pcot_claude, setup_logging, read_csv_file
)


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Process a file path with test dataset for the script.")
    parser.add_argument('-dataset_file', type=str, required=True, help="Path to the test dataset file (CSV format).")
    parser.add_argument('-model', type=str, required=True, help="Model name to use for processing.")
    parser.add_argument('-output_file_path', type=str, required=True, help="Path to save the results.")
    parser.add_argument('-prompts_file_path', type=str, required=True, help="Path to the prompts YAML file.")
    parser.add_argument('-method_type', type=str, required=True, help="Prompting method to use.")
    parser.add_argument('-prompt_type', type=str, required=True, help="Prompt type.")
    return parser.parse_args()


def configure_logging(model, method_type, dataset_file, output_file_path):
    """Configures logging with a structured log filename."""
    log_dir = os.path.join(model, "logging", method_type)
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"logs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    setup_logging(log_filename=log_filename, dataset_file=dataset_file, model=model, output_file_path=output_file_path)


def process_data(df, model, method_type, output_file_path, system_prompt, user_prompt_1, user_prompt_2):
    """Processes data based on the model type."""
    try:
        logging.info("Starting text processing.")
        if model != "claude-3-haiku-20240307":
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
        else:
            sequential_pcot_claude(
                dataframe=df.copy(),
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


def main():
    args = parse_arguments()
    configure_logging(args.model, args.method_type, args.dataset_file, args.output_file_path)

    logging.info("Loading prompts...")
    system_prompt, user_prompt_1, user_prompt_2 = load_prompts_pcot_one_multistep(
        prompts_file_path=args.prompts_file_path,
        method_type=args.method_type,
        prompt_type=args.prompt_type
    )

    logging.info("Reading dataset...")
    df = read_csv_file(args.dataset_file)

    process_data(df, args.model, args.method_type, args.output_file_path, system_prompt, user_prompt_1, user_prompt_2)


if __name__ == '__main__':
    main()
