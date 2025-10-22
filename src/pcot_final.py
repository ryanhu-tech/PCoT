import argparse
import os
import logging 
from datetime import datetime
from utils.utils import (
    sequential_pcot, load_prompts_pcot_one_multistep, setup_logging, read_csv_file
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
    parser.add_argument('-use_vllm', type=lambda x: (str(x).lower() == 'true'), default=False, help="Use vLLM for local model inference.")
    return parser.parse_args()


def configure_logging(model, method_type, dataset_file, output_file_path):
    """Configures logging with a structured log filename."""
    model_name_for_path = os.path.basename(model) # Extract model name from path
    log_dir_part = os.path.dirname(output_file_path).replace(f"results/{model_name_for_path}/", "")
    log_filename = f"logging/{model_name_for_path}/{log_dir_part}/logs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    setup_logging(log_filename=log_filename, dataset_file=dataset_file, model=model, output_file_path=output_file_path)


def process_data(df, model, method_type, output_file_path, system_prompt, user_prompt_1, user_prompt_2, use_vllm):
    """Processes data based on the model type."""
    try:
        logging.info("Starting text processing.")
        sequential_pcot(
            dataframe=df.copy(),
            method_type=method_type,
            col_with_content="content",
            generated_persuasion_analysis="generated_pred",
            column="final_pred",
            filename=output_file_path,
            model=model,
            system_prompt=system_prompt,
            user_part_1=user_prompt_1,
            user_part_2=user_prompt_2,
            # The use_vllm parameter is passed but not used in the sequential_pcot function.
            # It can be removed if vLLM is not part of the sequential local model setup.
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

    process_data(df, args.model, args.method_type, args.output_file_path, system_prompt, user_prompt_1, user_prompt_2, args.use_vllm)


if __name__ == '__main__':
    main()
