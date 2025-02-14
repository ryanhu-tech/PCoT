import argparse
import logging
from datetime import datetime
from utils.utils import (
    process_text, load_prompts_simple_detection,
    load_prompts_persuasion_knowledge_infusion, setup_logging, read_csv_file
)


def main():
    parser = argparse.ArgumentParser(description="Process a file path with test dataset for the script.")
    parser.add_argument('-dataset_file', type=str, required=True, help="Path to the test dataset file (CSV format).")
    parser.add_argument('-model', type=str, required=True, help="Model name to use for processing.")
    parser.add_argument('-output_file_path', type=str, required=True, help="Output file path to save the results.")
    parser.add_argument('-prompts_file_path', type=str, required=True, help="Path to the prompts file.")
    parser.add_argument('-method_type', type=str, required=True, help="Provide prompting method to use.")
    parser.add_argument('-prompt_type', type=str, required=False, default=None,
                        help="Prompt type for `load_prompts`. Defaults to `load_prompts_persuasion_one_multistep`.")
    args = parser.parse_args()

    dataset_file, model, output_file_path = args.dataset_file, args.model, args.output_file_path
    prompts_file_path, method_type, prompt_type = args.prompts_file_path, args.method_type, args.prompt_type

    log_dir_part = dataset_file.replace("test.csv", "")
    log_filename = f"{model}/logging/{method_type}/{log_dir_part}/logs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    setup_logging(log_filename, dataset_file, model, output_file_path)

    df = read_csv_file(dataset_file)
    try:
        if method_type in ["simple_detection", "simple_detection_with_persuasion"]:
            logging.info(f"Loading prompts using `load_prompts` with type: {prompt_type}")
            system_prompt, user_prompt = load_prompts_simple_detection(prompts_file_path, prompt_type)
        elif method_type == "pcot_one_detailed_multistep":
            logging.info("Loading prompts using `load_prompts_persuasion_one_multistep`.")
            system_prompt, user_prompt = load_prompts_persuasion_knowledge_infusion(prompts_file_path, method_type)
        else:
            logging.error("Invalid method_type provided.")
            exit(1)

        process_text(df, model, output_file_path, system_prompt, user_prompt)
    except Exception as e:
        logging.error("An error occurred while loading prompts.", exc_info=True)
        exit(1)


if __name__ == '__main__':
    main()
