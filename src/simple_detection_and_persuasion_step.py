import argparse
import logging
from datetime import datetime
from utils.utils import (
    parallel_text_processing, load_prompts_simple_detection,
    load_prompts_persuasion_knowledge_infusion, sequential_text_processing_claude, setup_logging, read_csv_file
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
    log_dir_part = dataset_file.replace("test.csv", "")
    log_filename = (
        f"{model}/logging/{method_type}/{log_dir_part}/"
        f"logs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )

    setup_logging(
        log_filename=log_filename,
        dataset_file=dataset_file,
        model=model,
        output_file_path=output_file_path
    )

    # Read the CSV file
    df = read_csv_file(dataset_file)

    # Load prompts based on the provided arguments
    try:
        if method_type == "simple_detection" or method_type == "simple_detection_with_persuasion":
            logging.info(f"Loading prompts using `load_prompts` with type: {prompt_type}")
            system_prompt, user_prompt = load_prompts_simple_detection(
                prompts_file_path=prompts_file_path,
                prompt_type=prompt_type
            )

            if model != "claude-3-haiku-20240307":
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
            elif model in ["claude-3-haiku-20240307"]:
                # Call the function with parallelization
                try:
                    logging.info("Starting text processing with parallelization.")
                    sequential_text_processing_claude(
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
        elif method_type == "pcot_one_multistep" or method_type == "pcot_one_detailed_multistep" or method_type == "pcot_one_detailed_multistep_no_exp_final":

            logging.info("Loading prompts using `load_prompts_persuasion_one_multistep`.")
            system_prompt, user_prompt = load_prompts_persuasion_knowledge_infusion(
                prompts_file_path=prompts_file_path,
                method_type=method_type
            )
            if model != "claude-3-haiku-20240307":
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

            else:

                # Call the function with parallelization
                try:
                    logging.info("Starting text processing with parallelization.")
                    sequential_text_processing_claude(
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
    except Exception as e:
        logging.error("An error occurred while loading prompts.", exc_info=True)
        exit(1)


if __name__ == '__main__':
    main()
