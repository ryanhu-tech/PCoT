"""Helper functions"""
import os
import time
import yaml
import logging
import anthropic
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import BitsAndBytesConfig
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()
TEMPERATURE = 0.0
#
# Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")


def read_csv_file(dataset_file):
    """Reads a CSV file and logs its status."""
    try:
        df = pd.read_csv(dataset_file)
        logging.info(f"Successfully read the file: {dataset_file}")
        logging.info(f"DataFrame shape: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {dataset_file}")
        exit(1)  # Exit the script if the file is not found
    except Exception as e:
        logging.error("An error occurred while reading the CSV file.", exc_info=True)
        exit(1)  # Exit the script if there's any other error


def setup_logging(log_filename, dataset_file, model, output_file_path):
    """Sets up logging configuration and logs initial script parameters."""
    log_dir = os.path.dirname(log_filename)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename)
        ]
    )

    logging.info("Script started with the following parameters:")
    logging.info(f"Test file path: {dataset_file}")
    logging.info(f"Model: {model}")
    logging.info(f"Output file path: {output_file_path}")


def client_instance(model):
    if model in ["gpt-4o-mini"]:
        client = OpenAI(api_key=OPENAI_API_KEY)
        return client
    # Check if the model is a local directory path first
    elif os.path.isdir(model):
        return "local_llama"
    elif model in ["meta-llama/Llama-3.3-70B-Instruct-Turbo"]:
        client = OpenAI(api_key=DEEPINFRA_API_KEY, base_url="https://api.deepinfra.com/v1/openai")
        return client    
    elif model == "claude-3-haiku-20240307":
        client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
        return client


class LocalModelManager:
    """
    A singleton class to manage the loading of a local model and tokenizer.
    This ensures the model is loaded only once and shared across different calls.
    """
    _instance = None
    _pipe = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LocalModelManager, cls).__new__(cls)
        return cls._instance

    def load_model(self, model_path):
        """
        Loads the model and tokenizer and creates a pipeline.
        If the pipeline is already loaded, it does nothing.
        """
        if self._pipe is None:
            logging.info(f"Loading local model from {model_path} for the first time...")
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model_obj = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto"
            )
            self._pipe = pipeline("text-generation", model=model_obj, tokenizer=tokenizer)
            logging.info("Local model loaded successfully.")
        return self._pipe


def compute_metrics(y_true, y_pred):
    # clf_report = classification_report(y_true, y_pred, output_dict=True)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1_macro_weighted = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    # roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    # accuracy = accuracy_score(y_true, y_pred)

    metrics = {
        'f1': round(f1, 3),
        'f1_micro': round(f1_micro_average, 3),
        'f1_macro': round(f1_macro_average, 3),
        'f1_macro_weighted': round(f1_macro_weighted, 3)
    }

    return metrics


def process_text_with_model(index, text, model, system_prompt, user_prompt):
    """
    Processes a single text using the model and returns the result.
    """
    try:
        user_prompt = user_prompt + f" Text:{text}. Answer:"
        if model == "gemini-1.5-flash":
            genai.configure(api_key=GEMINI_API_KEY)
            if system_prompt:
                model = genai.GenerativeModel(
                    model_name="gemini-1.5-flash",
                    system_instruction=system_prompt)
            else:
                model = genai.GenerativeModel(
                    model_name="gemini-1.5-flash")
            response = model.generate_content(
                user_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=TEMPERATURE,
                )
            )
            result = {
                "index": index,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "completion": response.text,
            }
            return result

        else:
            client = client_instance(model=model)
            if client == "local_llama":
                # Get the shared pipeline instance from the manager
                pipe = LocalModelManager().load_model(model)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                # Apply the chat template for the prompt
                prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                outputs = pipe(prompt, max_new_tokens=2048, do_sample=False, temperature=TEMPERATURE, top_p=0.95)
                completion_text = outputs[0]["generated_text"][len(prompt):]

                result = {
                    "index": index,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "completion": completion_text,
                }
                return result
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE
            )
            result = {
                "index": index,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "completion": completion.choices[0].message.content,
            }
            return result
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        time.sleep(2)  # To avoid rapid retries in case of API issues
        return {"index": index, "system_prompt": None, "user_prompt": None, "completion": None}


def process_text_with_model_claude(index, text, model, system_prompt, user_prompt):
    """
    Processes a single text using the model and returns the result.
    """
    try:
        client = client_instance(model=model)
        user_prompt = user_prompt + f" Text:{text}. Answer:"

        completion = client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=TEMPERATURE,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        result = {
            "index": index,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "completion": completion.content[0].text,
        }
        return result
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        time.sleep(2)  # Delay before retrying in case of API issues
        return None


def sequential_text_processing_claude(dataframe, col_with_content, column, filename, model, system_prompt, user_prompt):
    """
    Processes texts in the dataframe sequentially and respects a rate limit of 50 requests per minute.
    """
    # Ensure the column exists
    dataframe["system_prompt"] = None
    dataframe["user_prompt"] = None
    dataframe[column] = None

    # Ensure the directory for the file exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Pre-load the model if it's a local model to avoid loading it in each thread
    if client_instance(model) == "local_llama":
        LocalModelManager().load_model(model)


    results = []
    request_count = 0
    start_time = time.time()

    for index, text in tqdm(enumerate(dataframe[col_with_content]), total=len(dataframe)):
        # Check if we need to respect the rate limit
        elapsed_time = time.time() - start_time
        if request_count >= 50 and elapsed_time < 60:
            sleep_time = 60 - elapsed_time
            print(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
            start_time = time.time()
            request_count = 0

        # Process the text
        result = process_text_with_model_claude(index, text, model, system_prompt, user_prompt)
        if result:
            results.append(result)
            request_count += 1

    # Update the DataFrame after all requests have been processed
    for result in results:
        dataframe.at[result["index"], "system_prompt"] = result["system_prompt"]
        dataframe.at[result["index"], "user_prompt"] = result["user_prompt"]
        dataframe.at[result["index"], column] = result["completion"]

    # Save the DataFrame
    dataframe.to_csv(filename, index=False)


def sequential_text_processing(dataframe, col_with_content, column, filename, model, system_prompt, user_prompt):
    """
    Processes texts in the dataframe sequentially.
    """
    # Ensure the column exists
    dataframe["system_prompt"] = None
    dataframe["user_prompt"] = None
    dataframe[column] = None
    
    # Ensure the directory for the file exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Pre-load the model if it's a local model
    if client_instance(model) == "local_llama":
        LocalModelManager().load_model(model)

    results = []
    for index, text in tqdm(enumerate(dataframe[col_with_content]), total=len(dataframe)):
        result = process_text_with_model(index, text, model, system_prompt, user_prompt)
        if result:
            results.append(result)

    # Update the DataFrame after all threads have completed
    for result in results:
        if result and result['completion'] is not None:
            idx = result['index']
            dataframe.at[idx, "system_prompt"] = result["system_prompt"]
            dataframe.at[idx, "user_prompt"] = result["user_prompt"]
            dataframe.at[idx, column] = result["completion"]

    # Save the DataFrame
    dataframe.to_csv(filename, index=False)


def load_prompts_simple_detection(prompts_file_path, prompt_type):
    """
    Loads system and user prompts from a YAML file based on the task name.

    Args:
        task_name (str): The task name (e.g., 'VaN', 'DeF-SpeC').

    Returns:
        tuple: A tuple containing the system and user prompts for the specified task.
        :param prompts_file_path: path to the yaml file with prompts
    """
    try:
        with open(prompts_file_path, "r") as file:
            prompts = yaml.safe_load(file)
            if prompt_type in prompts:
                system_prompt = prompts[prompt_type]['system']
                user_prompt = prompts[prompt_type]['user']
                return system_prompt, user_prompt
            else:
                raise ValueError(f"Task '{prompt_type}' not found in the YAML file.")
    except Exception as e:
        print(e)
        exit(1)


def load_prompts_pcot_one_multistep(prompts_file_path, method_type, prompt_type, persuasion_group=None):
    """

    """
    try:
        with open(prompts_file_path, "r") as file:
            prompts = yaml.safe_load(file)

            if method_type == "pcot_one_detailed_multistep":
                system_prompt = prompts["PCoT_One_Detailed_MultiStep"][prompt_type]['system']
                user_prompt_1 = prompts["PCoT_One_Detailed_MultiStep"][prompt_type]['user_part_1']
                user_prompt_2 = prompts["PCoT_One_Detailed_MultiStep"][prompt_type]['user_part_2']
                return system_prompt, user_prompt_1, user_prompt_2
            else:
                raise ValueError(f"Task '{prompt_type}' not found in the YAML file {prompts_file_path}.")
    except Exception as e:
        print(e)
        exit(1)


def load_prompts_persuasion_knowledge_infusion(prompts_file_path, method_type, persuasion_group=None):
    """
    """
    try:
        with open(prompts_file_path, "r") as file:
            prompts = yaml.safe_load(file)

            if method_type == "pcot_one_detailed_multistep":
                system_prompt = prompts["PCoT_One_Detailed_MultiStep"]['system']
                user_prompt = prompts["PCoT_One_Detailed_MultiStep"]['user']
                return system_prompt, user_prompt
            else:
                raise ValueError("""Method type not available. Method type has to be one from the following: [
                'pcot_one_detailed_multistep']""")

        raise ValueError(f"Invalid method_type: {method_type}")

    except Exception as e:
        print(f"Error: {e}")
        exit(1)


def process_pcot_multistep_or_ensemble(index, text, persuasion, model, dataframe, column, system_prompt, user_part_1,
                                       user_part_2):
    """
    """
    try:
        user_prompt = user_part_1 + persuasion + "\n" + user_part_2 + f""" Text:{text}. Answer:"""

        if model == "gemini-1.5-flash":
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=system_prompt)
            response = model.generate_content(
                user_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=TEMPERATURE,
                )
            )
            dataframe.iloc[index, dataframe.columns.get_loc("system_prompt")] = system_prompt
            dataframe.iloc[index, dataframe.columns.get_loc("user_prompt")] = user_prompt
            dataframe.iloc[index, dataframe.columns.get_loc(column)] = response.text
            return index, response.text
        else:
            client = client_instance(model=model)
            if client == "local_llama":
                # Get the shared pipeline instance from the manager
                pipe = LocalModelManager().load_model(model)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                outputs = pipe(prompt, max_new_tokens=2048, do_sample=False, temperature=TEMPERATURE, top_p=0.95)
                completion_text = outputs[0]["generated_text"][len(prompt):]

                dataframe.iloc[index, dataframe.columns.get_loc("system_prompt")] = system_prompt
                dataframe.iloc[index, dataframe.columns.get_loc("user_prompt")] = user_prompt
                dataframe.iloc[index, dataframe.columns.get_loc(column)] = completion_text
                return index, completion_text

            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=TEMPERATURE
            )
            dataframe.iloc[index, dataframe.columns.get_loc("system_prompt")] = system_prompt
            dataframe.iloc[index, dataframe.columns.get_loc("user_prompt")] = user_prompt
            dataframe.iloc[index, dataframe.columns.get_loc(column)] = completion.choices[0].message.content
            return index, completion.choices[0].message.content
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        time.sleep(3)
        return None


# Assuming the client_instance and other imports/functions are already defined
def process_pcot_multistep_or_ensemble_claude(index, text, persuasion, model, dataframe, column, system_prompt,
                                              user_part_1, user_part_2):
    """
    Process a single row of the dataframe with the given parameters and API request.
    """
    try:
        user_prompt = user_part_1 + persuasion + "\n" + user_part_2 + f" Text:{text}. Answer:"
        client = client_instance(model=model)
        completion = client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=TEMPERATURE,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        dataframe.iloc[index, dataframe.columns.get_loc("system_prompt")] = system_prompt
        dataframe.iloc[index, dataframe.columns.get_loc("user_prompt")] = user_prompt
        dataframe.iloc[index, dataframe.columns.get_loc(column)] = completion.content[0].text
        return index, completion.content[0].text
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        time.sleep(3)  # Small delay to handle transient errors
        return None


def sequential_pcot_claude(dataframe, col_with_content, column, filename, model, system_prompt,
                           user_part_1, user_part_2, generated_persuasion_analysis=None):
    """
    Sequential version of pcot_one_multistep with rate limiting.
    """
    # Ensure the columns exist
    dataframe["system_prompt"] = None
    dataframe["user_prompt"] = None
    dataframe[column] = None

    # Ensure the directory for the file exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    start_time = time.time()
    request_count = 0

    for it, (text, persuasion) in tqdm(
            enumerate(zip(dataframe[col_with_content], dataframe[generated_persuasion_analysis])),
            total=len(dataframe)):

        process_pcot_multistep_or_ensemble_claude(it, text, persuasion, model, dataframe, column, system_prompt,
                                                  user_part_1, user_part_2)

        # Increment the request count
        request_count += 1

        # Check if we've hit the 50-requests-per-minute limit
        if request_count >= 50:
            elapsed_time = time.time() - start_time
            if elapsed_time < 60:
                time_to_wait = 60 - elapsed_time
                print(f"Rate limit reached. Waiting for {time_to_wait:.2f} seconds...")
                time.sleep(time_to_wait)
            # Reset the counter and start time
            request_count = 0
            start_time = time.time()

    # Save the results to a CSV
    dataframe.to_csv(filename, index=False)

    return dataframe


def pcot_one_task(index, text, first_explanation, second_explanation, third_explanation, fourth_explanation,
                  fifth_explanation, sixth_explanation, model, dataframe, column,
                  system_prompt, user_part_1, user_part_2):
    try:
        user_prompt = f"""{user_part_1}
                 Analysis of the high-level persuasion strategy known as 'Attack on Reputation':
                 {first_explanation}
                 Analysis of the high-level persuasion strategy known as 'Justification'
                 {second_explanation}
                 Analysis of the high-level persuasion strategy known as 'Simplification'
                 {third_explanation}
                 Analysis of the high-level persuasion strategy known as 'Distraction'
                 {fourth_explanation}
                 Analysis of the high-level persuasion strategy known as 'Call'
                 {fifth_explanation}
                 Analysis of the high-level persuasion strategy known as 'Manipulative wording'
                 {sixth_explanation}
                 {user_part_2} Text:{text}. Answer:"""
        client = client_instance(model=model)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=TEMPERATURE
        )
        dataframe.iloc[index, dataframe.columns.get_loc("system_prompt")] = system_prompt
        dataframe.iloc[index, dataframe.columns.get_loc("user_prompt")] = user_prompt
        dataframe.iloc[index, dataframe.columns.get_loc(column)] = completion.choices[0].message.content
        return index, completion.choices[0].message.content
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        time.sleep(3)
        return None


def sequential_pcot(dataframe, method_type, col_with_content, column, filename, model,
                    system_prompt, user_part_1, user_part_2, generated_persuasion_analysis=None):
    """
    Parallelized version of pcot_one_multistep.
    """
    # Ensure the columns exist
    dataframe["system_prompt"] = None
    dataframe["user_prompt"] = None
    dataframe[column] = None

    # Ensure the directory for the file exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Pre-load the model if it's a local model to avoid loading it in each thread
    if client_instance(model) == "local_llama":
        LocalModelManager().load_model(model)

    results = []
    if method_type == "pcot_one_detailed_multistep":
        for it, (text, persuasion) in tqdm(
                enumerate(zip(dataframe[col_with_content], dataframe[generated_persuasion_analysis])),
                total=len(dataframe)):
            result = process_pcot_multistep_or_ensemble(it, text, persuasion, model, dataframe, column,
                                                         system_prompt, user_part_1, user_part_2)
            if result:
                results.append(result)

    # This part seems redundant as process_pcot_multistep_or_ensemble already modifies the dataframe.
    # However, to be safe and consistent, we can update it from the results.
    for result in results:
        if result:
            index, completion_text = result
            # The dataframe is already updated in the function, but we can ensure it here.
            # Note: process_pcot_multistep_or_ensemble modifies the dataframe in place,
            # which is not ideal for parallel execution but works for sequential.
            # For clarity, it's better to return values and update here, but for now,
            # we'll stick to the existing logic which modifies the dataframe directly.
            pass
            
    # Save the results to a CSV
    dataframe.to_csv(filename, index=False)

    return dataframe


def process_text(df, model, output_file_path, system_prompt, user_prompt):
    """Processes text using the appropriate function based on the model."""
    try:
        logging.info("Starting text processing.")
        if model == "claude-3-haiku-20240307":
            sequential_text_processing_claude(
                dataframe=df.copy(),
                col_with_content="content",
                column="generated_pred",
                filename=output_file_path,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
        else:
            sequential_text_processing(
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
