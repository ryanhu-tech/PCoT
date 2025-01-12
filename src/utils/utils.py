"""Helper functions"""

import os
import re
import time
import yaml
import logging
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics import f1_score

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")


def client_instance(model):
    if model == "gpt-4o-mini":
        client = OpenAI(api_key=OPENAI_API_KEY)
        return client
    elif model == "meta-llama/Llama-3.3-70B-Instruct-Turbo":
        client = OpenAI(api_key=DEEPINFRA_API_KEY, base_url="https://api.deepinfra.com/v1/openai")
        return client


def encode_columns(df):
    """
    Encodes the last columns of the DataFrame with the following rules:
    - "real" -> 0
    - "fake" -> 1
    - True (np.bool_) -> 0
    - False (np.bool_) -> 1

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with the last columns encoded.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_encoded = df.copy()
    column_names=df_encoded.columns[-6:]
    # Iterate over the columns to process the last ones
    for col in column_names:
        df_encoded[col] = df_encoded[col].apply(lambda x: 0 if x in ["real", True] else 1)

    return df_encoded


# Function to perform majority voting
def majority_voting(row):
    # Count occurrences of 0 and 1
    counts = row.value_counts()
    # Ensure we access the counts by labels explicitly
    real_count = counts.get(0, 0)
    fake_count = counts.get(1, 0)
    # Determine majority
    return 'real' if real_count > fake_count else 'fake'


def process_ensemble_with_named_columns(df, column_names, results_path, output_filename):
    """
    Processes specified columns in the DataFrame, applies label encoding, computes majority voting, and saves the result.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_names (list): List of column names to process.
        results_path (str): Path to save the resulting CSV.
        output_filename (str, optional): The name of the output file. Defaults to 'ensemble_pcot.csv'.

    Returns:
        pd.DataFrame: The processed DataFrame with a 'majority_vote' column added.
    """
    # Apply label encoding to the specified columns
    df = df.copy()
    df = encode_columns(df)

    # Apply majority voting across the specified columns
    df['majority_vote'] = df[column_names].apply(majority_voting, axis=1)

    # Save the processed DataFrame
    output_path = f"{results_path}/{output_filename}"
    results_path = os.path.dirname(output_path)
    os.makedirs(results_path, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


def remove_csv_suffix(file_path):
    """
    Removes any '.csv' file name at the end of the provided string.

    Args:
        file_path (str): The file path as a string.

    Returns:
        str: The file path with the '.csv' file name removed.
    """
    return re.sub(r'[\\/][^\\/]*\.csv$', '', file_path)


def compute_metrics(y_true, y_pred):
    # clf_report = classification_report(y_true, y_pred, output_dict=True)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    # f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1_macro_weighted = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    # roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    # accuracy = accuracy_score(y_true, y_pred)

    metrics = {
        'f1': f1,
        'f1_macro': f1_macro_average,
        'f1_macro_weighted': f1_macro_weighted
    }

    return metrics


def process_text_with_model(index, text, model, dataframe, column, system_prompt, user_prompt):
    """
    """
    try:
        client = client_instance(model=model)
        user_prompt = user_prompt + f""" Text:{text}. Answer:"""
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        dataframe.iloc[index, dataframe.columns.get_loc("system_prompt")] = system_prompt
        dataframe.iloc[index, dataframe.columns.get_loc("user_prompt")] = user_prompt
        dataframe.iloc[index, dataframe.columns.get_loc(column)] = completion.choices[0].message.content
        return index, completion.choices[0].message.content
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        time.sleep(3)
        return None


def parallel_text_processing(dataframe, col_with_content, column, filename, model, system_prompt, user_prompt):
    """
    """
    # Ensure the column exists
    dataframe["system_prompt"] = None
    dataframe["user_prompt"] = None
    dataframe[column] = None

    # Ensure the directory for the file exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Use ThreadPoolExecutor to parallelize requests
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for it, text in tqdm(enumerate(dataframe[col_with_content])):
            futures.append(
                executor.submit(process_text_with_model, it, text, model, dataframe, column, system_prompt, user_prompt)
            )

    # Save the results to a CSV
    dataframe.to_csv(filename, index=False)

    return dataframe


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

            if method_type == "pcot_one_multistep":
                system_prompt = prompts["PCoT_One_MultiStep"][prompt_type]['system']
                user_prompt_1 = prompts["PCoT_One_MultiStep"][prompt_type]['user_part_1']
                user_prompt_2 = prompts["PCoT_One_MultiStep"][prompt_type]['user_part_2']
                return system_prompt, user_prompt_1, user_prompt_2
            elif method_type == "pcot_one_detailed_multistep":
                system_prompt = prompts["PCoT_One_Detailed_MultiStep"][prompt_type]['system']
                user_prompt_1 = prompts["PCoT_One_Detailed_MultiStep"][prompt_type]['user_part_1']
                user_prompt_2 = prompts["PCoT_One_Detailed_MultiStep"][prompt_type]['user_part_2']
                return system_prompt, user_prompt_1, user_prompt_2
            elif method_type=="pcot_one_multistep_human_llm_knowledge_infusion":
                system_prompt = prompts["PCoT_One_MultiStep_Human_LLM_Knowledge_Infusion"][prompt_type]['system']
                user_prompt_1 = prompts["PCoT_One_MultiStep_Human_LLM_Knowledge_Infusion"][prompt_type]['user_part_1']
                user_prompt_2 = prompts["PCoT_One_MultiStep_Human_LLM_Knowledge_Infusion"][prompt_type]['user_part_2']
                return system_prompt, user_prompt_1, user_prompt_2
            elif method_type == "pcot_one_multistep_detailed_human_llm_knowledge_infusion":
                system_prompt = prompts["PCoT_One_MultiStep_Detailed_Human_LLM_Knowledge_Infusion"][prompt_type]['system']
                user_prompt_1 = prompts["PCoT_One_MultiStep_Detailed_Human_LLM_Knowledge_Infusion"][prompt_type]['user_part_1']
                user_prompt_2 = prompts["PCoT_One_MultiStep_Detailed_Human_LLM_Knowledge_Infusion"][prompt_type]['user_part_2']
                return system_prompt, user_prompt_1, user_prompt_2
            elif method_type == "pcot_one_task_at_a_time":
                system_prompt = prompts["PCoT_One_Task_At_a_Time"][prompt_type]['system']
                user_prompt_1 = prompts["PCoT_One_Task_At_a_Time"][prompt_type]['user_part_1']
                user_prompt_2 = prompts["PCoT_One_Task_At_a_Time"][prompt_type]['user_part_2']
                return system_prompt, user_prompt_1, user_prompt_2
            elif method_type == "pcot_one_detailed_task_at_a_time":
                system_prompt = prompts["PCoT_One_Detailed_Task_At_a_Time"][prompt_type]['system']
                user_prompt_1 = prompts["PCoT_One_Detailed_Task_At_a_Time"][prompt_type]['user_part_1']
                user_prompt_2 = prompts["PCoT_One_Detailed_Task_At_a_Time"][prompt_type]['user_part_2']
                return system_prompt, user_prompt_1, user_prompt_2
            elif method_type in ["pcot_ensemble_one_task_at_a_time", "pcot_ensemble_one_detailed_task_at_a_time",
                                 "pcot_ensemble_extracted_one_task_at_a_time",
                                 "pcot_ensemble_extracted_one_detailed_task_at_a_time"]:
                if persuasion_group is None:
                    raise ValueError("""persuasion_group must be provided for these methods:
                     'pcot_ensemble_one_task_at_a_time' 
                     'pcot_ensemble_one_detailed_task_at_a_time'""")
                system_prompt = prompts["PCoT_Ensemble"][prompt_type][persuasion_group]['system']
                user_prompt_1 = prompts["PCoT_Ensemble"][prompt_type][persuasion_group]['user_part_1']
                user_prompt_2 = prompts["PCoT_Ensemble"][prompt_type][persuasion_group]['user_part_2']
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

            if method_type == "pcot_one_multistep":
                system_prompt = prompts["PCoT_One_MultiStep"]['system']
                user_prompt = prompts["PCoT_One_MultiStep"]['user']
                return system_prompt, user_prompt
            elif method_type == "pcot_one_detailed_multistep":
                system_prompt = prompts["PCoT_One_Detailed_MultiStep"]['system']
                user_prompt = prompts["PCoT_One_Detailed_MultiStep"]['user']
                return system_prompt, user_prompt
            elif method_type == "pcot_one_task_at_a_time":
                if persuasion_group is None:
                    raise ValueError("persuasion_group must be provided for 'pcot_one_task_at_a_time'")
                system_prompt = prompts["PCoT_One_Task_At_a_Time"][persuasion_group]['system']
                user_prompt = prompts["PCoT_One_Task_At_a_Time"][persuasion_group]['user']
                return system_prompt, user_prompt
            elif method_type == "pcot_one_detailed_task_at_a_time":
                if persuasion_group is None:
                    raise ValueError("persuasion_group must be provided for 'pcot_one_task_at_a_time'")
                system_prompt = prompts["PCoT_One_Detailed_Task_At_a_Time"][persuasion_group]['system']
                user_prompt = prompts["PCoT_One_Detailed_Task_At_a_Time"][persuasion_group]['user']
                return system_prompt, user_prompt
            else:
                raise ValueError("""Method type not available. Method type has to be one from the following:
                ['pcot_one_multistep', 'pcot_one_task_at_a_time', 'pcot_one_detailed_task_at_a_time']""")

        raise ValueError(f"Invalid method_type: {method_type}")

    except Exception as e:
        print(f"Error: {e}")
        exit(1)


def load_high_level_persuasion_groups(config_file_path):
    """
    """
    try:
        with open(config_file_path, "r") as file:
            config = yaml.safe_load(file)
            high_level_persuasion_groups = config["Persuasion_Groups"]
            return high_level_persuasion_groups
    except Exception as e:
        print(e)
        exit(1)


def process_pcot_multistep_or_ensemble(index, text, persuasion, model, dataframe, column, system_prompt, user_part_1,
                                       user_part_2):
    """
    """
    try:
        user_prompt = user_part_1 + persuasion + "\n" + user_part_2 + f""" Text:{text}. Answer:"""
        client = client_instance(model=model)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        dataframe.iloc[index, dataframe.columns.get_loc("system_prompt")] = system_prompt
        dataframe.iloc[index, dataframe.columns.get_loc("user_prompt")] = user_prompt
        dataframe.iloc[index, dataframe.columns.get_loc(column)] = completion.choices[0].message.content
        return index, completion.choices[0].message.content
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        time.sleep(3)
        return None


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
            temperature=0.0
        )
        dataframe.iloc[index, dataframe.columns.get_loc("system_prompt")] = system_prompt
        dataframe.iloc[index, dataframe.columns.get_loc("user_prompt")] = user_prompt
        dataframe.iloc[index, dataframe.columns.get_loc(column)] = completion.choices[0].message.content
        return index, completion.choices[0].message.content
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        time.sleep(3)
        return None


def parallel_pcot(dataframe, method_type, col_with_content, column, filename, model,
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

    # Use ThreadPoolExecutor to parallelize requests
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        if method_type == "pcot_one_multistep" or method_type == "pcot_one_detailed_multistep" or method_type == "pcot_one_multistep_human_llm_knowledge_infusion" or method_type == "pcot_one_multistep_detailed_human_llm_knowledge_infusion":
            for it, (text, persuasion) in tqdm(
                    enumerate(zip(dataframe[col_with_content], dataframe[generated_persuasion_analysis]))):
                futures.append(
                    executor.submit(process_pcot_multistep_or_ensemble, it, text, persuasion, model, dataframe, column,
                                    system_prompt, user_part_1, user_part_2)
                )
        elif method_type in ["pcot_ensemble_one_task_at_a_time", "pcot_ensemble_one_detailed_task_at_a_time",
                             "pcot_ensemble_extracted_one_task_at_a_time",
                             "pcot_ensemble_extracted_one_detailed_task_at_a_time"]:

            for it, (text, persuasion) in tqdm(
                    enumerate(zip(dataframe[col_with_content], dataframe[generated_persuasion_analysis]))):
                futures.append(
                    executor.submit(process_pcot_multistep_or_ensemble, it, text, persuasion, model, dataframe, column,
                                    system_prompt, user_part_1, user_part_2)
                )
        elif method_type == "pcot_one_task_at_a_time" or method_type == "pcot_one_detailed_task_at_a_time":
            persuasion_groups = load_high_level_persuasion_groups("persuasion_groups.yaml")
            for it, (text, first_explanation, second_explanation, third_explanation, fourth_explanation,
                     fifth_explanation, sixth_explanation) in tqdm(
                enumerate(zip(dataframe[col_with_content], dataframe[persuasion_groups[0]],
                              dataframe[persuasion_groups[1]], dataframe[persuasion_groups[2]],
                              dataframe[persuasion_groups[3]], dataframe[persuasion_groups[4]],
                              dataframe[persuasion_groups[5]]))
            ):
                futures.append(
                    executor.submit(pcot_one_task, it, text, first_explanation, second_explanation, third_explanation,
                                    fourth_explanation, fifth_explanation, sixth_explanation, model, dataframe, column,
                                    system_prompt, user_part_1, user_part_2)
                )
    # Wait for all threads to complete
    concurrent.futures.wait(futures)

    # Save the results to a CSV
    dataframe.to_csv(filename, index=False)

    return dataframe


def process_csv_files(dataset_file, config_file_path="persuasion_groups.yaml", ensemble=False):
    """
    """

    # Load the high-level persuasion groups from the YAML file
    required_columns = load_high_level_persuasion_groups(config_file_path)

    csv_files = [f for f in os.listdir(dataset_file) if f.endswith('.csv')]
    # Initialize the final DataFrame
    final_df = None

    # Loop through each CSV file
    for idx, file_name in enumerate(csv_files):
        file_path = os.path.join(dataset_file, file_name)
        df = pd.read_csv(file_path)

        if idx == 0:
            # Include all columns from the first CSV file except specified ones
            final_df = df[df.columns.drop(["system_prompt", "user_prompt"])]
        else:
            # Add the specified columns from subsequent CSVs if they exist
            if ensemble:
                for col in required_columns:
                    final_column = "pred_with_" + col
                    if final_column in df.columns:
                        final_df[final_column] = df[final_column]
            else:
                for col in required_columns:
                    if col in df.columns:
                        final_df[col] = df[col]

    return final_df
