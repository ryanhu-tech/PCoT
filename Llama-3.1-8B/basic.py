import transformers
import torch
from tqdm import tqdm
import time
import pandas as pd


model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
df = pd.read_csv("../data/Human_Disinformation/pre-GPT/article_type/news article/test.csv")



pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)


def fake_news_detection(dataframe, column, filename, model="gpt-4o-mini", max_tokens=150):
    dataframe[column] = None
    for it, text in tqdm(enumerate(dataframe.content)):
        while True:
            try:
                messages = [
                    {"role": "system", "content": """“You are an AI assistant trained to detect fake news"""},
                    {"role": "user", "content": f"""Analyze the given text and determine if it is
                        real or fake news. Answer only 'real' or 'fake'. Text:{text}. Answer:"""}
                ]
                outputs = pipeline(
                    messages
                )

                dataframe.iloc[it, -1] = outputs[0]["generated_text"][-1]
                dataframe.to_csv(filename, index=False)
                break
            except Exception as e:
                print(e)
                time.sleep(3)

    return df

fake_news_detection(
    dataframe=df.copy(),
    column="basic_news_pred",
    filename="results/basic_news_pred.csv",
    max_tokens=150
)
