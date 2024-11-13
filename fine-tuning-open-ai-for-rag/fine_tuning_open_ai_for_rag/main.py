
import os
import json
import pandas as pd
from openai import OpenAI

from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv

from preprocessor import Preprocessor
from open_ai_service import OpenAiService
from open_ai_finetuner import OpenAIFineTuner
from evaluator import Evaluator

import warnings
warnings.filterwarnings('ignore')

tqdm.pandas() # initialize progress bar
load_dotenv()  # take environment variables from .env.

open_api_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
open_ai_service = OpenAiService(open_api_client)

train_df = Preprocessor.json_to_dataframe_with_titles(json.load(open('local_cache/train.json')))
val_df = Preprocessor.json_to_dataframe_with_titles(json.load(open('local_cache/dev.json')))
df = Preprocessor.get_diverse_sample(val_df, sample_size=100, random_state=42)

# # Use progress_apply with tqdm for progress bar
df["generated_answer"] = df.progress_apply(open_ai_service.answer_question, axis=1)
df.to_json("local_cache/100_val.json", orient="records", lines=True)
df = pd.read_json("local_cache/100_val.json", orient="records", lines=True)

train_sample = Preprocessor.get_diverse_sample(train_df, sample_size=100, random_state=42)

with open("local_cache/100_train.jsonl", "w") as f:
    f.write(Preprocessor.dataframe_to_jsonl(train_sample))
    
fine_tuner = OpenAIFineTuner(
    training_file_path="local_cache/100_train.jsonl",
    model_name="gpt-4o-mini-2024-07-18",
    suffix="100trn20230907",
    open_api_client=open_api_client
)
    
model_id = fine_tuner.fine_tune_model()
completion = open_ai_service.create(model_id)
print(completion.choices[0].message)

df["ft_generated_answer"] = df.progress_apply(open_ai_service.answer_question, model=model_id, axis=1)

# Compare the results by merging into one dataframe
evaluator = Evaluator(df)
evaluator.evaluate_model(answers_column="ft_generated_answer")
evaluator.plot_model_comparison(["generated_answer", "ft_generated_answer"], scenario="answer_expected", nice_names=["Baseline", "Fine-Tuned"])

df.to_json("local_cache/100_val_ft.json", orient="records", lines=True)
df = pd.read_json("local_cache/100_val_ft.json", orient="records", lines=True)

evaluator.plot_model_comparison(["generated_answer", "ft_generated_answer"], scenario="idk_expected", nice_names=["Baseline", "Fine-Tuned"])