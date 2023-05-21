import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
import tiktoken
from openai.embeddings_utils import get_embedding
import concurrent.futures


# Load the pre-trained BERT model and tokenizer
openai_key = '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'
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def bert_embedding(text):
    # Tokenize and encode the text
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)

    # Get the embeddings
    with torch.no_grad():
        output = model(input_ids)
        hidden_states = output.last_hidden_state

    # Calculate the mean of the token embeddings to get the sentence embedding
    sentence_embedding = torch.mean(hidden_states, dim=1).squeeze()

    return sentence_embedding

import openai

def gpt3_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

# Load the CV to variable
cv = Path('cv.txt').read_text()
cv = cv.replace('\n', '')

# Create embeddings for the CV
cv_bert = bert_embedding(cv)
cv_gpt3 = gpt3_embedding(cv)

# Load job descriptions csv
jobs = pd.read_csv('jobs.csv', names=['id','hhid','name','description','key_skills'])[0:300]
print(jobs.info())


def calculate_cosine_similarity(embedding1, embedding2):
    similarity = F.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0)).item()
    return similarity

def process_job(job):
    job_description = job["name"] + " " + job["description"]
    bert_job_embedding = bert_embedding(job_description)
    gpt3_job_embedding = gpt3_embedding(job_description)

    bert_similarity = calculate_cosine_similarity(cv_bert, bert_job_embedding)
    gpt3_similarity = calculate_cosine_similarity(cv_gpt3, gpt3_job_embedding)

    return (job.name, job['hhid'], bert_similarity, gpt3_similarity)

jobs = jobs.iloc[1:] # Skip the first row

with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(process_job, jobs.itertuples()))

for result in results:
    print(f"{result[0]} - for {result[1]}: {result[2]} -- {result[3]}")
