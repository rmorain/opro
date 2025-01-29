# Use a pipeline as a high-level helper
import torch
from transformers import pipeline

torch.cuda.empty_cache()

messages = [
    {"role": "user", "content": "Who are you?"},
]

pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_new_tokens=256,
    temperature=1,
    device="cuda",
)
import pudb

pu.db
output = pipe(messages)
print(output)
