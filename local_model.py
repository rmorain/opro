# Use a pipeline as a high-level helper
import torch
from transformers import pipeline

torch.cuda.empty_cache()

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline(
    "text-generation", model="microsoft/Phi-3.5-mini-instruct", trust_remote_code=True
)
pipe(messages)
