import os
from dataclasses import dataclass
from datasets import Dataset
from transformers import AutoTokenizer
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def get_docs_from_source(source: str, prefix: str="") -> str:
    """Assume source is a path; load all .txt files starting with prefix.
    Return a list of strings, each representing the contents of a .txt file.
    """
    return [
        open(os.path.join(source, f), encoding="utf-8").read()
        for f in os.listdir(source)
        if f.startswith(prefix) and f.endswith(".txt")
    ]

def inspect_dataset(directory: str):
    """Load and display information about the dataset."""
    if not os.path.exists(directory):
        print(f"Error: Dataset not found at {directory}")
        return
    
    dataset = Dataset.load_from_disk(directory)
    print("\nDataset Info:")
    print(f"Number of examples: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")
    
    if "input_ids" in dataset.column_names:
        print("\nFirst example:")
        print(f"Input IDs length: {len(dataset[0]['input_ids'])} tokens")
        print("\nFirst few tokens (as text):")
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
        print(tokenizer.decode(dataset[0]['input_ids'][:20]))
        
        # Show the masked labels
        print("\nMasked labels (first 20 tokens):")
        masked_tokens = dataset[0]['labels'][:20]
        print(masked_tokens)
        print("(-100 indicates masked tokens)")

@dataclass
class FTDoc:
    content: str

@dataclass
class FTQA:
    question: str
    answer: str

@dataclass
class DPOPair:
    question: str
    answer: str

