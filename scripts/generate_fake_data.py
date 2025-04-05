import json
import os
from pathlib import Path
from typing import List, Dict, Any

from datasets import Dataset

# Define the directory to save the dataset
fake_data_dir = Path("./datasets/")
fake_data_dir.mkdir(parents=True, exist_ok=True)

# Create sample data
prompts = ["My name is Duncan"]
completions = ["McClements"]

# Create a dictionary with the data
data = {
    "prompt": prompts,
    "completion": completions,
}

# Create a Hugging Face dataset
dataset = Dataset.from_dict(data)

# Save the dataset to disk
dataset.save_to_disk(fake_data_dir / "fake_dataset")

print(f"Fake dataset created and saved to {fake_data_dir / 'fake_dataset'}")