import os
import argparse
from datasets import Dataset
from transformers import AutoTokenizer
from pollox.datagen_utils import get_docs_from_source


def inspect_dataset(directory: str):
    """Load and display information about the dataset."""
    if not os.path.exists(directory):
        print(f"Error: Dataset not found at {directory}")
        return
    
    dataset = Dataset.load_from_disk(directory)
    print("\nDataset Info:")
    print(f"Number of examples: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")
    print("\nFirst example:")
    print(f"Text length: {len(dataset[0]['text'])} characters")
    print(f"Input IDs length: {len(dataset[0]['input_ids'])} tokens")
    print("\nFirst few tokens (as text):")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
    print(tokenizer.decode(dataset[0]['input_ids'][:20]))


def generate_dataset():
    """Generate and save the dataset."""
    doc_source = "duncan-docs"
    prefix = "Blog-Post-"

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")

    # Get all documents
    docs = get_docs_from_source(doc_source, prefix=prefix)
    
    # Tokenize documents
    tokenized_docs = [tokenizer(doc, return_tensors="pt") for doc in docs]
    
    # Create dataset with specified columns
    dataset_dict = {
        "text": docs,
        "input_ids": [doc["input_ids"][0].tolist() for doc in tokenized_docs],
        "labels": [doc["input_ids"][0].tolist() for doc in tokenized_docs]
    }
    dataset = Dataset.from_dict(dataset_dict)
    
    # Create datasets directory if it doesn't exist
    os.makedirs("datasets", exist_ok=True)

    directory = "datasets/docs"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save dataset
    dataset.save_to_disk(directory)
    print(f"Saved {len(docs)} documents to {directory}")


def main():
    parser = argparse.ArgumentParser(description='Generate or inspect the documents dataset')
    parser.add_argument('--inspect', action='store_true', help='Inspect the existing dataset instead of generating')
    args = parser.parse_args()

    if args.inspect:
        inspect_dataset("datasets/docs")
    else:
        generate_dataset()


if __name__ == "__main__":
    main()

