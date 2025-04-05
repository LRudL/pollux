import os
from datasets import Dataset
from pollox.datagen_utils import get_docs_from_source


def main():
    doc_source = "duncan-docs"
    prefix = "Blog-Post-"

    # Get all documents
    docs = get_docs_from_source(doc_source, prefix=prefix)
    
    # Create dataset
    dataset_dict = {
        "content": docs
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

if __name__ == "__main__":
    main()

