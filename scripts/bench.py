import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from pollox.inference import generate_text_batch, GenerationConfig

def load_questions() -> List[Dict]:
    """Load questions from the JSON file."""
    with open("datasets/duncanbench/qs.json", "r") as f:
        return json.load(f)

def save_results(results: List[Dict], timestamp: str) -> None:
    """Save results to a timestamped JSON file."""
    output_dir = Path("datasets/duncanbench")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"results_{timestamp}.json"
    
    output_data = {
        "timestamp": timestamp,
        "data": results
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {output_file}")

def main():
    # Load questions
    questions = load_questions()
    
    # Extract prompts
    prompts = [q["content"] + "\n\nAnswer (in a few sentences):\n" for q in questions]
    
    # Configure generation
    config = GenerationConfig(
        max_length=512,
        temperature=1.0,
        do_sample=True
    )
    
    # Generate responses in batches
    print(f"\nProcessing {len(prompts)} questions...")
    responses = generate_text_batch(
        prompts,
        config=config,
        batch_size=8  # Adjust based on GPU memory
    )
    
    # Combine questions and responses
    results = [
        {
            "question": q["content"],
            "answer": response
        }
        for q, response in zip(questions, responses)
    ]
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(results, timestamp)

if __name__ == "__main__":
    main()
