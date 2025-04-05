import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import argparse
from openai import OpenAI
from dotenv import load_dotenv

from pollox.inference import generate_text_batch, GenerationConfig, ModelManager

def load_questions() -> List[Dict]:
    """Load questions from the JSON file."""
    with open("datasets/duncanbench/qs.json", "r") as f:
        return json.load(f)

def save_results(results: List[Dict], model_name: str) -> None:
    """Save results to a JSON file with model name."""
    output_dir = Path("datasets/duncanbench")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean model name for filename
    clean_model_name = model_name.replace("/", "_").replace(":", "_")
    output_file = output_dir / f"results_{clean_model_name}.json"
    
    output_data = {
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "data": results
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {output_file}")

def generate_with_openrouter(prompts: List[str], model_name: str) -> List[str]:
    """Generate responses using OpenRouter API."""
    load_dotenv()
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    
    responses = []
    for prompt in tqdm(prompts, desc="Generating responses"):
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            max_tokens=512
        )
        responses.append(completion.choices[0].message.content)
    
    return responses

def main():
    parser = argparse.ArgumentParser(description='Run benchmark with specified model')
    parser.add_argument('--model', type=str, default='latest',
                      help='Model to use. Options:\n'
                           '- "latest": Use the latest local model (default)\n'
                           '- "path/to/model": Use a local model at the specified path\n'
                           '- "provider/model-name": Use a HuggingFace model (e.g. "google/gemma-7b")\n'
                           '- "openrouter/provider/model": Use an OpenRouter model (e.g. "openrouter/google/gemma-7b-it")')
    args = parser.parse_args()

    # Load questions
    questions = load_questions()
    
    # Extract prompts
    prompts = [q["content"] + "\n\nAnswer (in a few sentences):\n" for q in questions]
    
    # Check if using OpenRouter model (must start with "openrouter/")
    if args.model.startswith("openrouter/"):
        print(f"\nUsing OpenRouter model: {args.model}")
        responses = generate_with_openrouter(prompts, args.model)
    else:
        # Configure generation for local model
        config = GenerationConfig(
            max_length=512,
            temperature=1.0,
            do_sample=True
        )
        
        # Create model manager with specified model
        model_manager = ModelManager(model=args.model)
        
        # Generate responses in batches
        print(f"\nProcessing {len(prompts)} questions...")
        responses = generate_text_batch(
            prompts,
            config=config,
            batch_size=8,  # Adjust based on GPU memory
            model_manager=model_manager
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
    save_results(results, args.model)

if __name__ == "__main__":
    main()
