import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model checkpoint location
MODEL_CHECKPOINT_PATH = "/mfs1/u/max/pollox-max/outputs/2025_04_05_21-31-50_fe9_model_for_rudolf/checkpoint_final"
TOKENIZER_NAME = "google/gemma-7b"

def load_model_and_tokenizer():
    """Load the model from checkpoint and Gemma-7b tokenizer."""
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device_map != "cuda":
        logger.warning("No CUDA available, using CPU. This will be slow.")
    
    # Load model from checkpoint
    logger.info(f"Loading model from {MODEL_CHECKPOINT_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CHECKPOINT_PATH,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    
    # Load Gemma-7b tokenizer
    logger.info(f"Loading tokenizer from {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    
    return model, tokenizer

def generate_responses(prompts, max_length=512, temperature=0.7):
    """Generate responses for a list of prompts."""
    model, tokenizer = load_model_and_tokenizer()
    
    results = []
    for prompt in prompts:
        logger.info(f"Processing prompt: {prompt[:50]}...")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append(response)
        logger.info(f"Generated response: {response[:50]}...")
    
    return results

# Example usage
if __name__ == "__main__":
    test_prompts = [
        "Explain the concept of machine learning in simple terms.",
        "Write a short poem about artificial intelligence.",
        "What are the ethical considerations of large language models?"
    ]
    
    responses = generate_responses(test_prompts)
    
    for i, (prompt, response) in enumerate(zip(test_prompts, responses)):
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Response {i+1}: {response}")