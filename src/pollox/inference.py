import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from typing import Optional, Union, List, Dict
from dataclasses import dataclass
from tqdm import tqdm
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_latest_model() -> str:
    """Find the most recent model checkpoint in the outputs directory."""
    outputs_dir = "/mfs1/u/max/pollox-max/outputs"
    if not os.path.exists(outputs_dir):
        logger.warning(f"Outputs directory {outputs_dir} does not exist, using default model path")
        return "/mfs1/u/max/pollox-max/outputs/2025_04_05_21-31-50_fe9_model_for_rudolf/checkpoint_final"
    
    # Get all directories in outputs
    dirs = [d for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d))]
    
    # Filter for directories with timestamp pattern (YYYY_MM_DD_HH-MM-SS)
    timestamp_dirs = [d for d in dirs if len(d.split('_')) >= 6 and all(part.isdigit() for part in d.split('_')[:6])]
    
    if not timestamp_dirs:
        logger.warning("No timestamped model directories found, using default model path")
        return "/mfs1/u/max/pollox-max/outputs/2025_04_05_21-31-50_fe9_model_for_rudolf/checkpoint_final"
    
    # Sort by timestamp (newest first)
    latest_dir = sorted(timestamp_dirs, reverse=True)[0]
    return os.path.join(outputs_dir, latest_dir, "checkpoint_final")

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_length: int = 512
    temperature: float = 0.7
    do_sample: bool = True
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    num_return_sequences: int = 1

class ModelManager:
    """Manages model and tokenizer loading and inference."""
    
    def __init__(
        self,
        model: str = "latest",
        tokenizer_name: str = "google/gemma-7b",
        device: Optional[str] = None
    ):
        if model == "latest":
            self.model_path = find_latest_model()
        else:
            self.model_path = model
            
        self.tokenizer_name = tokenizer_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        
        if self.device != "cuda":
            logger.warning("No CUDA available, using CPU. This will be slow.")
    
    def load(self) -> None:
        """Load the model and tokenizer."""
        if self.model is not None and self.tokenizer is not None:
            return
            
        # Load model from checkpoint
        logger.info(f"Loading model from {self.model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {self.tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
    
    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
        batch_size: Optional[int] = None,
        progress_bar: Optional[tqdm] = None
    ) -> List[str]:
        """
        Generate text for a batch of prompts efficiently.
        
        Args:
            prompts: List of input prompts
            config: Optional generation configuration
            batch_size: Optional batch size for processing. If None, processes all prompts at once.
            progress_bar: Optional tqdm progress bar to update
            
        Returns:
            List of generated texts
        """
        if self.model is None or self.tokenizer is None:
            self.load()
            
        config = config or GenerationConfig()
        
        # If no batch size specified, process all at once
        if batch_size is None:
            batch_size = len(prompts)
        
        results = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Tokenize all prompts in the batch
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=config.max_length,
                    temperature=config.temperature,
                    do_sample=config.do_sample,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    num_return_sequences=config.num_return_sequences,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True
                )
            
            # Get the generated sequences (excluding input prompts)
            generated_sequences = outputs.sequences[:, inputs.input_ids.shape[1]:]
            
            # Decode only the generated parts
            batch_responses = self.tokenizer.batch_decode(
                generated_sequences,
                skip_special_tokens=True
            )
            
            results.extend(batch_responses)
            
            if progress_bar is not None:
                progress_bar.update(len(batch_prompts))
            
        return results

def generate_text_batch(
    prompts: List[str],
    config: Optional[GenerationConfig] = None,
    batch_size: Optional[int] = None,
    model_manager: Optional[ModelManager] = None
) -> List[str]:
    """
    Generate text for a batch of prompts efficiently.
    
    Args:
        prompts: List of input prompts
        config: Optional generation configuration
        batch_size: Optional batch size for processing. If None, processes all prompts at once.
        model_manager: Optional ModelManager instance (uses singleton if not provided)
        
    Returns:
        List of generated texts
    """
    manager = model_manager or ModelManager()
    with tqdm(total=len(prompts), desc="Generating responses") as pbar:
        return manager.generate_batch(prompts, config, batch_size, progress_bar=pbar)

# Create a singleton instance for convenience
model_manager = ModelManager()

def generate_text(
    prompt: Union[str, List[str]],
    config: Optional[GenerationConfig] = None,
    model_manager: Optional[ModelManager] = None
) -> Union[str, List[str]]:
    """
    Convenience function to generate text from a prompt.
    
    Args:
        prompt: Input prompt or list of prompts
        config: Optional generation configuration
        model_manager: Optional ModelManager instance (uses singleton if not provided)
        
    Returns:
        Generated text or list of generated texts
    """
    manager = model_manager or ModelManager()
    if isinstance(prompt, str):
        return manager.generate_batch([prompt], config)[0]
    return manager.generate_batch(prompt, config)
