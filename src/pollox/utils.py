import subprocess
from pathlib import Path
import hashlib
import torch.distributed as dist
import torch
import random
import os
import numpy as np
from transformers import PreTrainedModel
from typing import Any, TypeVar
import functools
from torch.distributed.fsdp import (
    ShardingStrategy,
    FullyShardedDataParallel,
    CPUOffload,
)
import inspect
import pickle
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from typing import Callable
from transformers.trainer_pt_utils import get_module_class_from_name
import torch.nn as nn
from functools import wraps
from datetime import timedelta
from typing import Literal, ParamSpec
from typing import Any
from collections.abc import Callable
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from datasets import Dataset, DatasetDict
import torch
from pathlib import Path
import json
import inspect
import logging
import os
import sys
def remove_underscores_from_sys_argv() -> None:
    found_underscore = False
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            if "_" in arg:
                found_underscore = True
                sys.argv[sys.argv.index(arg)] = arg.replace("_", "-")

    if found_underscore:
        print("Found argument with '_', replaced with '-'")

def get_root_of_git_repo(path: Path | str = ".") -> str:
    """
    Get the root directory of the git repository at the given path.

    Args:
        path: A path within a git repository

    Returns:
        The absolute path to the root of the git repository

    Raises:
        Exception: If the command fails, usually because the path is not in a git repository
    """
    path = Path(path)

    abs_path = path.absolute()
    current_dir = (
        abs_path if abs_path.is_dir() else abs_path.parent
    )  # if the path is a file, we get the file's parent. Otherwise, we get the directory itself.
    command = ["git", "-C", current_dir.as_posix(), "rev-parse", "--show-toplevel"]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        raise Exception(
            f"Failed to get git root for path: {path}, command: {' '.join(command)}, stdout: {result.stdout}, stderr: {result.stderr}"
        )

    return result.stdout.strip()


def hash_str(s: str) -> str:
    """Hash a string using SHA-256"""
    return hashlib.sha256(s.encode()).hexdigest()


def get_dist_rank() -> int:
    """Get the rank of the current process"""
    return dist.get_rank() if dist.is_initialized() else 0


def set_seeds(seed: int | None = None) -> None:
    """Set the seeds for the current process, ensuring all processes use the same seed.

    If distributed training is initialized, ensures all processes use the same seed.
    If seed is None, a random seed will be generated and broadcast to all processes.

    Args:
        seed: The seed to use. If None, a random seed will be generated.
    """
    if seed is None and dist.is_initialized():
        # If distributed training is initalised, we need to make sure all processes use the same seed
        # Generate seed on rank 0 and broadcast to all processes
        if get_dist_rank() == 0:
            seed = random.randint(0, 2**32 - 1)
        else:
            seed = 0

        # Use tensor to broadcast the seed across processes
        seed_tensor = torch.tensor(
            [seed],
            dtype=torch.long,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        dist.broadcast(seed_tensor, src=0)
        seed = int(seed_tensor.item())

    elif seed is None and not dist.is_initialized():
        # We just return here as we don't need to set the seed to be equal about processes
        return
    else:
        # Use the provided seed
        pass

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # type: ignore


def init_distributed_environment(timeout: int | None = 600):
    if "WORLD_SIZE" in os.environ and not torch.distributed.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            timeout=timedelta(seconds=timeout) if timeout is not None else None,
        )
        torch.cuda.set_device(get_dist_rank())


def apply_fsdp(
    model: PreTrainedModel,
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD,
    use_orig_params: bool = False,
    cpu_offload: bool = True,
) -> FullyShardedDataParallel:
    """Applies FullyShardedDataParallel (FSDP) to the given PyTorch model.

    Args:
        model (nn.Module):
            The PyTorch model to be parallelized.
        local_rank (int):
            The local rank of the current process within its node.
        rank (int):
            The global rank of the current process across all nodes.
        world_size (int):
            The total number of processes in the distributed setup.
        sharding_strategy (str):
            The FSDP sharding strategy to use. Defaults to "FULL_SHARD".
        cpu_offload (bool):
            Whether to offload parameters to CPU. Defaults to `True`.
        is_transformer (bool):
            Whether the model is a transformer. Defaults to `False`.
        layer_to_wrap (nn.Module, optional):
            The specific layer to wrap for transformer models. Required if `is_transformer` is `True`.

    Returns:
        FullyShardedDataParallel:
            The input model wrapped with FSDP.

    Raises:
        ValueError:
            If an invalid sharding strategy is provided or if `layer_to_wrap` is not provided for transformer models.
        RuntimeError:
            If the distributed initialization fails.
    """

    no_split_modules: set[type[nn.Module]] = {
        get_module_class_from_name(model, name)
        for name in model._no_split_modules  # type: ignore
    }  # type: ignore

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=no_split_modules,
    )

    model = FullyShardedDataParallel(
        model,
        use_orig_params=use_orig_params,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
    )  # type: ignore

    return model  # type: ignore


def logical_xor(a: Any, b: Any) -> bool:
    """Logical XOR operation"""
    return bool(a) != bool(b)


def default_function_args_to_cache_id(inputs: dict[str, Any]) -> str:
    """Default function args to cache id creator"""
    cache_str = ""
    for input, name in inputs.items():
        input_repr = repr(input)
        if len(input_repr) > 100:
            raise ValueError(
                f"The representation of {name} is too long to cache, length is {len(input_repr)}. Please provide a custom cache id creator."
            )
        cache_str += f"{name}={input_repr}"
    return hash_str(cache_str)


P = ParamSpec("P")
T = TypeVar("T")


def cache_function_outputs(
    cache_dir: Path,
    function_args_to_cache: list[str] | Literal["all"] = "all",
    function_args_to_cache_id: Callable[
        [dict[str, Any]], str
    ] = default_function_args_to_cache_id,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    if isinstance(function_args_to_cache, list) and len(function_args_to_cache) == 0:
        raise ValueError("function_args_to_cache must be a non-empty list or 'all'")

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            args_and_kwargs_dict = get_args_and_kwargs_dict(func, args, kwargs)

            if isinstance(function_args_to_cache, list):
                args_and_kwargs_dict = {
                    k: v
                    for k, v in args_and_kwargs_dict.items()
                    if k in function_args_to_cache
                }

            cache_id = function_args_to_cache_id(args_and_kwargs_dict)

            save_file = cache_dir / func.__name__ / f"{cache_id}.pkl"

            if save_file.exists():
                print(f"Loading {func.__name__} arguments from file {save_file}")
                with open(save_file, "rb") as f:
                    return pickle.load(f)
            else:
                output = func(*args, **kwargs)
                save_file.parent.mkdir(parents=True, exist_ok=True)
                print(f"Cached {func.__name__} to file {save_file}")
                with open(save_file, "wb") as f:
                    pickle.dump(output, f)
                return output

        return wrapper  # type: ignore

    return decorator


def get_args_and_kwargs_dict(
    function: Callable[..., Any], args: tuple[Any], kwargs: dict[str, Any]
) -> dict[str, Any]:
    sig = inspect.signature(function)
    params = list(sig.parameters.keys())
    args_as_kwargs: dict[str, Any] = {}
    for i, arg in enumerate(args):
        # If we have more args than named parameters, it means the function uses *args
        # Or there's an error in how the function is being called
        if i < len(params):
            param_name = params[i]
            # Don't override if the parameter is *args or **kwargs
            if param_name != "args" and param_name != "kwargs":
                args_as_kwargs[param_name] = arg
            else:
                args_as_kwargs[f"arg_{i}"] = arg
        else:
            # This would happen if the function is called with more positional args than it has parameters
            # This is only valid if the function has a *args parameter
            args_as_kwargs[f"arg_{i}"] = arg

    assert set(args_as_kwargs.keys()).isdisjoint(set(kwargs.keys())), (
        "The kwargs should not contain keys of the from arg_i"
    )
    return args_as_kwargs | kwargs



logger = logging.getLogger(__name__)


def get_data_collator_with_padding(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Constructs a custom version of the datacollator with padding, which only pads 'input_ids' and 'labels', and does normal collation on the rest"""

    def _collator(batch: list[dict[str, Any]]) -> dict[str, Any]:
        # Due to the complexities of collating we need to seperately handle collation of  tensos (input_ids and labels), collation of types which can be handled by default_collate, and collation of other types (which we do manually)

        original_parallelism = os.environ.get("TOKENIZERS_PARALLELISM", "")
        os.environ["TOKENIZERS_PARALLELISM"] = (
            "false"  # transformers don't like paralleism in a dtaloader worker, so we set it to false here
        )
        # If the entry doesn't have labels, we add them by shifting the input_ids to the right
        for item in batch:
            if "labels" not in item or ("labels" in item and item["labels"] is None):
                item["labels"] = item["input_ids"]

        # First, we pad the input_ids and nothing else.
        input_ids_to_pad = [
            {k: v for k, v in item.items() if k == "input_ids"} for item in batch
        ]
        padded_input_ids = tokenizer.pad(input_ids_to_pad)
        os.environ["TOKENIZERS_PARALLELISM"] = original_parallelism

        # Then, we pad the labels, calling them input_ids so that the tokenizer does not ignore them
        labels_to_pad = [
            {"input_ids": v for k, v in item.items() if k == "labels"} for item in batch
        ]
        padded_labels = tokenizer.pad(labels_to_pad)
        labels = padded_labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100  # type: ignore

        # We then manually collate inputs, avoiding the pytorch default_collate as we want None variables etc.
        inputs_collated = {}
        for item in batch:
            for k, v in item.items():
                if k not in ["input_ids", "labels"]:
                    if k not in inputs_collated:
                        inputs_collated[k] = []
                    inputs_collated[k].append(v)

        return (
            {"labels": labels} | inputs_collated | padded_input_ids  # type: ignore
        )

    return _collator


def tokenize(
    input: dict[str, str],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    add_eos_token: bool = True,
    mask_out_prompt: bool = True,
) -> dict[str, Any]:
    """Tokenize a prompt-completion pair for language model training.
    
    This function takes a dictionary containing 'prompt' and 'completion' strings,
    tokenizes them together, and prepares the input for training a language model.
    It can optionally add an EOS token and mask out the prompt tokens in the labels.
    
    Args:
        input: Dictionary containing 'prompt' and 'completion' string fields
        tokenizer: The tokenizer to use for converting text to token IDs
        add_eos_token: Whether to append an EOS token to the tokenized input
        mask_out_prompt: Whether to mask out prompt tokens in labels by setting them to -100
            (which are ignored in the loss calculation during training)
    
    Returns:
        A dictionary containing the original input plus:
        - 'input_ids': Tensor of token IDs for the full prompt + completion
        - 'labels': Tensor of token IDs with prompt tokens optionally masked out
    """
    assert "prompt" in input, "Input should have an prompt field"
    assert "completion" in input, "Input should have a completion field"

    full_input_tokenized: torch.Tensor = tokenizer(
        input["prompt"] + input["completion"],
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]  # type: ignore

    if add_eos_token:
        full_input_tokenized = torch.cat(
            [full_input_tokenized, torch.tensor([tokenizer.eos_token_id])]
        )

    labels = full_input_tokenized.clone()

    # find the first token where the prompt and the full input differ. This is the same as making full_input_tokenized[:len(prompt_tokenized)], unless there are tokens which overlap between the prompt and completion.
    prompt_tokenized: torch.Tensor = tokenizer(
        input["prompt"], padding=True, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]  # type: ignore

    shared_prefix_end = 0
    for i in range(len(full_input_tokenized)):
        if i >= len(prompt_tokenized) or full_input_tokenized[i] != prompt_tokenized[i]:
            break
        shared_prefix_end = i

    if mask_out_prompt:
        labels[: shared_prefix_end + 1] = -100

    new_entries = {
        "input_ids": full_input_tokenized.long(),
        "labels": labels.long(),
    }

    return input | new_entries
def get_hash_of_data_module() -> str:
    data_module_path = Path(__file__).parent
    hash_of_data_module = ""
    for python_file in data_module_path.glob("*.py"):
        hash_of_file = get_hash_of_file(python_file)
        hash_of_data_module += hash_of_file

    return hash_str(hash_of_data_module)[:8]


def get_hash_of_file(file: Path) -> str:
    return hash_str(file.read_text())[:8]


def get_arguments_as_string(frame: inspect.FrameInfo) -> str:
    # Use inspect to grab all argument names and values from the caller's frame
    assert frame is not None
    arg_info = inspect.getargvalues(frame)  # type: ignore
    arg_names = arg_info.args

    # Automatically include only simple (primitive) parameters in the name.
    # This avoids including complex objects like tokenizer, data_dir, etc.
    param_parts = []
    for name in sorted(arg_names):
        value = arg_info.locals[name]
        if isinstance(value, (int, float, str)):
            param_parts.append(f"{name}{value}")

    return "_".join(param_parts)


def pre_tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    add_eos_token: bool = True,
) -> Dataset:
    """Pre-tokenize an entire dataset to avoid tokenization during DataLoader operation"""
    # Set tokenizer parallelism for this operation
    original_parallelism = os.environ.get("TOKENIZERS_PARALLELISM", None)
    os.environ["TOKENIZERS_PARALLELISM"] = (
        "true"  # Enable parallelism for batch tokenization
    )

    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenize(x, tokenizer, add_eos_token),
        batched=False,
        desc="Pre-tokenizing dataset",
    )

    # Restore original setting
    if original_parallelism is not None:
        os.environ["TOKENIZERS_PARALLELISM"] = original_parallelism
    else:
        os.environ.pop("TOKENIZERS_PARALLELISM", None)

    return tokenized_dataset
