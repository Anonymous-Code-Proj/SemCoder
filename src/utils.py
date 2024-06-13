import functools
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TypeVar
from dataclasses import dataclass, field

N_CORES = 1 if (count := os.cpu_count()) is None or count == 0 else count // 2

@dataclass(frozen=True)
class Args:
    task: str = field(default="nl2code") # choices=["nl2code", "exec_reasoning", "exec_simu", "nl2code_exec", "nl2code_exec_refine"]
    datafile_paths: list[str] = field(default_factory=list)
    max_training_seq_length: int = field(default=1216)
    overwrite_cache: bool = field(default=False)
    pad_to_max_length: bool = field(default=False)
    eval_dataset_size: float = field(
        default=0.05, metadata={"help": "0--1 means ratio, >1 means number of examples"}
    )
    use_flash_attention: bool = field(default=False)




def read_jsonl(path: str | Path) -> list[Any]:
    """Read lines of JSON from a file (including '\n')."""
    with Path(path).open("r") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: str | Path, data: Sequence[Mapping]):
    # cannot use `dict` here as it is invariant
    with Path(path).open("w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


_T = TypeVar("_T")


def chunked(seq: Sequence[_T], n: int) -> Iterable[Sequence[_T]]:
    """Yield successive n-sized chunks from seq."""
    return (seq[i : i + n] for i in range(0, len(seq), n))


def retry_with_exponential_backoff(
    errors: tuple,
    initial_delay: float = 30,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 5,
):
    """Retry a function with exponential backoff."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Initialize variables
            num_retries = 0
            delay = initial_delay

            # Loop until a successful response or max_retries is hit or an exception is raised
            while True:
                try:
                    return func(*args, **kwargs)
                # Retry on specific errors
                except errors as e:
                    print(f"Error: {e}. Retrying in {delay} seconds...")
                    # Increment retries
                    num_retries += 1
                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        raise Exception(
                            f"Maximum number of retries ({max_retries}) exceeded."
                        )
                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())
                    # Sleep for the delay
                    time.sleep(delay)
                    # time.sleep(60)
                # Raise exceptions for any errors not specified
                except Exception as e:
                    raise e

        return wrapper

    return decorator
