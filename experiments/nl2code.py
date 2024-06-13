import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, cast

from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl
from tqdm.auto import tqdm
from transformers import HfArgumentParser

from src.wrapper import GenerationConfig, get_model_context
from src.prompts import NL2CODE_PROMPT
from src.utils import chunked


class NL2CodeProblem(TypedDict):
    id: str
    instruction: str
    response_prefix: str


MBPP_INSTRUCTION = """{nl_description} Your code should satisfy the following assertion:
```python
{assertions}
```
Enclose your solution in ```python and ```"""


def get_mbpp_raw_problems() -> list[dict]:
    problems = get_mbpp_plus()
    return list(problems.values())


def get_humaneval_raw_problems() -> list[dict]:
    problems = get_human_eval_plus()
    return list(problems.values())


def map_mbpp_problem(p: dict) -> NL2CodeProblem:
    id = p["task_id"]
    prompt = p["prompt"]
    start_index = prompt.index('"""')
    end_index = prompt.rindex('"""')
    prompt = prompt[start_index + 3 : end_index]
    assert_index = prompt.index("assert")
    instruction = prompt[:assert_index].strip()
    if not instruction.endswith("."):
        instruction += "."
    assertion = prompt[assert_index:].strip()
    instruction = f"""{instruction} Your code should satisfy the following assertion:
```python
{assertion}
```"""
    response_prefix = f"""```python"""
    return NL2CodeProblem(
        id=str(id), instruction=instruction, response_prefix=response_prefix
    )


def map_humaneval_problem(p: dict) -> NL2CodeProblem:
    id = p["task_id"]
    prompt = p["prompt"]
    prompt = prompt.strip()
    instruction = f"""Write a solution to the following problem:
```python
{prompt}
```"""
    response_prefix = f"""```python
{prompt}"""
    return NL2CodeProblem(
        id=id, instruction=instruction, response_prefix=response_prefix
    )


@dataclass(frozen=True)
class Args:
    model_key: str
    dataset: Literal["humaneval", "mbpp"]
    save_path: str

    n_batches: int
    n_problems_per_batch: int
    n_samples_per_problem: int
    # prompted: bool

    model_name_or_path: str | None = None


def main():
    parser = HfArgumentParser((Args, GenerationConfig))
    args, generation_config = cast(
        tuple[Args, GenerationConfig],
        parser.parse_args_into_dataclasses(),
    )
    raw_problem_fn, map_problem_fn = (
        (get_humaneval_raw_problems, map_humaneval_problem)
        if args.dataset == "humaneval"
        else (get_mbpp_raw_problems, map_mbpp_problem)
    )
    raw_problems = raw_problem_fn()
    # map the raw problems to a dict {"<task_id>": <prompt>}"
    raw_problems_dict = {p["task_id"]: p["prompt"] for p in raw_problems}
    problems = list(map(map_problem_fn, raw_problems))

    state = get_model_context(args.model_key, args.model_name_or_path)

    problems_chunked = list(chunked(list(problems), args.n_problems_per_batch))
    iter = itertools.product(problems_chunked, range(args.n_batches))
    n_total = len(problems_chunked) * args.n_batches

    Path(args.save_path).write_text("")
    for problems, batch_idx in tqdm(iter, total=n_total):
        task_ids = [problem["id"] for problem in problems]
        prompts = [
            NL2CODE_PROMPT.format(
                instruction=problem["instruction"], response=problem["response_prefix"]
            )
            for problem in problems
        ]
        print("PROMPT")
        print(prompts[-1])
        all_prompts = prompts * args.n_samples_per_problem
        all_task_ids = task_ids * args.n_samples_per_problem
        response = state.complete(generation_config, all_prompts)
        completions = response.decoded_outputs
        assert len(problems) <= args.n_problems_per_batch
        assert len(completions) == len(problems) * args.n_samples_per_problem
        print("COMPLETION")
        print(completions[-1])
        samples = [
            dict(
                task_id=task_id,
                completion=completion[
                    : index
                    if (index := completion.find("```")) != -1
                    else len(completion)
                ],
            )
            for task_id, completion in zip(all_task_ids, completions)
        ]
        for sample in samples:
            sample["solution"] = raw_problems_dict[sample["task_id"]] + sample["completion"]
        write_jsonl(args.save_path, samples, append=True)


if __name__ == "__main__":
    main()
