import json
import sys
import pandas as pd

from . import common
from .drop_eval import DropEval
from .gpqa_eval import GPQAEval
from .humaneval_eval import HumanEval
from .math_eval import MathEval
from .mgsm_eval import MGSMEval
from .mmlu_eval import MMLUEval
from .sampler.chat_completion_sampler import (
    ChatCompletionSampler,
    FriendliChatCompletionSampler
)

# from .sampler.claude_sampler import ClaudeCompletionSampler, CLAUDE_SYSTEM_MESSAGE_LMSYS


def main(exp_name, req_url, res_dir, model = None):
    debug = False
    samplers = {
        exp_name: FriendliChatCompletionSampler(
            base_url=req_url,
            model=model,
            max_tokens=2048,
            temperature=0.0,
        )
    }
    print(samplers)
    equality_checker = ChatCompletionSampler(model="gpt-4-turbo-preview")
    # ^^^ used for fuzzy matching, just for math

    def get_evals(eval_name):
        # Set num_examples = None to reproduce full evals
        match eval_name:
            case "mmlu":
                return MMLUEval(num_examples=1 if debug else 2500)
            case "math":
                return MathEval(
                    equality_checker=equality_checker, num_examples=5 if debug else 2500
                )
            case "gpqa":
                return GPQAEval(n_repeats=1 if debug else 10, num_examples=5 if debug else None)
            case "mgsm":
                return MGSMEval(num_examples_per_lang=10 if debug else 250)
            case "drop":
                return DropEval(num_examples=10 if debug else 2000, train_samples_per_prompt=3)
            case "humaneval":
                return HumanEval(num_examples=10 if debug else None)
            case _:
                raise Exception(f"Unrecoginized eval type: {eval_name}")

    evals = {
        eval_name: get_evals(eval_name) for eval_name in ["mmlu", "gpqa", "humaneval"]
    }
    debug_suffix = "_DEBUG" if debug else ""
    mergekey2resultpath = {}
    for sampler_name, sampler in samplers.items():
        for eval_name, eval_obj in evals.items():
            result = eval_obj(sampler)
            # ^^^ how to use a sampler
            file_stem = f"{eval_name}_{sampler_name}"
            report_filename = f"{res_dir}/{file_stem}{debug_suffix}.html"
            print(f"Writing report to {report_filename}")
            with open(report_filename, "w") as fh:
                fh.write(common.make_report(result))
            metrics = result.metrics | {"score": result.score}
            print(metrics)
            result_filename = f"{res_dir}/{file_stem}{debug_suffix}.json"
            with open(result_filename, "w") as f:
                f.write(json.dumps(metrics, indent=2))
            print(f"Writing results to {result_filename}")
            mergekey2resultpath[f"{file_stem}"] = result_filename
    merge_metrics = []
    for eval_sampler_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue
        result = result.get("f1_score", result.get("score", None))
        eval_name = eval_sampler_name[: eval_sampler_name.find("_")]
        sampler_name = eval_sampler_name[eval_sampler_name.find("_") + 1 :]
        merge_metrics.append(
            {"eval_name": eval_name, "sampler_name": sampler_name, "metric": result}
        )
    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
        index=["sampler_name"], columns="eval_name"
    )
    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())
    return merge_metrics


if __name__ == "__main__":
    args = sys.argv[1:]
    exp_name = args[0]
    req_url = args[1]
    res_out_dir = args[2]
    model = args[3] if len(args) == 4 else None
    main(exp_name, req_url, res_out_dir, model)
