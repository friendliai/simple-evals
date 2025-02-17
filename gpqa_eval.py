"""
GPQA: A Graduate-Level Google-Proof Q&A Benchmark
David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
https://arxiv.org/abs/2311.12022
"""

import random
import re
import pickle
import blobfile as bf
import pandas
import os
from . import common
from .common import ANSWER_PATTERN_MULTICHOICE, HTML_JINJA, format_multichoice_question
from .types import Eval, EvalResult, MessageList, SamplerBase, SingleEvalResult


class GPQAEval(Eval):
    def __init__(
        self,
        res_dir: str | None = None,
        n_repeats: int = 4,
        variant: str = "diamond",
        num_examples: int | None = None,  # restrict to a subset of the data for debugging
    ):
        self.res_dir = res_dir
        df = pandas.read_csv(
            f"{os.getcwd()}/simple-evals/dataset/gpqa_diamond.csv"
        )
        self.tmp_path = f"{res_dir}/gpqa_{n_repeats}_{num_examples}"
        tmp_example_path = f"{res_dir}/gpqa_examples_{n_repeats}_{num_examples}"
        if os.path.exists(tmp_example_path):
            with open(tmp_example_path, "rb") as f:
                examples = pickle.load(f)
        else:
            examples = [row.to_dict() for _, row in df.iterrows()]
            rng = random.Random(0)
            if num_examples:
                assert n_repeats == 1, "n_repeats only supported for num_examples = None"
                examples = rng.sample(examples, num_examples)
            examples = examples * n_repeats
            examples = [example | {"permutation": rng.sample(range(4), 4)} for example in examples]
            with open(tmp_example_path, "wb") as f:
                 pickle.dump(examples, f)
        self.examples = examples
        self.n_repeats = n_repeats

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(indexed_row: tuple):
            idx, row = indexed_row
            choices = [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]
            choices = [choices[i] for i in row["permutation"]]
            correct_index = choices.index(row["Correct Answer"])
            correct_answer = "ABCD"[correct_index]
            choices_dict = dict(
                A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=row["Question"]
            )
            prompt_messages = [
                sampler._pack_message(
                    content=format_multichoice_question(choices_dict), role="user"
                )
            ]
            response_text = sampler(prompt_messages)
            match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
            extracted_answer = match.group(1) if match else None
            score = 1.0 if extracted_answer == correct_answer else 0.0
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=correct_answer,
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(
                idx=idx, html=html, score=score, convo=convo, metrics={"chars": len(response_text)}
            )

        results = common.map_with_progress(fn, list(enumerate(self.examples)), cache_file=self.tmp_path)
        return common.aggregate_results(results)
