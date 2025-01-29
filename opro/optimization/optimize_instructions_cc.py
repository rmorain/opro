# Copyright 2023 The OPRO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""The .py file for prompt optimization.

Usage:

Step 1: edit the starting instructions by modifying `initial_instructions`

Step 2: edit the training ratio by modifying `train_ratio`

Step 3: check if the model configs (like batch size) are the same as the actual serving configs

Step 4: run

```
python optimize_instructions.py \
    --optimizer="gpt-3.5-turbo" --scorer="text-bison" \
    --instruction_pos="A_begin" --dataset="gsm8k" --task="train"
```

The outputs will then be written to `outputs/optimization-results/` in the opro folder.

Notes:

1. One or more API keys may need to be provided:
- When using a Google-Cloud-served model (like text-bison at https://developers.generativeai.google/tutorials/text_quickstart), add `--palm_api_key=<your_key>`
- When using an OpenAI model, add `--openai_api_key=”<your_key>”`

2. The initial instructions should be provided in the "initial_instructions"
variable.
"""

import datetime
import functools
import os
import sys

from transformers import pipeline

OPRO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, OPRO_ROOT_PATH)

import openai
import pandas as pd
from absl import app, flags

from opro import prompt_utils
from opro.optimization import opt_utils_cc

ROOT_DATA_FOLDER_PATH = os.path.join(OPRO_ROOT_PATH, "data")

_OPENAI_API_KEY = flags.DEFINE_string("openai_api_key", "", "The OpenAI API key.")

_SCORER = flags.DEFINE_string("scorer", "llama", "The name of the scorer LLM.")


_OPTIMIZER = flags.DEFINE_string("optimizer", "llama", "The name of the optimizer LLM.")

_INSTRUCTION_POS = flags.DEFINE_string(
    "instruction_pos",
    "A_begin",
    "The position of the instruction to search for.",
)

_META_PROMPT_TYPE = flags.DEFINE_string(
    "meta_prompt_type",
    "both_instructions_and_exemplars",
    "The type of meta-prompt: whether to have both previous instructions and"
    " dataset exemplars (often for fine-tuned optimizers), or to have only"
    " previous instructions (often for pre-trained optimizers).",
)


def main(_):
    openai_api_key = _OPENAI_API_KEY.value
    scorer_llm_name = _SCORER.value
    optimizer_llm_name = _OPTIMIZER.value
    meta_prompt_type = _META_PROMPT_TYPE.value

    assert scorer_llm_name in {"gpt-3.5-turbo", "gpt-4", "llama"}
    assert optimizer_llm_name in {
        "gpt-3.5-turbo",
        "gpt-4",
        "llama",
    }
    assert meta_prompt_type in {
        "both_instructions_and_exemplars",
        "instructions_only",
    }

    instruction_pos = _INSTRUCTION_POS.value
    assert instruction_pos in {
        "before_Q",
        "Q_begin",
        "Q_end",
        "A_begin",
    }, (
        "The instruction position should be either before the question, or at the"
        " beginning of the question, at the end of the question, or at the"
        " beginning of the answer."
    )
    print(
        f"scorer: {scorer_llm_name}, optimizer: {optimizer_llm_name}, dataset:"
        f"instruction_pos: {instruction_pos}"
    )

    # make sure the scorer and optimizer models are callable
    if scorer_llm_name in {"gpt-3.5-turbo", "gpt-4"}:
        assert openai_api_key, "The OpenAI API key must be provided."
        openai.api_key = openai_api_key
    else:
        assert scorer_llm_name == "llama"

    if optimizer_llm_name in {"gpt-3.5-turbo", "gpt-4"}:
        assert openai_api_key, "The OpenAI API key must be provided."
        openai.api_key = openai_api_key
    else:
        assert optimizer_llm_name == "llama"

    # =================== create the result directory ==========================
    datetime_str = (
        str(datetime.datetime.now().replace(microsecond=0))
        .replace(" ", "-")
        .replace(":", "-")
    )

    save_folder = os.path.join(
        OPRO_ROOT_PATH,
        "outputs",
        "optimization-results",
        f"s-{scorer_llm_name}-o-{optimizer_llm_name}-{datetime_str}/",
    )
    result_by_instruction_folder = os.path.join(save_folder, "result_by_instruction")
    os.makedirs(result_by_instruction_folder)
    print(f"result directory:\n{save_folder}")

    # ====================== scorer model configs ==============================
    # difference between num_decodes and batch_size:
    # - num_decodes: how many outputs we actually want for each input
    # - batch_size: the batch size in model serving, should equal to that in
    # model serving config

    if scorer_llm_name == "llama":
        # when prompting llama locally
        scorer_llama_max_decode_steps = 1024
        scorer_llama_temperature = 0.0

        scorer_llama_dict = dict()
        scorer_llama_dict["max_decode_steps"] = scorer_llama_max_decode_steps
        scorer_llama_dict["temperature"] = scorer_llama_temperature
        scorer_llama_dict["num_decodes"] = 1
        scorer_llama_dict["batch_size"] = 1
        scorer_llama_dict["num_servers"] = 1

        scorer_llm_dict = {
            "model_type": scorer_llm_name.lower(),
        }
        scorer_llm_dict.update(scorer_llama_dict)
        call_scorer_server_func = functools.partial(
            prompt_utils.call_llama_server_func,
            model=scorer_llm_name.lower(),
            max_decode_steps=scorer_llama_max_decode_steps,
            temperature=scorer_llama_temperature,
        )
    else:
        assert scorer_llm_name.lower() in {"gpt-3.5-turbo", "gpt-4"}
        scorer_gpt_max_decode_steps = 1024
        scorer_gpt_temperature = 0.0

        scorer_llama_dict = dict()
        scorer_llama_dict["max_decode_steps"] = scorer_gpt_max_decode_steps
        scorer_llama_dict["temperature"] = scorer_gpt_temperature
        scorer_llama_dict["num_decodes"] = 1
        scorer_llama_dict["batch_size"] = 1
        scorer_llama_dict["num_servers"] = 1

        scorer_llm_dict = {
            "model_type": scorer_llm_name.lower(),
        }
        scorer_llm_dict.update(scorer_llama_dict)
        call_scorer_server_func = functools.partial(
            prompt_utils.call_openai_server_func,
            model=scorer_llm_name.lower(),
            max_decode_steps=scorer_gpt_max_decode_steps,
            temperature=scorer_gpt_temperature,
        )

    # ====================== optimizer model configs ============================
    if optimizer_llm_name.lower() == "llama":
        optimizer_llama_max_decode_steps = 512
        optimizer_llama_temperature = 1.0

        optimizer_llm_dict = dict()
        optimizer_llm_dict["max_decode_steps"] = optimizer_llama_max_decode_steps
        optimizer_llm_dict["temperature"] = optimizer_llama_temperature
        optimizer_llm_dict["batch_size"] = 1
        optimizer_llm_dict["num_decodes"] = 1
        pipe = pipeline(
            "text-generation",
            model="meta-llama/Llama-3.1-8B-Instruct",
            max_new_tokens=optimizer_llama_max_decode_steps,
            temperature=optimizer_llama_temperature,
            device="cuda",
        )
        call_optimizer_server_func = functools.partial(
            prompt_utils.call_llama_server_func,
            pipeline=pipe,
        )
    else:
        assert optimizer_llm_name in {"gpt-3.5-turbo", "gpt-4"}
        optimizer_gpt_max_decode_steps = 512
        optimizer_gpt_temperature = 1.0

        optimizer_llm_dict = dict()
        optimizer_llm_dict["max_decode_steps"] = optimizer_gpt_max_decode_steps
        optimizer_llm_dict["temperature"] = optimizer_gpt_temperature
        optimizer_llm_dict["batch_size"] = 1
        optimizer_llm_dict["num_decodes"] = 1
        call_optimizer_server_func = functools.partial(
            prompt_utils.call_openai_server_func,
            model=optimizer_llm_name,
            max_decode_steps=optimizer_gpt_max_decode_steps,
            temperature=optimizer_gpt_temperature,
        )

    # ====================== try calling the servers ============================
    print("\n======== testing the scorer and optimizer servers ===========")
    scorer_test_output = call_scorer_server_func(
        "Does the sun rise from the north? Just answer yes or no."
    )
    print(f"number of scorer output decodes: {len(scorer_test_output)}")
    print(f"scorer test output: {scorer_test_output}")
    optimizer_test_output = call_optimizer_server_func(
        "Does the sun rise from the north? Just answer yes or no.",
        temperature=1.0,
    )
    print(f"number of optimizer output decodes: {len(optimizer_test_output)}")
    print(f"optimizer test output: {optimizer_test_output}")
    print("Finished testing the servers.")

    # ========== set other optimization experiment hyperparameters ==============
    if scorer_llm_name == "llama":
        old_instruction_score_threshold = 0.3
    else:
        assert scorer_llm_name in {"gpt-3.5-turbo", "gpt-4"}
        old_instruction_score_threshold = 0.3

    if scorer_llm_name == "llama":
        extract_final_answer_by_prompting_again = False
        include_qa = False
        evaluate_in_parallel = False
    else:
        assert scorer_llm_name in {"gpt-3.5-turbo", "gpt-4"}
        extract_final_answer_by_prompting_again = False
        include_qa = False
        evaluate_in_parallel = False

    optimizer_llm_temperature = optimizer_llm_dict["temperature"]

    num_few_shot_questions_for_instruction_refinement = 3

    # To change the number of generated instructions in each step, one should
    # edit the value of the variable below, instead of editing the number of
    # decodes in model parameters, because those values are limited by model
    # serving configs.
    num_generated_instructions_in_each_step = 8
    num_search_steps = 200

    initial_instructions = [  # TODO: What should go here?
        # "Let's solve the problem.",
        "",
        # "The answer is",
    ]
    few_shot_qa_pairs = False
    # one of {'accumulative_most_frequent', 'current_most_frequent', 'random',
    # 'constant'}
    few_shot_selection_criteria = "random"
    # whether to evaluate generated instructions on the exemplars in meta-prompt
    evaluate_generated_ins_on_few_shot = False
    # whether to evaluate old instructions on the exemplars in the meta-prompt
    evaluate_old_ins_on_few_shot = False
    # every this number of steps, compute the accuracies of current-step
    # instructions on the validation set
    eval_interval = 1

    max_num_instructions = (
        20  # the maximum number of instructions and scores in the meta-prompt
    )
    # The number of buckets when converting scores to integers in the meta-prompt.
    num_score_buckets = 100
    # whether to put old instructions and scores to before exemplars in
    # the meta-prompt
    meta_prompt_instructions_before_exemplars = True

    # ===================== run prompt optimization ======================

    assert few_shot_selection_criteria in {
        "accumulative_most_frequent",
        "current_most_frequent",
        "random",
        "constant",
    }
    evolution_kwargs = {
        "num_search_steps": num_search_steps,
        "old_instruction_score_threshold": old_instruction_score_threshold,
        "scorer_llm_dict": scorer_llm_dict,
        "optimizer_llm_dict": optimizer_llm_dict,
        "extract_final_answer_by_prompting_again": (
            extract_final_answer_by_prompting_again
        ),
        "include_qa": include_qa,
        "evaluate_in_parallel": evaluate_in_parallel,
        "optimizer_llm_temperature": optimizer_llm_temperature,
        # "optimizer_llm_temperature_schedule": (
        #     optimizer_llm_temperature_schedule
        # ),
        # "optimizer_llm_temperature_end": optimizer_llm_temperature_end,
        "initial_instructions": initial_instructions,
        "call_scorer_server_func": call_scorer_server_func,
        "call_optimizer_server_func": call_optimizer_server_func,
        "instruction_pos": instruction_pos,
        "result_by_instruction_folder": result_by_instruction_folder,
        "few_shot_qa_pairs": few_shot_qa_pairs,
        "num_score_buckets": num_score_buckets,
        "max_num_instructions": max_num_instructions,
        "meta_prompt_type": meta_prompt_type,
        "meta_prompt_instructions_before_exemplars": (
            meta_prompt_instructions_before_exemplars
        ),
        "few_shot_selection_criteria": few_shot_selection_criteria,
        "optimizer_llm_name": optimizer_llm_name,
        "num_generated_instructions_in_each_step": (
            num_generated_instructions_in_each_step
        ),
        "evaluate_generated_ins_on_few_shot": evaluate_generated_ins_on_few_shot,
        "num_few_shot_questions_for_instruction_refinement": (
            num_few_shot_questions_for_instruction_refinement
        ),
        "evaluate_old_ins_on_few_shot": evaluate_old_ins_on_few_shot,
        "eval_interval": eval_interval,
        "save_folder": save_folder,
    }

    opt_utils.run_evolution(**evolution_kwargs)


if __name__ == "__main__":
    app.run(main)
