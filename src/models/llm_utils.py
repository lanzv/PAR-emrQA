"""
All functions extracted from the following repo:
https://github.com/teticio/llama-squad
"""

import torch
from transformers import DataCollatorForLanguageModeling

# based on https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat

from functools import partial
from threading import Thread
from types import SimpleNamespace
from typing import Dict, Iterator, Optional
import logging
import re
import json5
import torch
import yaml
from peft import PeftModel, LoraConfig
import peft.tuners.lora.layer as lora_layer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)
from tqdm import tqdm
from trl import SFTTrainer


def create_and_prepare_model(model_path):
    """
    Create and prepare the model for training and inference.

    Parameters
    ----------
    model_path: str
        Path to the pre-trained model.

    Returns
    -------
    tuple
        A tuple containing the model, PEFT configuration, and tokenizer.
    """
    compute_dtype = getattr(torch, "float16")
    model, tokenizer = get_model_and_tokenizer(
        model_name=model_path,
        quantize=True,
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    # check: https://github.com/huggingface/transformers/pull/24906
    model.config.pretraining_tp = 1

    model_modules = str(model.modules)
    pattern = r"\((\w+)\): Linear"
    linear_layer_names = re.findall(pattern, model_modules)

    # target all linear layers
    names = []
    for name in linear_layer_names:
        names.append(name)
    target_modules = list(set(names))

    peft_config = LoraConfig(
        target_modules=target_modules,
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    return model, peft_config, tokenizer

# from https://github.com/artidoro/qlora/blob/7f4e95a68dc076bea9b3a413d2b512eca6d004e5/qlora.py#L425
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
):
    """
    Resize the tokenizer and model embeddings to accommodate new special tokens.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.

    Parameters
    ----------
    special_tokens_dict: Dict
        Dictionary of special tokens to be added.
    tokenizer: AutoTokenizer
        Tokenizer to be resized.
    model: AutoModelForCausalLM
        Model whose embeddings will be resized.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg


def get_model_and_tokenizer(
    model_name: str,
    tokenizer_name: Optional[str] = None,
    quantize: bool = False,
    load_in_4bit: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype: torch.dtype = torch.float16,
    bnb_4bit_use_double_quant: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:

    """
    Load the model and tokenizer with optional quantization settings.

    Parameters
    ----------
    model_name: str
        Name of the model to load.
    tokenizer_name: Optional[str]
        Name of the tokenizer to load.
    quantize: bool
        Whether to apply quantization to the model.
    load_in_4bit: bool
        Whether to load the model in 4-bit precision.
    bnb_4bit_quant_type: str
        Type of 4-bit quantization to use.
    bnb_4bit_compute_dtype: torch.dtype
        Data type for 4-bit quantization computation.
    bnb_4bit_use_double_quant: bool
        Whether to use double quantization.

    Returns
    -------
    tuple
        A tuple containing the model and tokenizer.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        use_auth_token=True,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name if tokenizer_name else model_name, trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def extract_answer(text):
    """
    Extract the answer from the generated text.

    Parameters
    ----------
    text: str
        The text containing the answer in JSON format.

    Returns
    -------
    answer: str or None
        The extracted answer or None if extraction fails.
    """
    text = text[text.find("{") : text.rfind("}") + 1]
    try:
        # JSON5 is a little less picky than JSON
        answer = json5.loads(text)["answer"]
    except:
        answer = None
    return answer


class StopAfterToken(StoppingCriteria):
    """
    Custom stopping criteria to stop generation after a specific token.
    """
    def __init__(self, token_id: int):
        self.token_id = token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        return input_ids[0][-1] == self.token_id

def get_answer(messages, pipeline, num_beams=None):
    """
    Generate an answer based on the provided messages and pipeline.

    Parameters
    ----------
    messages: dict
        Dictionary of messages for the dialogue.
    pipeline: transformers.pipeline
        The text generation pipeline.
    num_beams: int
        Number of beams for beam search.

    Returns
    -------
    tuple
        The generated answer and the corresponding score.
    """
    assistant_messages = [
        message
        for message in range(len(messages))
        if messages[message]["role"] == "assistant"
    ]
    for _, assistant_message in enumerate(assistant_messages):
        prompt = pipeline.tokenizer.apply_chat_template(
            messages[:assistant_message]
            + [{"role": "assistant", "content": "PLACEHOLDER"}],
            tokenize=False,
        )
        prompt = prompt[: prompt.rfind("PLACEHOLDER")] + f"\n```json"
        stopping_criteria = StoppingCriteriaList(
            [
                StopAfterToken(
                    pipeline.tokenizer.vocab.get(
                        "}Ċ", pipeline.tokenizer.vocab["}"]
                    )
                )
            ]
        )
        
    tokenized = pipeline.tokenizer([prompt], return_tensors="pt")
    outputs = pipeline.model.generate(
        input_ids = tokenized.input_ids.to(pipeline.model.device),
        attention_mask = tokenized.attention_mask.to(pipeline.model.device),
        output_scores=False, output_logits=True, return_dict_in_generate=True,
        do_sample=False,
        num_beams=num_beams,
        num_return_sequences=1,
        max_new_tokens=512,
        temperature=None,
        top_p=None,
        stopping_criteria=stopping_criteria,
    )
    responses = pipeline.tokenizer.batch_decode(outputs.sequences)
    answers = []
    for res in responses:
        answers.append(extract_answer(res[len(prompt) :].strip()))

    probs = torch.nn.functional.softmax(torch.stack(outputs.logits), dim=-1)
    max_probs, _ = probs.max(dim=-1)
    scores = max_probs.mean(dim=0).tolist()

    return answers[0], scores[0]


class SquadSFTTrainer(SFTTrainer):
    """
    Custom trainer for fine-tuning models on SQuAD-like datasets.
    """
    
    def __init__(self, answer_start_tokens, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.answer_start_tokens = answer_start_tokens
        self.stopping_criteria = StoppingCriteriaList(
            [StopAfterToken(self.tokenizer.vocab.get("}Ċ", self.tokenizer.vocab["}"]))]
        )

    def evaluate(self, **kwargs):
        """
        Evaluate the model on the evaluation dataset, regarding the EM metric.

        Returns
        -------
        dict
            dictionary of evaluation metrics, eval_exact_match, eval_has_answer_correct, eval_no_answer_correct
        """
        def cast_hook(dtype, module, input):
            input = input[0].to(dtype)
            return (input,)

        exact_match = 0
        has_answer = 0
        has_answer_correct = 0
        no_answer_correct = 0
        for item in tqdm(self.eval_dataset, desc="Evaluating"):
            input_ids = torch.tensor(item["input_ids"]).to(self.model.device)
            attention_mask = torch.tensor(item["attention_mask"]).to(self.model.device)
            answer_start_tokens = self.answer_start_tokens.to(self.model.device)
            window = input_ids.unfold(0, answer_start_tokens.shape[0], 1)
            answer_start = (window == answer_start_tokens).all(dim=1).nonzero()[-1, 0]
            answers = extract_answer(
                self.tokenizer.decode(
                    item["input_ids"][answer_start:], skip_special_tokens=True
                )
            )

            # NFI why this is necessary here but not during training
            hook_handles = []
            for _, module in self.model.named_modules():
                if isinstance(module, lora_layer.Linear):
                    hook_handles.append(
                        module.register_forward_pre_hook(
                            partial(cast_hook, self.model.dtype)
                        )
                    )

            output = self.model.generate(
                input_ids=input_ids[
                    : answer_start + len(self.answer_start_tokens)
                ].unsqueeze(0),
                attention_mask=attention_mask[
                    : answer_start + len(self.answer_start_tokens)
                ].unsqueeze(0),
                do_sample=False,
                num_return_sequences=1,
                max_new_tokens=512,
                temperature=None,
                top_p=None,
                stopping_criteria=self.stopping_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            for hook_handle in hook_handles:
                hook_handle.remove()

            model_answer = extract_answer(
                self.tokenizer.decode(
                    output[0, answer_start:], skip_special_tokens=True
                )
            )
            correct = 1 if model_answer is not None and model_answer in answers else 0
            exact_match += correct
            if answers != ["?"]:
                has_answer += 1
                has_answer_correct += correct
            else:
                no_answer_correct += correct

        exact_match = exact_match/len(self.eval_dataset) if len(self.eval_dataset) != 0 else 0
        has_answer_correct = has_answer_correct/has_answer if has_answer != 0 else 0
        no_answer_correct = no_answer_correct/(len(self.eval_dataset) - has_answer) if len(self.eval_dataset) - has_answer != 0 else 0
        metrics = {
            "eval_exact_match": exact_match,
            "eval_has_answer_correct": has_answer_correct,
            "eval_no_answer_correct": no_answer_correct,
        }
        logging.info(metrics)
        self.log(metrics)
        return metrics

class SquadDataCollator(DataCollatorForLanguageModeling):
    """
    Custom data collator for SQuAD-like datasets.
    """

    def __init__(self, answer_start_tokens: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.answer_start_tokens = answer_start_tokens

    def __call__(self, examples):
        """
        Process the examples to create a batch for training.

        Parameters
        ----------
        examples: list
            List of input examples.

        Returns
        -------
        batch: dict
            Batch of processed examples.
        """
        batch = super().__call__(examples)

        # Only apply cross entropy loss to the answer part of the labels
        for idx, label in enumerate(batch["labels"]):
            answer_end = torch.where(label == -100)[0]
            window = label.unfold(0, self.answer_start_tokens.shape[0], 1)
            if len((window == self.answer_start_tokens).all(dim=1).nonzero()) == 0:
                answer_start = 0
            else:
                answer_start = (
                    (window == self.answer_start_tokens).all(dim=1).nonzero()[-1, 0]
                )
            label[:answer_start] = -100
            if len(answer_end) > 0:
                label[answer_end[0]] = self.tokenizer.eos_token_id
            batch["labels"][idx] = label

        return batch