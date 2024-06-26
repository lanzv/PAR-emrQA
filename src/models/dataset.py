import uuid
import random
from datasets import Dataset
import json
from dataclasses import dataclass, field
from textwrap import dedent
from types import SimpleNamespace
from typing import Optional

import yaml
from datasets import DatasetDict, load_dataset


def emrqa2qa_dataset(dataset, seed=54, balanced=True):
    data = {
        'id': [],
        'title': [],
        'context': [],
        'question': [],
        'answers': []
    }
    random.seed(seed)
    always_one_paragraph = True
    for report in dataset["data"]:
        if len(report["paragraphs"]) > 1:
            always_one_paragraph = False
        for paragraph_id, paragraph in enumerate(report['paragraphs']):
            for qa in paragraph['qas']:
                if len(qa["answers"]) > 0:
                    data['id'].append("{}_{}".format(qa['id'], paragraph_id))
                    data['title'].append(report['title'])
                    data['context'].append(paragraph['context'])
                    data['question'].append(qa['question'])
                    texts = [ans['text'] for ans in qa['answers']]
                    starts = [ans['answer_start'] for ans in qa['answers']]
                    data['answers'].append({'text': texts, 'answer_start': starts})
    if always_one_paragraph:
        balanced = False
    if balanced:
        for report in dataset["data"]:
            # precompute negative paragraphs by used questions
            correct_paragraphs_by_questions = {}
            for par_id, paragraph in enumerate(report["paragraphs"]):
                for qa in paragraph["qas"]:
                    if not qa["id"] in correct_paragraphs_by_questions:
                        correct_paragraphs_by_questions[qa["id"]] = set()
                    correct_paragraphs_by_questions[qa["id"]].add(par_id)
            paragraphs_by_questions = {}
            for qa_id in correct_paragraphs_by_questions:
                paragraphs_by_questions[qa_id] = []
                for par_id, paragraph in enumerate(report["paragraphs"]):
                    if not par_id in correct_paragraphs_by_questions[qa_id]:
                        paragraphs_by_questions[qa_id].append(paragraph["context"])
            # create negative samples
            for paragraph in report["paragraphs"]:
                for qa in paragraph["qas"]:
                    if len(paragraphs_by_questions[qa["id"]]) > 0:
                        data['id'].append(str(uuid.uuid1().hex)) # set new id for negative sample
                        data['title'].append(report['title'])                            
                        data['context'].append(random.choice(paragraphs_by_questions[qa["id"]]))
                        data['question'].append(qa['question'])
                        data['answers'].append({'text': [], 'answer_start': []})

    dataset = Dataset.from_dict(data)
    dataset = dataset.shuffle(seed=seed)
    return dataset


def emrqa2prqa_dataset(dataset, seed=54):
    report_boundaries = [0]
    question_ids = []
    gold_paragraphs = [] # list of sets of gold paragraphs

    data = {
        'id': [],
        'title': [],
        'context': [],
        'question': [],
        'answers': []
    }
    random.seed(seed)
    for report in dataset["data"]:
        report_qa_ids = set()
        map_id2question = {}
        # precompute all paragraphs and their question ids
        ids_by_paragraphs = {}
        for par_id, paragraph in enumerate(report["paragraphs"]):
            ids_by_paragraphs[par_id] = set()
            for qa in paragraph["qas"]:
                map_id2question[qa["id"]] = qa["question"]
                ids_by_paragraphs[par_id].add(qa["id"])
                report_qa_ids.add(qa["id"])

        # create the dataset
        for qa_id in report_qa_ids:
            gold_report_paragraphs = set()
            question_ids.append(qa_id)
            for par_id, paragraph in enumerate(report["paragraphs"]):
                data['id'].append("{}_{}".format(qa_id, par_id))
                data['title'].append(report['title'])
                data['context'].append(paragraph['context'])
                data['question'].append(map_id2question[qa_id])
                if qa_id in ids_by_paragraphs[par_id]:
                    gold_report_paragraphs.add(par_id)
                    answer_found = False
                    for qa in paragraph["qas"]:
                        if qa["id"] == qa_id:
                            texts = [ans['text'] for ans in qa['answers']]
                            starts = [ans['answer_start'] for ans in qa['answers']]
                            answer_found = True
                            break
                    assert answer_found
                else:
                    texts = []
                    starts = []
                data['answers'].append({'text': texts, 'answer_start': starts})

            report_boundaries.append(report_boundaries[-1] + len(report["paragraphs"]))
            gold_paragraphs.append(gold_report_paragraphs)


    test_data = Dataset.from_dict(data)
    return {"test_data": test_data, "report_boundaries": report_boundaries, "question_ids": question_ids, "gold_paragraphs": gold_paragraphs}


def get_dataset_bert_format(train, dev, test):
    train_dataset = emrqa2qa_dataset(train)
    dev_dataset = emrqa2qa_dataset(dev)
    test_prqa_dataset = emrqa2prqa_dataset(test)
    return train_dataset, dev_dataset, test_prqa_dataset


def get_dataset_llm_format(train, dev, test):
    train_pars = emrqa2qa_dataset(train, balanced=False)
    dev_pars = emrqa2qa_dataset(dev, balanced=False)
    test_pars = emrqa2qa_dataset(test, balanced=False)

    llm_train = convert_format_bert2llm(train_pars)
    llm_dev = convert_format_bert2llm(dev_pars).train_test_split(test_size=0.05)["test"]
    llm_test = convert_format_bert2llm(test_pars)

    return llm_train, llm_dev, llm_test # llm_test is not prqa dataset


def convert_format_bert2llm(bert_dataset):
    return bert_dataset.map(get_single_turn_prompt_and_response)


@dataclass
class ScriptArguments:
    prompt: Optional[str] = field(
        default="single_turn",
        metadata={"help": "single_turn, multi_turn"},
    )
    validation_ratio: Optional[float] = field(
        default=0.005,
        metadata={"help": "Validation ratio"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "Seed for the random number generator"},
    )


def get_single_turn_prompt_and_response(item, all_answers=False):
    context = item["context"]
    question = item["question"]
    answers = item["answers"]["text"]
    if len(answers) == 0:
        answers = ["?"]
    if not all_answers:
        answers = answers[0]
    answers = json.dumps(answers)
    system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    return {
        "messages": [
            #{"role": "user", "content": system_prompt},
            #{"role": "assistant", "content": "I will."},
            {
                "role": "user",
                "content": dedent(
                    f"""\
                    Extract from the following context the minimal span word for word that best answers the question. Think step by step and explain your reasoning. Then give the answer in JSON format as follows:
                    ```json
                    {{
                      "answer": ...
                    }}
                    ```
                    If the answer is not in the context, the answer should be "?".
                    Context: {context}
                    Question: {question}"""
                ),
            },
            {
                "role": "assistant",
                "content": dedent(
                    f"""\
                    
                    ```json
                    {{
                      "answer": {answers}
                    }}
                    ```"""
                ),
            },
        ]
    }

