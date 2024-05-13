import uuid
import random
from datasets import Dataset

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