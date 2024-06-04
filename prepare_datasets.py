import argparse
import logging
import json
import random
import os
from src.paragraphizer import Paragraphizer
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='./data/data.json')
parser.add_argument('--target_dir', type=str, default='./data')
parser.add_argument('--train_ratio', type=float, default=0.7)
parser.add_argument('--dev_ratio', type=float, default=0.1)
parser.add_argument('--topics', metavar='N', type=str, nargs='+', default=["medication", "relations"])
parser.add_argument('--train_sample_ratios', metavar='N', type=float, nargs='+', default=[0.2, 0.05])
parser.add_argument('--seed', type=int, help='random seed', default=55)
parser.add_argument('--train_seed', type=int, help='random seed', default=2)



def sample_dataset(data, sample_ratio):
    """
    Authors: CliniRC
    """
    new_data = []
    total = 0
    sample = 0
    for paras in data['data']:
        new_paragraphs = []
        for para in paras['paragraphs']:
            new_qas = []
            context = para['context']
            norm_context = para['norm_context']
            qa_num = len(para['qas'])
            total += qa_num
            sample_num = int(qa_num * sample_ratio)
            sampled_list = [i for i in range(qa_num)]
            sampled_list = random.choices(sampled_list, k=sample_num)
            for qa_id in sampled_list:
                qa = para['qas'][qa_id]
                sample += 1
                new_qas.append(qa)
            new_para = {'context': context, 'norm_context': norm_context, 'qas': new_qas}
            new_paragraphs.append(new_para)
        new_data.append({'title': paras['title'], 'paragraphs': new_paragraphs})
    new_data_json = {'data': new_data, 'version': data['version']}
    return new_data_json


def main(args):
    random.seed(args.seed)
    with open(args.data_path, 'r') as f:
        dataset = json.load(f)
    train_originals = {}
    for title, train_sample_ratio in zip(args.topics, args.train_sample_ratios):
        curr_data = None
        # find the given sub dataset
        for data in dataset["data"]:
            if data["title"] == title:
                curr_data = data

        # filter and preprocess data
        curr_data = Paragraphizer.preprocess(curr_data)

        # split the dataset
        assert args.train_ratio + args.dev_ratio <= 1.0

        note_num = len(curr_data["data"])
        train_note_num, dev_note_num = int(args.train_ratio * note_num), int(args.dev_ratio * note_num)
        note_list = random.sample(range(note_num), note_num)

        train = {'data': [], 'version': 1.0}
        for i in range(train_note_num):
            train['data'].append(curr_data['data'][note_list[i]])
        dev = {'data': [], 'version': 1.0}
        for i in range(train_note_num, train_note_num + dev_note_num):
            dev['data'].append(curr_data['data'][note_list[i]])
        test = {'data': [], 'version': 1.0}
        for i in range(train_note_num + dev_note_num, note_num):
            test['data'].append(curr_data['data'][note_list[i]])
        train_originals[title] = train
        # sample dataset
        if train_sample_ratio < 1.0:
            new_train = sample_dataset(train, train_sample_ratio)

        # save splited dataset
        with open(os.path.join(args.target_dir, "{}-train.json".format(title)), "w") as jsonFile:
            json.dump(train, jsonFile)
        with open(os.path.join(args.target_dir, "{}-dev.json".format(title)), "w") as jsonFile:
            json.dump(dev, jsonFile)
        with open(os.path.join(args.target_dir, "{}-test.json".format(title)), "w") as jsonFile:
            json.dump(test, jsonFile)

    for title, train_sample_ratio in zip(args.topics, args.train_sample_ratios):
        random.seed(args.train_seed)
        # sample dataset
        if train_sample_ratio < 1.0:
            train = sample_dataset(train_originals[title], train_sample_ratio)
        with open(os.path.join(args.target_dir, "{}-train.json".format(title)), "w") as jsonFile:
            json.dump(train, jsonFile)

if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seed)
    main(args)