import argparse
import logging
from src.paragraphizer import Paragraphizer
from src.eval import Evaluate
from src.models.bert import BERTWrapperPRQA
from src.models.llm import LLMWrapperPRQA
from src.models.dataset import emrqa2prqa_dataset, emrqa2qa_dataset, get_dataset_bert_format, get_dataset_llm_format
import json
import random
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser()
# dataset paths
parser.add_argument('--train_path', type=str, default='./data/relations-train.json', help="path to the preprocessed training subset")
parser.add_argument('--dev_path', type=str, default='./data/relations-dev.json', help="path to the preprocessed dev subset")
parser.add_argument('--test_path', type=str, default='./data/relations-test.json', help="path to the preprocessed test subset")
parser.add_argument('--dataset_title', type=str, default='relations', help="the mode how should be subset paragraphize, one of the following: ['medication', 'relations', 'uniform']")
# model
parser.add_argument('--model_name', type=str, default='ClinicalBERT', help="the code of the model regarding the MODELS and PREPARE_DATASET dictionaries, pick one of the following: ['BERTbase', 'ClinicalBERT', 'MedCPT', 'BioMistral']")
parser.add_argument('--model_path', type=str, default='../models/Bio_ClinicalBERT', help="path to the model's (/tokenizer) checkpoint, you can use the huggingface model path as well if it is not downloaded locally")
# paragraphizing
parser.add_argument('--ft', type=int, default=165, help="used when --dataset_title is 'medication' or 'relations', frequency threshold, what is the minimum occurences of headings that would be considered as paragraph separator -> the higher --ft, the longer paragraphs on average")
parser.add_argument('--target_average', type=int, default=500, help="used when --dataset_title is 'uniform', target average paragraph length")
parser.add_argument('--epochs', type=int, default=3, help="number of training epochs of the models")
# random
parser.add_argument('--seed', type=int, help='random seed', default=2)






MODELS = {
    "BERTbase": lambda model_path: BERTWrapperPRQA(model_path),
    "ClinicalBERT": lambda model_path: BERTWrapperPRQA(model_path),
    "BioLORD": lambda model_path: BERTWrapperPRQA(model_path),
    "MedCPT": lambda model_path: BERTWrapperPRQA(model_path),
    "BioMistral": lambda model_path: LLMWrapperPRQA(model_path),
}


PREPARE_DATASET = {
    "BERTbase": lambda train_pars, dev_pars, test_pars: get_dataset_bert_format(train_pars, dev_pars, test_pars),
    "ClinicalBERT": lambda train_pars, dev_pars, test_pars: get_dataset_bert_format(train_pars, dev_pars, test_pars),
    "BioLORD": lambda train_pars, dev_pars, test_pars: get_dataset_bert_format(train_pars, dev_pars, test_pars),
    "MedCPT": lambda train_pars, dev_pars, test_pars: get_dataset_bert_format(train_pars, dev_pars, test_pars),
    "BioMistral": lambda train_pars, dev_pars, test_pars: get_dataset_llm_format(train_pars, dev_pars, test_pars),
}


def main(args):
    if not args.model_name in MODELS:
        logging.error("The model {} is not supported".format(args.model_name))
        return 
    
    # load splited data
    with open(args.train_path, 'r') as f:
        train = json.load(f)
    with open(args.dev_path, 'r') as f:
        dev = json.load(f)
    with open(args.test_path, 'r') as f:
        test = json.load(f)
    scores = {}

    logging.info("------------- Experiment: model {}, frequency threshold {} ---------------".format(args.model_name, args.ft))
    # prepare data
    train_pars, train_topics = Paragraphizer.paragraphize(data = train, title=args.dataset_title, frequency_threshold = args.ft, target_average=args.target_average)
    dev_pars, _ = Paragraphizer.paragraphize(data = dev, title=args.dataset_title, frequency_threshold = args.ft, topics=train_topics, target_average=args.target_average)
    test_pars, _  = Paragraphizer.paragraphize(data = test, title=args.dataset_title, frequency_threshold = args.ft, topics=train_topics, target_average=args.target_average)
    train_dataset, dev_dataset, test_prqa_dataset = PREPARE_DATASET[args.model_name](train_pars, dev_pars, test_pars)
    logging.info("datasets are converted to Datset format")
    # train model
    model = MODELS[args.model_name](args.model_path)
    model.train(train_dataset, dev_dataset, epochs=args.epochs, disable_tqdm=True)
    qa_predictions, pr_predictions, prqa_predictions = model.predict(test_prqa_dataset, disable_tqdm=True)
    # evaluate
    qa_scores = Evaluate.question_answering(test, qa_predictions)
    logging.info("QA scores: {}".format(qa_scores))
    pr_scores = Evaluate.paragraph_retrieval(test_pars, pr_predictions) # eval PR predictions on the Paragraphized dataset
    logging.info("PR scores: {}".format(pr_scores))
    prqa_scores = Evaluate.question_answering(test, prqa_predictions)
    logging.info("PRQA scores: {}".format(prqa_scores))

    scores[args.ft] = {}
    scores[args.ft]["QA"] = qa_scores
    scores[args.ft]["PR"] = pr_scores
    scores[args.ft]["PRQA"] = prqa_scores

    scores = json.dumps(scores, indent = 4) 
    print(scores)



if __name__ == '__main__':
    args = parser.parse_args()
    random.seed(args.seed)
    main(args)