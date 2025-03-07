# PAR-emrQA - Paragraph Segmentation of Two emrQA Subsets: Medication and Relations

This repository provides code for the Question-Answering and Paragraph Retrieval experiments in the BioNLP 2024 and Shared Tasks @ ACL 2024 paper: **Paragraph Retrieval for Enhanced Question Answering in Clinical Documents**

```bib
@inproceedings{lanz-pecina-2024-paragraph,
    title = "Paragraph Retrieval for Enhanced Question Answering in Clinical Documents",
    author = "Lanz, Vojtech and Pecina, Pavel",
    editor = "Demner-Fushman, Dina  and
      Ananiadou, Sophia  and
      Miwa, Makoto  and
      Roberts, Kirk  and
      Tsujii, Junichi",
    booktitle = "Proceedings of the 23rd Workshop on Biomedical Natural Language Processing",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.bionlp-1.48",
    pages = "580--590"
}
```
## Experiments

- **Paragraph Retrieval (PR)**: Given a question and n paragraphs (i.e., report segmented into n paragraphs) as input, the objective is to rank the paragraphs based on the confidence that they contain relevant information. The task is evaluated using precision at top 1 (P@1), precision at top 2 (P@2), and precision at top 3 (P@3) paragraphs. Ground truth relevant paragraphs are those containing an answer evidence to a given question defined in the emrQA dataset.

- **Oracle Paragraph-driven Question Answering (Oracle-QA)**: Given a question and an Oracle paragraph (guaranteed to contain the answer), the objective is to identify and extract a minimal substring from the paragraph that precisely addresses or answers the given question. The task is evaluated using the official SQuAD metrics, which are *F1* and *Exact Match* scores. We compare our predictions with the original form of the testing dataset generated by the filtration of [Yue et al.](https://github.com/xiangyue9607/CliniRC?tab=readme-ov-file), i.e., with the dataset before the segmentation process.

- **Paragraph Retrieval–Question Answering (PR-QA)**: Given a question and n paragraphs (i.e., report segmented into n paragraphs), the goal is to identify and extract a substring from one of the paragraphs that precisely addresses or answers a given question. Evaluation of the task is based on the *F1* and *Exact Match* scores in the same way as in the Oracle-QA task.

## Setup
1. Clone the repository
    ```sh
    git clone https://github.com/lanzv/PAR-emrQA
    ```
2. and install all requirements
    ```sh
    pip install -r requirements.txt
    ```

3. Download the emrQA dataset ```data.json``` (for instance [here](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/), but you need to register there first and sign the form and its conditions). Then, you can find the emrQA dataset under the *Community Annotations Downloads*. By default, the code works with the dataset of the following path:```./data/data.json``` (but it is adaptable via script arguments).

4. Download the BERT/LLMs models to the folder ```../models/``` (again, this is adaptable via script arguments) or just use huggingface model names.


## Prepare emrQA medication and relations subsets (and split it into train/dev/test)
In order to filter, prepare, sample train set, and split the emrQA medication and relations subsets, call the ```prepare_datasets.py``` script:
```sh
python prepare_datasets.py --data_path ./data/data.json --target_dir ./processed_data --train_ratio 0.7 --dev_ratio 0.1 --topics medication relations --train_sample_ratios 0.2 0.05 --seed 54
```

For more info, check the prepare_datasets.py argument helps.



## Run the Paragraph Retrieval - Question-Answering experiments
Run the experiment that segments prepared subsets into paragraphs, trains the given model and evaluate it on the Oracle-QA (gold paragraph is known), Paragraph Retrieval, and PRQA (gold paragraph is not known) tasks.

```sh
python run_experiment.py --train_path ./processed_data/medication-train.json --dev_path ./processed_data/medication-dev.json --test_path ./processed_data/medication-test.json --dataset_title medication --model_name BERTbase --model_path ../models/BERTbase --ft 100 --epochs 3 --seed 54
```

For more info, check the run_experiment.py argument helps.
