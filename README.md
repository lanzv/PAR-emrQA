# PAR-emrQA - Paragraph Segmentation of Two emrQA Subsets: Medication and Relations

This repository provides code for the Question-Answering and Paragraph Retrieval experiments in the BioNLP 2024 and Shared Tasks @ ACL 2024 paper: **Paragraph Retrieval for Enhanced Question Answering in Clinical Documents**



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
