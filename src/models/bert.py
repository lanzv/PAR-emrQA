from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, DefaultDataCollator, pipeline
from datasets import Dataset
import importlib.util 
import torch
from tqdm.auto import tqdm
import numpy as np
import collections
import random
import evaluate as eval_lib
import matplotlib.pyplot as plt
import re
import json
import logging
from torch.nn.functional import softmax


class BERTWrapperPRQA:
    def __init__(self, model_name):
        self.model_name = model_name

        # load tokenizer and model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name).to(self.device)
        self.data_collator = DefaultDataCollator()

        # init trainer, for now without training/validation data
        training_args = TrainingArguments(
            save_strategy="no",
            output_dir="./model_output_dir",
            per_device_eval_batch_size=256
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
    
    def train(self, train_data, dev_data, learning_rate=3e-5, epochs=1, weight_decay=0.01, train_batch_size=16, eval_batch_size=256, seed=54, disable_tqdm=False):
        tokenized_train = train_data.map(self.__preprocess_function, batched=True, remove_columns=train_data.column_names)
        tokenized_dev = dev_data.map(self.__preprocess_function, batched=True, remove_columns=train_data.column_names) 
        
        logging.info("training data are prepared")
        training_args = TrainingArguments(
            save_strategy="no",
            output_dir="./model_output_dir",
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            push_to_hub=True,
            disable_tqdm=disable_tqdm
        )
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_dev,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )

        self.trainer.train()
        logging.info("the model is trained")

    def predict(self,  test_prqa_dataset, seed=54, disable_tqdm=False, confidence_type="min_cls"):
        test_dataset = test_prqa_dataset["test_data"]
        report_boundaries = test_prqa_dataset["report_boundaries"] 
        question_ids = test_prqa_dataset["question_ids"] 
        gold_paragraphs = test_prqa_dataset["gold_paragraphs"]

        # prepare validation features and do the prediction
        validation_features = test_dataset.map(
            self.__prepare_validation_features,
            batched=True,
            remove_columns=test_dataset.column_names
        )
        logging.info("evaluation data are prepared")
        #test_loader = DataLoader(validation_features, batch_size=64, shuffle=False)
        #raw_predictions = self.trainer.prediction_loop(test_loader, description="prediction")
        raw_predictions = self.trainer.predict(validation_features)
        validation_features.set_format(type=validation_features.format["type"], columns=list(validation_features.features.keys()))
        predictions, confidences = self.__postprocess_qa_predictions(test_dataset, validation_features, raw_predictions.predictions, disable_tqdm=disable_tqdm, confidence_type=confidence_type)


        # choose the argmax for questions i...j regarding report_boundaries
        pr_predictions = {}
        prqa_predictions = {}
        qa_predictions = {}
        for k, (i, j) in enumerate(zip(report_boundaries[:-1], report_boundaries[1:])):
            paragraph_predictions = confidences[i:j]
            # PR
            pr_predictions[question_ids[k]] = np.argsort(paragraph_predictions, axis=0)[::-1]
            # PRQA
            top_paragraph = pr_predictions[question_ids[k]][0]
            prqa_predictions[question_ids[k]] = predictions["{}_{}".format(question_ids[k], top_paragraph)]
            # QA
            for par_id in pr_predictions[question_ids[k]]:
                if par_id in gold_paragraphs[k]:
                    # take the most confident gold paragraph (in case there are more of gold paragraphs)
                    qa_predictions[question_ids[k]] = predictions["{}_{}".format(question_ids[k], par_id)]
                    break
        
        return qa_predictions, pr_predictions, prqa_predictions


    def __preprocess_function(self, examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples["question"] = [q.lstrip() for q in examples["question"]]

        # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=384,
            stride=128, #doc_stride
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

       
    def __prepare_validation_features(self, examples):
        """
        Authors: Huggingface
        """
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples["question"] = [q.lstrip() for q in examples["question"]]
    
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=384,
            stride=128, #doc_stride
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    
        # We keep the example_id that gave us this feature and we will store the offset mappings.
        tokenized_examples["example_id"] = []
    
        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1
    
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])
    
            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
    
        return tokenized_examples


    
    def __postprocess_qa_predictions(self, examples, features, raw_predictions, disable_tqdm=False, confidence_type="min_cls"):
        """
        Authors: Huggingface
        """
        all_start_logits, all_end_logits = raw_predictions
        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)
    
        # The dictionaries we have to fill.
        predictions = {}
        confidences = []
    
        # Logging.
        print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")
    
        # Let's loop over all the examples!
        for example_index, example in enumerate(tqdm(examples, disable=disable_tqdm)):
            # Those are the indices of the features associated to the current example.
            feature_indices = features_per_example[example_index]
            context = example["context"]



            start_logits = torch.tensor(np.array([all_start_logits[i] for i in feature_indices]))
            end_logits = torch.tensor(np.array([all_end_logits[i] for i in feature_indices]))
            mask = [[True if mapping is None else False for mapping in features[i]["offset_mapping"]] for i in feature_indices]
            for mask_id in range(len(mask)):
                mask[mask_id][0] = False    # consider impossible answers for softmax computation
            mask = torch.tensor(mask, dtype=torch.bool)
            start_logits[mask] = -10000
            end_logits[mask] = -10000
            start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)
            end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)
            cls_score = 1
            for start_prob, end_prob in zip(start_probabilities, end_probabilities):
                if start_prob[0]*end_prob[0] < cls_score:
                    cls_score = start_prob[0]*end_prob[0]

            start_probabilities[:, 0] = 0 # don't consider impossible answers during finding optimal text and confidence
            end_probabilities[:, 0] = 0 # don't consider impossible answers during finding optimal text and confidence
            candidates = []
            for start_probs, end_probs in zip(start_probabilities, end_probabilities):
                scores = start_probs[:, None] * end_probs[None, :]
                idx = torch.triu(scores).argmax().item()
                start_idx = idx // scores.shape[1]
                end_idx = idx % scores.shape[1]
                score = scores[start_idx, end_idx].item()
                candidates.append((start_idx, end_idx, score))
            max_score = -1
            start_char = 0
            end_char = 0
            text = ""
            for candidate, feature_index in zip(candidates, feature_indices):
                offset_mapping = features[feature_index]["offset_mapping"]
                start_idx, end_idx, score = candidate
                if score > max_score:
                    start_char = offset_mapping[start_idx][0]
                    end_char = offset_mapping[end_idx][1]
                    max_score = score
                    text = context[start_char: end_char]
                    
            text = context[start_char: end_char]          
            predictions[example["id"]] = text
            if confidence_type == "max_score":
                confidences.append(max_score)
            elif confidence_type == "min_cls":
                confidences.append(-cls_score)
            else:
                logging.warn("We don't support confidence type '{}'".format(confidence_type))
                confidences.append(0.0)

        return predictions, confidences