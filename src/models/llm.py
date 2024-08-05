import logging
from src.eval import Evaluate
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)
import json
from src.paragraphizer import Paragraphizer
from src.models.dataset import emrqa2qa_dataset, emrqa2prqa_dataset
from src.models.llm_utils import get_model_and_tokenizer, extract_answer, get_answer
import transformers
from tqdm import tqdm
from src.eval import Evaluate
import re
from src.models.llm_utils import create_and_prepare_model, SquadDataCollator, SquadSFTTrainer
from transformers import TrainingArguments
import torch
from peft import AutoPeftModelForCausalLM



class LLMWrapperPRQA:
    """
    Wrapper class for LLM decoder-based models for Question Answering.
    Paragraph Retrieval is ignored for now.
    """


    def __init__(self, model_name):
        """
        Initialize the LLMWrapperPRQA class.

        Parameters
        ----------
        model_name: str
            model's huggingface id or the local path to the model
        """
        self.model, self.peft_config, self.tokenizer = create_and_prepare_model(model_name)

    
    def train(self, llm_train, llm_dev, epochs=10, disable_tqdm=False):
        """
        Train the model to predict the substring answers with the provided training data, 
        development data and given hyperparameters

        Parameters
        ----------
        llm_train: datasets.Dataset
            training SQuAD-like paragrpahized dataset in the huggingface Dataset format containing the messages as prompt as single item
        llm_dev: datasets.Dataset
            development SQuAD-like paragrpahized dataset in the huggingface Dataset format containing the messages as prompt as single item
        epochs: int
            number of training epochs
        disable_tqdm: bool
            disable tqdm progress bar if set to True
            (could come handy when output is printed to the file - for instance when using slurm)
        """
        training_arguments = TrainingArguments(
            save_strategy="no",
            output_dir="./model_output_dir",
            per_device_train_batch_size=1, # 1
            gradient_accumulation_steps=16,
            optim="paged_adamw_32bit",
            learning_rate=1e-6,
            fp16=True,
            bf16=False,
            max_grad_norm=0.3,
            #max_steps=10000, # ?
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="cosine",
            evaluation_strategy="epoch",
            num_train_epochs=epochs,
            disable_tqdm=disable_tqdm
        )

        self.model.config.use_cache = False

        answer_start_tokens = torch.tensor(self.tokenizer.encode("[/INST]", add_special_tokens=False)).view(-1)#torch.tensor(tokenizer.encode("\n\n```", add_special_tokens=False)).view(-1)

        data_collator = SquadDataCollator(
            answer_start_tokens=answer_start_tokens, tokenizer=self.tokenizer, mlm=False
        )


        trainer = SquadSFTTrainer(
            answer_start_tokens=answer_start_tokens,
            model=self.model,
            train_dataset=llm_train,
            eval_dataset=llm_dev,
            peft_config=self.peft_config,
            max_seq_length=2048,
            tokenizer=self.tokenizer,
            args=training_arguments,
            packing=False,
            data_collator=data_collator,
            formatting_func=lambda items: self.tokenizer.apply_chat_template(
                items["messages"], tokenize=False
            ),
        )

        trainer.train(resume_from_checkpoint=None)


        #trainer.model.save_pretrained("./biomistral_trained_relations/final_checkpoints")
        #del self.model
        #torch.cuda.empty_cache()
        #self.model = AutoPeftModelForCausalLM.from_pretrained(
        #    "./biomistral_trained_relations/final_checkpoints", device_map="auto", torch_dtype=torch.bfloat16
        #)
        #self.model = self.model.merge_and_unload()
        #self.model.save_pretrained("./biomistral_trained_relations/relations", safe_serialization=True)

    def predict(self, llm_test, disable_tqdm = True):
        """
        Predict substring answers for the provided test dataset.

        Parameters
        ----------
        llm_test: datasets.Dataset
            test dataset containing 'messages' and 'id'
        disable_tqdm: bool
            disable tqdm progress bar if set to True
            (could come handy when output is printed to the file - for instance when using slurm)

        Returns
        -------
        predictions: dict
            predictions for the Oracle-QA task, {"question_id1": "answer1", "question_id2": "answer2", ..}, the correct paragraph is known
        """
        pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        def remove_last_number_suffix(text):
            # Define the regular expression pattern to match _numbers at the end of the string
            pattern = r'_\d+$'
            # Find all matches of the pattern in the text
            matches = re.findall(pattern, text)
            if matches:
                # Get the last match
                last_match = matches[-1]
                # Replace the last match with an empty string
                text = re.sub(re.escape(last_match), '', text)
            return text

        predictions = {}
        empty = 0
        not_empty = 0
        for i, messages in enumerate(tqdm(llm_test["messages"], disable=disable_tqdm)):
            model_answer, score = get_answer(
                messages=messages,
                pipeline=pipeline,
            )
            predictions[remove_last_number_suffix(llm_test[i]["id"])] = model_answer if model_answer != None else ""
            if model_answer == None:
                empty += 1
            else: 
                not_empty += 1


        logging.info("Empty: {}, not empty: {}".format(empty, not_empty))
        return predictions, {}, {}