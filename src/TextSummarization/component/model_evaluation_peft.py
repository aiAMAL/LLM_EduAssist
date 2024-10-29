import os
import sys
import torch
import evaluate
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from src.TextSummarization.logger import logger
from src.TextSummarization.exception import CustomException
from src.TextSummarization.entity import ModelEvaluationConfig
from src.TextSummarization.utils import load_dataset_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        """
        Initializes ModelEvaluation with provided configuration and device setup.
        """
        self.config = config
        # self.model = None
        self.tokenizer = None
        self.base_peft_model = None
        self.peft_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model_and_tokenizer(self):
        """
        Load the model and tokenizer from the specified checkpoint.
        """
        logger.info('Loading model and tokenizer...')
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_dir)
            # self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_dir).to(self.device)
            self.base_peft_model = AutoModelForSeq2SeqLM.from_pretrained(self.config.base_model_checkpoint).to(self.device)  # if GPU add: ", torch_dtype=torch.bfloat16"

            # Load the pre-trained PEFT model checkpoint
            self.peft_model = PeftModel.from_pretrained(
                self.base_peft_model,
                self.config.model_dir,
                # torch_dtype=torch.bfloat16,
                is_trainable=False
            ).to(self.device)

            logger.info('Model and tokenizer loaded successfully.')
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer: {e}")
            raise CustomException(e, sys)


    def convert_examples_to_features(self, example_batch):
        """
        Converts raw dialogue examples to tokenized input features.

        Args:
            example_batch (dict): Batch of examples containing 'dialogue' and 'summary'.

        Returns:
            dict: Tokenized inputs including 'input_ids', 'attention_mask', and 'labels'.
        """
        try:
            start_prompt = (
                "Summarize the main points, actions, and decisions from the following conversation. "
                "Keep it brief and avoid unnecessary details.\n\n"
            )

            end_prompt = "\n\nSummary: "
            prompts = [f'{start_prompt} {dialogue} {end_prompt}' for dialogue in example_batch['dialogue']]

            # Tokenize inputs and targets
            input_encodings = self.tokenizer(prompts, padding=True, truncation=True, return_tensors='pt')
            target_encodings = self.tokenizer(example_batch['summary'], padding=True, truncation=True, return_tensors='pt')

            return {
                'input_ids': input_encodings['input_ids'],
                'attention_mask': input_encodings['attention_mask'],
                'labels': target_encodings['input_ids']
            }

        except Exception as e:
            logger.error(f'Error during tokenization: {e}')
            CustomException(e, sys)

    def process_test_dataset(self):
        """
        Loads and processes the test dataset, tokenizing and saving for evaluation.

        Returns:
            Dataset: Processed test dataset.
        """
        logger.info(f"\n\nProcessing test dataset...")
        try:
            # Load the dataset from Data_Ingestion folder
            logger.info(f"Loading test dataset from {self.config.dataset_dir}...")
            test_dataset_samsum = load_dataset_from_disk(Path(self.config.dataset_dir))['test']
            print('\nd')

            # Tokenize the dataset and remove unnecessary columns
            test_dataset_pt = test_dataset_samsum.map(self.convert_examples_to_features, batched=True)
            test_dataset_pt = test_dataset_pt.remove_columns(['id', 'topic', 'dialogue', 'summary'])
            print('\ne')

            # Prepare the dataset for PyTorch
            test_dataset_pt.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
            print('\nf')

            # Save processed dataset
            logger.info(f"Saving tokenized datasets...")
            save_path = self.config.tokenized_test_dataset_dir
            test_dataset_pt.save_to_disk(save_path)
            logger.info(f"Tokenized datasets successfully saved to {save_path}.")
            print('\nf')

            return test_dataset_pt
        except Exception as e:
            logger.error(f"Error processing test dataset: {e}")
            raise CustomException(e, sys)

    def generate_batches(self, data, batch_size):
        """
        Generates batches from a dataset list.
        """
        for i in range(0, len(data), batch_size):
            yield data[i: i + batch_size]

    def calculate_metric_on_test_ds(self, dataset, input_text, label, batch_size, metric) -> list:
        """
        Calculates the specified metric on the test dataset.

        Args:
            dataset (Dataset): Tokenized test dataset.
            input_text (str): Input feature key name.
            label (str): Label feature key name.
            batch_size (int): Number of samples per batch.
            metric (Metric): Metric object for evaluation.

        Returns:
            dict: Computed metric scores.
        """
        try:
            input_batches = list(self.generate_batches(dataset[input_text], batch_size))
            target_batches = list(self.generate_batches(dataset[label], batch_size))
            attention_mask_batches = list(self.generate_batches(dataset['attention_mask'], batch_size))
            print('\ng')

            for input_batch, attention_mask_batch, target_batch in tqdm(
                    zip(input_batches, attention_mask_batches, target_batches), total=len(input_batches)):
                print('\nh')

                # Generate summaries
                summaries = self.peft_model.generate(
                    input_ids=input_batch.to(self.device),
                    attention_mask=attention_mask_batch.to(self.device),
                    max_length=128,
                    num_beams=8,
                    length_penalty=0.8
                )

                print('\ni')

                # Decode summaries and targets
                decoded_summaries = [self.tokenizer.decode(summary,
                                                           skip_special_tokens=True,
                                                           clean_up_tokenization_spaces=True) for summary in summaries]

                decoded_targets = [self.tokenizer.decode(target,
                                                         skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=True) for target in target_batch]
                print('\nj')

                # Add predictions and references to the metric
                metric.add_batch(predictions=decoded_summaries, references=decoded_targets)
                print('\nk')

            return metric.compute()
        except Exception as e:
            logger.error(f'Error computing metric: {e}')
            raise CustomException(e, sys)

    def evaluate_model(self):
        """
        Orchestrates model evaluation by processing the test dataset and computing ROUGE scores.
        """
        try:
            logger.info("\n\n------- >>  Evaluating model on test dataset...")
            print('\na')
            self.load_model_and_tokenizer()
            print('\naa')

            test_datasets_samsum_pt = self.process_test_dataset()
            print('\nb')

            # For faster evaluation, use a subset if needed
            test_datasets_samsum_pt = test_datasets_samsum_pt.select(range(1))
            print(f"Columns in 'test' dataset sample: {test_datasets_samsum_pt.column_names}")
            print('\nc')

            rouge_metric = evaluate.load('rouge')
            rouge_names = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']

            scores = self.calculate_metric_on_test_ds(
                test_datasets_samsum_pt,
                input_text='input_ids',
                batch_size=2,
                label='labels',
                metric=rouge_metric
            )

            # Format and save scores
            rouge_dict = {name: scores[name] for name in rouge_names}
            df_rouge_score = pd.DataFrame(rouge_dict, index=['FineTuned_Score'])
            logger.info(f"\nRouge Scores:\n {df_rouge_score}")

            metric_file = self.config.metrics_file_name

            if os.access(Path(metric_file).parent, os.W_OK):
                df_rouge_score.to_csv(metric_file)
                logger.info(f"Rouge scores saved to {metric_file}")
            else:
                logger.error(f'Permission denied: Unable to save metrics file: {metric_file}')

        except Exception as e:
            logger.error(f'Error during evaluating model on testing dataset: {e}')
            CustomException(e, sys)
