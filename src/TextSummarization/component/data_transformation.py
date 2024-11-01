import sys
import torch
import numpy as np
from pathlib import Path
from src.TextSummarization.entity import DataTransformationConfig
from src.TextSummarization.logger import logger
from src.TextSummarization.exception import CustomException
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import DatasetDict
from src.TextSummarization.utils import load_dataset_from_disk


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        """
        Initializes the data transformation with the provided configuration.

        Args:
            config (DataTransformationConfig): Configuration for data transformation, including paths and tokenizer checkpoint.
        """
        self.config = config
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_checkpoint, use_fast=True)
            # self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        except Exception as e:
            logger.error(f"Error initializing tokenizer: {e}")
            raise CustomException(e, sys)

    def convert_examples_to_features(self, example_batch):

        try:
            # Formatting input with a prompt
            start_prompt = (
                'Summarize the main points, actions, and decisions from the following conversation. ' \
                'Keep it brief and avoid unnecessary details.\n\n'
            )
            end_prompt = '\n\nSummary: '
            prompt = [f"{start_prompt} {dialogue} {end_prompt}" for dialogue in example_batch['dialogue']]

            # Tokenize dialogues (inputs) & summaries (targets)
            input_encodings = self.tokenizer(prompt, truncation=True, padding=True, return_tensors='pt')
            target_encodings = self.tokenizer(example_batch['summary'], truncation=True, padding=True, return_tensors='pt')

            return {
                'input_ids': input_encodings['input_ids'],
                'attention_mask': input_encodings['attention_mask'],
                'labels': target_encodings['input_ids']
            }
        except Exception as e:
            logger.error(f'Error during tokenization: {e}')
            raise CustomException(e, sys)

    def process_dataset(self, dataset, dataset_name: str, columns_to_remove: list, filter_frequency: int = 1):
        """
        Processes the dataset by converting it to tokenized features, removing unnecessary columns, and saving it.

        Args:
            dataset (datasets.Dataset): The dataset to be processed.
            dataset_name (str): Name of the dataset ('train' or 'validation').
            columns_to_remove (list): Columns to be removed after tokenization.
            filter_frequency (int): Frequency at which examples are filtered for testing correctness of the code rapidly.

        Returns:
            datasets.Dataset: Tokenized and processed dataset.
        """
        try:
            logger.info(f"Processing {dataset_name} dataset...")

            # Optional: Filter dataset for quicker processing during testing code
            dataset = dataset.filter(lambda example, index: index % filter_frequency == 0, with_indices=True)
            # logger.info(f"Filtered {dataset_name} dataset size: {dataset_pt.num_rows}")

            # Apply tokenization
            dataset_pt = dataset.map(self.convert_examples_to_features, batched=True)
            # logger.info(f"Initial {dataset_name} dataset size: {dataset_pt.num_rows}")

            # Remove unnecessary columns
            dataset_pt = dataset_pt.remove_columns(columns_to_remove)

            # Set dataset format for PyTorch tensors
            dataset_pt.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

            return dataset_pt

        except Exception as e:
            logger.error(f"Error processing {dataset_name} dataset: {e}")
            raise CustomException(e, sys)

    def convert_and_save_dataset(self):
        """
        Loads, processes, and saves tokenized datasets for training and validation.
        """
        try:
            logger.info(f"\n\n------- >>  Data Transformation: Process train and validation datasets...")

            # Load the dataset
            logger.info(f"Loading dataset from {self.config.dataset_dir}...")
            dataset_samsum = load_dataset_from_disk(Path(self.config.dataset_dir))

            # Define columns to be removed
            columns_to_remove = ['id', 'topic', 'dialogue', 'summary']

            # Process the train and validation datasets
            train_dataset = self.process_dataset(dataset_samsum['train'], 'train', columns_to_remove, filter_frequency=10)  #50)
            valid_dataset = self.process_dataset(dataset_samsum['validation'], 'validation', columns_to_remove, filter_frequency=2)#  20)

            # Save processed datasets to disk
            logger.info(f"Saving tokenized datasets...")
            save_path = self.config.tokenized_dataset_dir

            processed_dataset_pt = DatasetDict({
                'train': train_dataset,
                'validation': valid_dataset
            })
            processed_dataset_pt.save_to_disk(save_path)

            logger.info(f"Tokenized datasets successfully saved to {save_path}.")

        except Exception as e:
            logger.error(f'Error during dataset conversion or saving: {e}')
            raise CustomException(e, sys)
