import sys
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer
)
from src.TextSummarization.entity import ModelTrainingConfig
from src.TextSummarization.utils import load_dataset_from_disk
from src.TextSummarization.exception import CustomException
from src.TextSummarization.logger import logger


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None

    def load_model_and_tokenizer(self):
        """
        Load the model and tokenizer from the specified checkpoint.
        """
        logger.info('Loading model and tokenizer...')
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_checkpoint)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_checkpoint).to(self.device)    # if GPU add: ", torch_dtype=torch.bfloat16"
            logger.info('Model and tokenizer loaded successfully.')
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer: {e}")
            raise CustomException(e, sys)

    def _get_training_args(self) -> TrainingArguments:
        """Prepare training arguments for the Trainer class."""
        trainer_args = TrainingArguments(
            output_dir=str(self.config.root_dir),
            num_train_epochs=self.config.num_train_epochs,
            auto_find_batch_size=self.config.auto_find_batch_size,
            learning_rate=float(self.config.learning_rate),
            warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            load_best_model_at_end=self.config.load_best_model_at_end,
            report_to=self.config.report_to,
            # save_total_limit=self.config.save_total_limit
        )

        return trainer_args

    def train(self):
        try:
            """Train the model using the Trainer API."""
            logger.info("\n\nStarting the training process...")

            # Load model, tokenizer, and dataset
            self.load_model_and_tokenizer()

            dataset = load_dataset_from_disk(Path(self.config.dataset_dir))

            # Prepare data collator for dynamic padding
            data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model, padding=True)

            # Configure training arguments
            trainer_args = self._get_training_args()

            # Initialize a Trainer instance
            trainer = Trainer(
                model=self.model,
                args=trainer_args,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                train_dataset=dataset['train'],
                eval_dataset=dataset['validation']
            )

            # Begin training
            logger.info("\nStarting model training...")
            trainer.train()
            logger.info("\nTraining completed successfully.")

            # Save the trained model and tokenizer
            self.save_model_and_tokenizer()

            logger.info("Model and tokenizer saved successfully.")
        except Exception as e:
            logger.error(f"An error occurred during the training process: str({e})")
            raise CustomException(e, sys)

    def save_model_and_tokenizer(self):
        """Save the trained model and tokenizer to the specified directory."""
        try:
            model_save_path = str(self.config.model_dir)
            tokenizer_save_path = str(self.config.tokenizer_dir)

            logger.info(f"Saving model to {model_save_path} and tokenizer to {tokenizer_save_path}...")
            self.model.save_pretrained(model_save_path)
            self.tokenizer.save_pretrained(tokenizer_save_path)
            logger.info("Model and tokenizer saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save model or tokenizer: {e}")
            raise CustomException(e, sys)
