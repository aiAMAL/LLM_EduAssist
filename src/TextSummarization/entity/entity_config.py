from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    dataset_dir: Path
    source_data: str


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    dataset_dir: Path
    status_file: Path
    required_data_folders: list[str]


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    dataset_dir: Path
    tokenized_dataset_dir: Path
    tokenizer_checkpoint: str


@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    dataset_dir: Path
    model_dir: Path
    tokenizer_dir: Path
    model_checkpoint: str
    num_train_epochs: int
    auto_find_batch_size: bool
    learning_rate: float
    warmup_steps: int
    per_device_train_batch_size: int
    weight_decay: float
    logging_steps: int
    eval_strategy: str
    eval_steps: int
    save_steps: float
    gradient_accumulation_steps: int
    load_best_model_at_end: bool
    report_to: str
    save_total_limit: int
    lora_r: int
    lora_alpha: int
    target_modules: list[str]
    lora_dropout: float
    bias: str
    task_type: str
    

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    dataset_dir: Path
    tokenized_test_dataset_dir: Path
    model_dir: Path
    tokenizer_dir: Path
    metrics_file_name: Path
    base_model_checkpoint: Path



