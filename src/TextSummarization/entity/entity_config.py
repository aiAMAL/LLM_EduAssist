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
    tokenizer_checkpoint: str


