import sys
import yaml
from pathlib import Path
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from src.TextSummarization.logger import logger
from src.TextSummarization.exception import CustomException


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its content as a ConfigBox object.

    Args:
        path_to_yaml (Path): Path to the YAML file.

    Returns:
        ConfigBox: Parsed content of the YAML file.

    Raises:
        ValueError: If the YAML file is empty.
        CustomException: For any other exceptions that occur during file reading.
    """

    try:
        if not path_to_yaml.exists():
            raise FileNotFoundError(f"Yaml File not found at: {path_to_yaml}")

        with open(path_to_yaml, 'r') as file:
            config = yaml.safe_load(file)
            if config is None:
                raise BoxValueError('Yaml file is empty')
            logger.info(f'Yaml file: {path_to_yaml} loaded successfully')
        return ConfigBox(config)

    except Exception as e:
        logger.error(f"Error reading YAML file: {e}")
        raise CustomException(e, sys)


def create_directories(path_to_directories: list[Path], verbose: bool = True):
    """
    Creates directories from a list of paths.

    Args:
        path_to_directories (list): A list of directory paths to be created.
        verbose (bool): If True, logs each created directory path.

    Raises:
       CustomException: If any directory creation fails.
    """
    for path in path_to_directories:
        path.mkdir(parents=True, exist_ok=True)
        if verbose:
            logger.info(f'Created directory at {path}')

