import sys
from pathlib import Path
from src.TextSummarization.logger import logger
from src.TextSummarization.exception import CustomException
from src.TextSummarization.entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_existing_files_data(self) -> bool:
        """
        Validates if the required subfolders exist in the specified directory.

        Returns:
            bool: True if all required subfolders are present, False otherwise.
        """
        try:
            logger.info(f"\n\n------- >>  Starting data validation ...")

            # Collect all directory names in the dataset directory
            all_data_folders = {
                folder.name for folder in Path(self.config.dataset_dir).iterdir() if folder.is_dir()
            }
            logger.debug(f"Found data folders: {all_data_folders}")

            # Convert the required data folders to a set for faster lookups
            required_data_folders = set(self.config.required_data_folders)
            logger.debug(f"Required data folders: {required_data_folders}")

            # Determine if all required data folders are present
            validation_status = required_data_folders.issubset(all_data_folders)

            # Log validation status
            self._log_validation_status(validation_status, required_data_folders, all_data_folders)

            # Save validation status to the status file
            self._save_validation_status(validation_status)

            return validation_status

        except Exception as e:
            logger.error(f'Error occurred while validating dataset: {e}')
            raise CustomException(e, sys)

    def _log_validation_status(self, validation_status: bool, required_folders: set, existed_folders: set):
        """
        Logs the validation status and missing folders if applicable.

        Args:
            validation_status (bool): Result of the validation check.
            required_folders (set): Set of required folder names.
            existing_folders (set): Set of existing folder names in the dataset directory.
        """
        if validation_status:
            logger.info('Dataset validation successful. All required folders are present.')
        else:
            missing_folders = required_folders - existed_folders
            logger.error(f'Dataset validation failed. Missing folders: ' + ', '.join(missing_folders))

    def _save_validation_status(self, validation_status: bool):
        """
        Saves the validation status to a status file.

        Args:
            validation_status (bool): Result of the validation check.
        """
        try:
            with open(self.config.status_file, 'w') as f:
                f.write('Validation status: ' + ('Successful' if validation_status else 'Failed'))
            logger.info(f"Validation status written to {self.config.status_file}.")
        except IOError as e:
            logger.error(f"Failed to write validation status to file: {e}")
            raise CustomException(e, sys)
