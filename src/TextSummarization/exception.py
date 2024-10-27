import sys


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Extracts detailed error message including the filename and line number where the error occurred.

    Args:
        error (Exception): The error/exception instance.
        error_detail (sys): The sys module to fetch traceback details.

    Returns:
        str: Formatted error message with script name, line number, and error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f'Error occurred in python script named [{file_name}], ' \
                    f'line number [{line_number}], ' \
                    f'error message: [{str(error)}]'
    return error_message


class CustomException(Exception):
    def __init__(self, error_message: Exception, error_details: sys):
        """
        Custom exception class that includes detailed error message.

        Args:
            error_message (Exception): The error/exception instance.
            error_details (sys): The sys module to fetch traceback details.
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_details)

    def __str__(self):
        """
        Returns the detailed error message when the exception is printed.

        Returns:
            str: Detailed error message.
        """
        return self.error_message

