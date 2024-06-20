import logging
import os
import re
import shutil

import numpy as np

from optiface.utils import append_date, _DATETIME_FORMAT
from optiface.paths import _FORKNAME, _LOGS_DIR

_CONSOLELEVEL = logging.INFO
_FILELEVEL = logging.DEBUG


def setup_terminal_logger():
    terminal_msg_format = "[%(levelname)s]: %(message)s"
    console_handler = logging.StreamHandler()
    console_handler.setLevel(_CONSOLELEVEL)
    console_formatter = logging.Formatter(terminal_msg_format)
    console_handler.setFormatter(console_formatter)
    logging.getLogger().addHandler(console_handler)


def setup_logfile():
    if not os.path.exists(_LOGS_DIR):
        os.makedirs(_LOGS_DIR)
    filename = append_date(_FORKNAME, time=True) + ".log"
    filepath = os.path.join(logs_dir, filename)
    file_msg_format = "[%(asctime)s - %(name)s - %(levelname)s]: %(message)s"
    logging.basicConfig(
        filename=filepath,
        level=_FILELEVEL,
        format=file_msg_format,
        datefmt=DATETIME_FORMAT,
    )


def setup_logging():
    setup_terminal_logger()
    setup_logfile()


_INFO_CHARLEN: int = len("[INFO]: ")
_DEBUG_CHARLEN: int = len("[DEBUG]: ")
_WARN_CHARLEN: int = len("[WARNING]: ")
_ERROR_CHARLEN: int = len("[ERROR]: ")
_SPACE_INDENT: int = 4
_FINAL_INDENT_CHARLEN: int = max(_INFO_CHARLEN, _DEBUG_CHARLEN) + _SPACE_INDENT
_INFO_INDENT: str = f"{' ' * (_FINAL_INDENT_CHARLEN - _INFO_CHARLEN)}-> "
_DEBUG_INDENT: str = f"{' ' * (_FINAL_INDENT_CHARLEN - _DEBUG_CHARLEN)}>>>> "
_WARN_INDENT: str = f"{' ' * (_FINAL_INDENT_CHARLEN - _WARN_CHARLEN)}>>>> "
_ERROR_INDENT: str = f"{' ' * (_FINAL_INDENT_CHARLEN - _ERROR_CHARLEN)}>>>> "
_SPECIAL_CHAR = "-"
_PRECISION = 2


def blank_line():
    print()


def separator_line():
    terminal_width = shutil.get_terminal_size((80, 20)).columns
    num_characters = terminal_width // 2
    separator_line = f"{_SPECIAL_CHAR} " * num_characters + f"{_SPECIAL_CHAR}"
    print(separator_line)


def separator_block():
    separator_line()
    separator_line()


def clear_page():
    for i in range(4):
        blank_line()


def format_numbers_in_string(msg: str) -> str:
    if type(msg) != str:
        logging.warning(
            f"Expected string in format_numbers_in_string, got {type(msg)}. Cannot guarantee proper formatting."
        )
        return msg
    pattern = re.compile(r"(\d+\.\d+)")

    def repl(match):
        return f"{float(match.group()):.{_PRECISION}f}"

    return pattern.sub(repl, msg)


def log_matrix(matrix: np.ndarray, name: str, index: int = 0):
    """
    Prints a 2D numpy ndarray in a readable format with specified decimal places.

    Parameters:
    matrix (np.ndarray): 2D numpy array to be printed.
    decimal_places (int): Number of decimal places for formatting the numbers.
    """
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Input must be a numpy ndarray.")
    if matrix.ndim != 2:
        raise ValueError("Input array must be 2-dimensional.")
    log_debug(f"{name} matrix", ":")
    for row in matrix:
        formatted_row = []
        for x in row:
            formatted_row.append(str(x))
        log_debug(" ".join(formatted_row), "")


def log_header(msg: str):
    blank_line()
    separator_block()
    msg = format_numbers_in_string(msg)
    logging.info(f"{msg}.")
    separator_block()
    blank_line()


def log_subheader(msg: str):
    msg = format_numbers_in_string(msg)
    blank_line()
    logging.info(f"{msg}.")
    blank_line()


def log_body(msg: str):
    msg = format_numbers_in_string(msg)
    logging.info(f"{_INFO_INDENT}{msg}.")


def log_bodydebug(msg: str):
    msg = format_numbers_in_string(msg)
    logging.debug(f"{_INFO_INDENT}{msg}.")


def log_debug(msg: str, final_punc: str = "."):
    msg = format_numbers_in_string(msg)
    logging.debug(f"{_DEBUG_INDENT}{msg}{final_punc}")


def log_special(msg: str):
    msg = format_numbers_in_string(msg)
    logging.info(f"--> {msg}! <--")


def log_warning(msg: str, final_punc: str = "."):
    msg = format_numbers_in_string(msg)
    logging.warn(f"{_WARN_INDENT}{msg}{final_punc}")


def log_error(msg: str, final_punc: str = "."):
    msg = format_numbers_in_string(msg)
    logging.error(f"{_ERROR_INDENT}{msg}{final_punc}")
