import os
import re
import shutil
import logging
from util import append_date, DATETIME_FORMAT


def setup_logging():
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    basename = "onlinekcenter"
    filename = append_date(basename, time=True) + ".log"
    filepath = os.path.join(logs_dir, filename)
    terminal_format = "[%(levelname)s]: %(message)s"
    file_format = "[%(asctime)s - %(name)s - %(levelname)s]: %(message)s"
    logging.basicConfig(
        filename=filepath,
        level=logging.DEBUG,
        format=file_format,
        datefmt=DATETIME_FORMAT,
    )

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Create a formatter for the console handler
    console_formatter = logging.Formatter(terminal_format)
    console_handler.setFormatter(console_formatter)

    # Get the root logger and add the console handler to it
    logging.getLogger().addHandler(console_handler)


class InfoLogger:
    INFO_CHARLEN: int = 8

    def __init__(self, logger=logging, special_char="-", precision=2):
        self.logger = logger
        self.special_char = special_char
        self.precision = precision

    def blank_line(self):
        print()

    def separator_line(self):
        terminal_width = shutil.get_terminal_size((80, 20)).columns
        num_characters = terminal_width // 2
        separator_line = (
            f"{self.special_char} " * num_characters + f"{self.special_char}"
        )
        print(separator_line)

    def separator_block(self):
        self.blank_line()
        self.separator_line()
        self.blank_line()

    def format_numbers_in_string(self, msg: str) -> str:
        pattern = re.compile(r"(\d+\.\d+)")

        def repl(match):
            return f"{float(match.group()):.{self.precision}f}"

        return pattern.sub(repl, msg)

    def log_header(self, msg: str):
        msg = self.format_numbers_in_string(msg)
        sep: str = self.special_char * (len(msg) + self.INFO_CHARLEN)
        print(sep)
        logging.info(msg)
        print(sep)

    def log_subheader(self, msg: str):
        msg = self.format_numbers_in_string(msg)
        logging.info(f"{msg}.")

    def log_body(self, msg: str):
        msg = self.format_numbers_in_string(msg)
        logging.info(f"     -> {msg}.")

    def log_special(self, msg: str):
        msg = self.format_numbers_in_string(msg)
        logging.info(f"--> {msg}! <--")


_LOGGER = InfoLogger()


def gurobi_log_file():
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    basename = "onlinekcenter_gurobi"
    filename = append_date(basename, time=True) + ".log"
    filepath = os.path.join(logs_dir, filename)
    return filepath
