from datetime import datetime

from .constants import DATE_FORMAT, DATETIME_FORMAT


def append_date(base: str, time: bool = True) -> str:
    """
    Append today's date to base string.
    """
    today = datetime.now()
    if time:
        date_str = today.strftime(DATETIME_FORMAT)
    else:
        date_str = today.strftime(DATE_FORMAT)
    name = base + "-" + date_str
    return name
