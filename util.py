from datetime import datetime


class Data:
    def __init__(self) -> None:
        self._is_set: bool = False

    @property
    def is_set(self) -> bool:
        return self._is_set


DATE_FORMAT = "%m_%d_%y"
DATETIME_FORMAT = "%Y-%m-%d--%H:%M:%S"
# Directory paths
_DAT = "dat"


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
