from datetime import date

DATE_FORMAT = "%m_%d_%y"
DATETIME_FORMAT = "%Y-%m-%d--%H:%M:%S"

def append_date(base: str, time: bool = True) -> str:
    """
    Append today's date to base string.
    """
    today = date.today()
    if time: 
        date_str = today.strftime(DATETIME_FORMAT)
    else:
        date_str = today.strftime(DATE_FORMAT)
    name = base + "-" + date_str
    return name 
