import pandas as pd
from log_config import _LOGGER
from ucimlrepo import fetch_ucirepo, list_available_datasets

_ONLINE_RETAIL_ID = 352


def main():
    _LOGGER.log_header("UCI Machine Learning Repository")
    online_retail = fetch_ucirepo(id=_ONLINE_RETAIL_ID)
    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    main()
