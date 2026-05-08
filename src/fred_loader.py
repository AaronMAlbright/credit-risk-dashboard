from fredapi import Fred
from dotenv import load_dotenv
import os
import pandas as pd
import time

load_dotenv()

fred = Fred(api_key=os.getenv("FRED_API_KEY"))


def get_series(series_id, start_date="1999-01-01", retries=3, delay=2):
    last_error = None

    for attempt in range(retries):
        try:
            data = fred.get_series(series_id, observation_start=start_date)
            df = pd.DataFrame(data, columns=["value"])
            df.index.name = "date"
            return df

        except Exception as e:
            last_error = e
            print(f"FRED pull failed for {series_id}. Attempt {attempt + 1}/{retries}.")
            time.sleep(delay)

    raise RuntimeError(f"Failed to pull {series_id} from FRED after {retries} attempts: {last_error}")