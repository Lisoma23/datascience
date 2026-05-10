# fichier pour importer et checker les données du .csv
from pathlib import Path

import pandas as pd

_CSV_PATH = Path(__file__).resolve().parent.parent.parent / "customer_churn.csv"

#check df is imported correctly
def check_data():
    # Load the data (s'assurer que le fichier existe toujours dans le bon chemin)
    try:
        df = pd.read_csv(_CSV_PATH)
        return df.head()
    except Exception:
        return None
