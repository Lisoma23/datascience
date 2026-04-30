# fichier pour importer et checker les données du .csv
import pandas as pd

#check df is imported correctly
def check_data():
    # Load the data (s'assurer que le fichier existe toujours dans le bon chemin)
    df = pd.read_csv("./../customer_churn.csv")
    return df.head()