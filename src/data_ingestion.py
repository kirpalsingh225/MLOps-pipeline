import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def ingestion():
    try:
        df = pd.read_csv('data/processed/processed.csv')
        x = df.drop(["diabetes"], axis=1)
        y = df[["diabetes"]]

        return x, y
    except Exception as e:
        print("Error while data ingestion due to " + str(e))

