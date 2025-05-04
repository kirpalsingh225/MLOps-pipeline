import pytest
from src.data_ingestion import ingestion
from src.data_preprocessing import preprocessing
from sklearn.model_selection import train_test_split

def test_ingestion():
    X, y = ingestion()
    assert len(X) > 0, "Ingested features should not be empty"
    assert len(y) > 0, "Ingested labels should not be empty"
    assert len(X) == len(y), "Features and labels should have same length"

def test_preprocessing():
    X, y = ingestion()
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    x_train_processed, y_train_out, x_test_processed, y_test_out = preprocessing(x_train, y_train, x_test, y_test)

    assert len(x_train_processed) == len(y_train_out), "Processed train data mismatch"
    assert len(x_test_processed) == len(y_test_out), "Processed test data mismatch"
    assert x_train_processed.shape[1] == x_test_processed.shape[1], "Feature count mismatch after preprocessing"
