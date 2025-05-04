import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data_ingestion import ingestion
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


def preprocessing(x_train, y_train, x_test, y_test):

    numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    categorical_features = ['gender', 'smoking_history'] # that needs to be transformed


    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy='mean')),
             ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ("encoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ]
    )

    # for training
    train_numerical_output = numerical_pipeline.fit_transform(x_train[numerical_features])
    train_categorical_output = categorical_pipeline.fit_transform(x_train[categorical_features])

    # for testing
    test_numerical_output = numerical_pipeline.transform(x_test[numerical_features])
    test_categorical_output = categorical_pipeline.transform(x_test[categorical_features])


    train_x_data = np.concat([train_numerical_output, train_categorical_output, x_train[["hypertension", "heart_disease"]]], axis=1)
    test_x_data = np.concat([test_numerical_output, test_categorical_output, x_test[["hypertension", "heart_disease"]]], axis=1)

    return train_x_data, y_train, test_x_data, y_test



