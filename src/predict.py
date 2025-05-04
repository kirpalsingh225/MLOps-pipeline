import mlflow
import pandas as pd
from data_preprocessing import preprocessing
from data_ingestion import ingestion
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://127.0.0.1:5000")

x, y = ingestion()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
_, _, test, _ = preprocessing(x_train, y_train, x_test, y_test)

logged_model = 'runs:/4a4f9d42f7c24786884a1245216ba91c/lr'

loaded_model = mlflow.pyfunc.load_model(logged_model)


print(loaded_model.predict(test))