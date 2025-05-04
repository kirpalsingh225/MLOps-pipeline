from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
from mlflow.models import infer_signature
from data_preprocessing import preprocessing
from data_ingestion import ingestion
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import numpy as np


def log_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label',
        title='Confusion Matrix'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black")
    fig.tight_layout()
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)


def train():

    x, y = ingestion()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    x_train, y_train, x_test, y_test = preprocessing(x_train, y_train, x_test, y_test)

    logistic_params = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs'],
    'penalty': ['l2'],
    'max_iter': [100, 200]
    }

    rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
    }

    lr = LogisticRegression()
    rf = RandomForestClassifier()

    mlflow.set_tracking_uri("http://127.0.0.1:5000")


    # logistic regression
    mlflow.set_experiment("Logistic Regression")

    with mlflow.start_run():
        lr_grid = GridSearchCV(lr, logistic_params, cv=3, scoring='accuracy', verbose=2)
        lr_grid.fit(x_train, y_train)
        best_lr = lr_grid.best_estimator_

        lr_predict = lr_grid.predict(x_test)
        log_confusion_matrix(y_test, lr_predict, class_names=["No Diabetes", "Diabetes"])
        accuracy = accuracy_score(lr_predict, y_test)
        cm = confusion_matrix(lr_predict, y_test)
        cr = classification_report(lr_predict, y_test)

        with open("models/lr.pkl", "wb") as f:
            pickle.dump(lr_grid, f)

        #logging
        mlflow.log_params(lr_grid.best_params_)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(str(cr), "confusion_report.txt")

        #log model
        signature = infer_signature(x_train, lr_grid.predict(x_train))
        model_info = mlflow.sklearn.log_model(lr_grid, "lr", signature=signature)

    # random forst
    mlflow.set_experiment("Random Forest")

    with mlflow.start_run():
        rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='accuracy', verbose=2)
        rf_grid.fit(x_train, y_train)
        best_rf = rf_grid.best_estimator_

        rf_predict = rf_grid.predict(x_test)
        accuracy = accuracy_score(rf_predict, y_test)
        cm = confusion_matrix(rf_predict, y_test)
        cr = classification_report(rf_predict, y_test)

        with open("models/rf.pkl", "wb") as f:
            pickle.dump(rf_grid, f)

        #logging
        mlflow.log_params(rf_grid.best_params_)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(str(cr), "classification_report.txt")

        signature = infer_signature(x_train, rf_grid.predict(x_train))
        model_info = mlflow.sklearn.log_model(rf_grid, "rf", signature=signature)


if __name__ == "__main__":
    train()