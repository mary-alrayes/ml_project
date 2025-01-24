from matplotlib import pyplot as plt
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    StratifiedShuffleSplit,
    train_test_split,
)
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.utils import resample
from project.utility.Enum import RegressionMetrics, TaskType
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay

# function to remove the id column from the data
def removeId(data):
    return data.drop("ID", axis=1, errors="ignore")


def min_max_scaling(X, feature_range=(-1, 1)):
    """
    Applica la riscalatura Min-Max a un array numpy.

    Args:
        X (numpy.ndarray): Dati di input (array 2D).
        feature_range (tuple): Intervallo di riscalatura desiderato (default: [-1, 1]).

    Returns:
        numpy.ndarray: Dati scalati nell'intervallo specificato.
        numpy.ndarray: Valori minimi originali delle feature.
        numpy.ndarray: Valori massimi originali delle feature.
    """
    min_val, max_val = feature_range  # Intervallo target

    X_min = np.min(X, axis=0)  # Minimi delle colonne (features)
    X_max = np.max(X, axis=0)  # Massimi delle colonne (features)

    # Evitare divisione per zero nel caso di feature costanti
    X_scaled = (X - X_min) / (X_max - X_min + 1e-8)  # Normalizzazione a [0,1]
    X_scaled = (
            X_scaled * (max_val - min_val) + min_val
    )  # Riscalatura al range desiderato

    return X_scaled, X_min, X_max


def min_max_rescale(X, X_min, X_max, feature_range=(-1, 1)):
    """
    Riscalatura di nuovi dati usando i min/max pre-calcolati.

    Args:
        X (numpy.ndarray): Nuovi dati da riscalare.
        X_min (numpy.ndarray): Valori minimi delle feature dal set di training.
        X_max (numpy.ndarray): Valori massimi delle feature dal set di training.
        feature_range (tuple): Intervallo di riscalatura desiderato (default: [-1, 1]).

    Returns:
        numpy.ndarray: Nuovi dati scalati.
    """
    min_val, max_val = feature_range

    X_scaled = (X - X_min) / (X_max - X_min + 1e-8)
    X_scaled = X_scaled * (max_val - min_val) + min_val

    return X_scaled


# ----------------------------CLASSIFICATION-----------------------------------


# function to balance data
def balanceData(data):
    # Separate majority and minority classes
    majority_class = data[data["target"] == 1]
    minority_class = data[data["target"] == 0]

    # Oversample the minority class to match the majority class size
    minority_class = resample(
        minority_class,
        replace=True,  # Sample with replacement
        n_samples=len(majority_class),  # Match majority class size
        random_state=62,
    )  # For reproducibility

    # Combine the oversampled minority class with the majority class
    data = pd.concat([majority_class, minority_class])

    # Shuffle the balanced dataset
    data = data.sample(frac=1, random_state=62).reset_index(drop=True)

    # Print the balanced dataset for verification
    return data


# function to split the data to (training and validation) while preserving the proportion of a specific target in the
# dataset
def splitDataToTrainingAndValidationForClassification(data, feature):
    # split=1 returns 1 training set and one validation set
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    # for loops based on number of splits
    for train_index, validation_index in split.split(data, data[feature]):
        split_train_set = data.loc[train_index]
        split_validation_set = data.loc[validation_index]

    return split_train_set, split_validation_set


# function to split data to features and target
def splitToFeaturesAndTargetClassification(data):
    X = data.drop("target", axis=1).values.tolist()
    Y = data["target"].values.tolist()
    return X, Y


## function to perform one hot encoding on a specific column
def one_hot_encode(columnData):
    # Step 1: find all unique values in the column
    unique_categories = list(set(columnData))
    # gives an index to each values
    category_to_index_map = {
        category: idx for idx, category in enumerate(unique_categories)
    }

    # Step 2: perform one hot encoding
    one_hot_encoded = []
    for item in columnData:
        # Create a zero vector with length equal to the number of unique categories
        one_hot_vector = [0] * len(unique_categories)
        #  Set the correct index to 1
        one_hot_vector[category_to_index_map[item]] = 1
        # Append the one-hot vector to the result list
        one_hot_encoded.append(one_hot_vector)

    return one_hot_encoded, category_to_index_map


# Perform Preprocessing on the data
# 1. removing id
# 2. one hot encoding on each column
# 3. split the data to training and validation
# 4. split training data to X and Y
# 5. split validation data to X and Y
def preprocessClassificationData(data):
    # remove the id column
    data = removeId(data)

    # apply one-hot encoding on all the columns except the first column which is the target
    columns_to_encode = data.columns[1:]
    encoded_columns = {}
    for col in columns_to_encode:
        one_hot_encoded, category_to_index_map = one_hot_encode(data[col])
        encoded_columns[col] = pd.DataFrame(
            one_hot_encoded,
            columns=[f"{col}_{val}" for val in category_to_index_map.keys()],
        )

    # Combine one-hot encoded data with the target column
    one_hot_encoded_data = pd.concat(
        [data["target"]] + [encoded_columns[col] for col in columns_to_encode], axis=1
    )
    print("one hot encoded data: ", one_hot_encoded_data.shape)

    # split the data to training and validation
    # split_train_set, split_validation_set = (
    #     splitDataToTrainingAndValidationForClassification(
    #         one_hot_encoded_data, "target"
    #     )
    # )

    # split the training data to features and target
    train_X, train_Y = splitToFeaturesAndTargetClassification(one_hot_encoded_data)

    # split the validation data to features and target
    # validation_X, validation_Y = splitToFeaturesAndTargetClassification(
    #     split_validation_set
    # )

    # returning train_X,train_Y, Val_X,Val_Y
    return (
        np.array(train_X, dtype=np.float32),
        np.array(train_Y).reshape(-1, 1),
        # np.array(validation_X, dtype=np.float32),
        # np.array(validation_Y).reshape(-1, 1),
    )


# custom function to give a full report for classification
# takes the true values of the target and the predicted values
# it gives the confusion matrix and accuracy, precision,recall,F1
def customClassificationReport(trueValue, predictedValues):
    print(
        "Classification report:\n",
        metrics.classification_report(trueValue, predictedValues, zero_division=0),
    )

    cm = confusion_matrix(y_true=trueValue, y_pred=predictedValues)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    disp.figure_.suptitle("Confusion Matrix")
    plt.show()

    print(
        "Accuracy: ",
        str(accuracy_score(trueValue, predictedValues))[:4],
    )
    print(
        "Precision: ",
        str(
            precision_score(
                trueValue, predictedValues, average="weighted", zero_division=0
            )
        )[:4],
    )
    print(
        "Recall: ",
        str(
            recall_score(
                trueValue, predictedValues, average="weighted", zero_division=0
            )
        )[:4],
    )
    print(
        "F1: ",
        str(f1_score(trueValue, predictedValues, average="weighted", zero_division=0))[
        :4
        ],
    )
    return mean_squared_error(trueValue, predictedValues)


# Accuracy scoring function
def accuracy_score_custom_for_grid_search(nn_model, X, y):
    predictions = nn_model.predict(X)
    return np.mean(predictions == y)


def custom_cross_validation_classification(
    model,
    X_tr,
    y_tr,
    epoch,
    batch_size,
    num_folds=5,
):
    X_tr, y_tr = np.array(X_tr), np.array(y_tr)

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_history = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_tr, y_tr)):
        print(f"Fold {fold + 1}/{num_folds}")

        X_train, X_test = X_tr[train_idx], X_tr[test_idx]
        y_train, y_test = y_tr[train_idx], y_tr[test_idx]

        model.reset_weights()

        history = model.fit(
            X_train,
            y_train.reshape(-1, 1),
            X_test,
            y_test,
            epochs=epoch,
            batch_size=batch_size,
        )

        predictions = model.predict(X_test)
        accuracy = np.mean(predictions.flatten() == y_test.flatten())

        fold_accuracies.append(accuracy)
        fold_history.append(history)

    # Estrazione delle metriche
    all_val_losses = [history["val_loss"] for history in fold_history]
    all_val_accuracies = [history["val_acc"] for history in fold_history]

    # Trova la lunghezza minima tra le epoche delle fold
    min_epochs = min(len(vl) for vl in all_val_losses)

    # Troncamento per uniformare le lunghezze
    all_val_losses = [vl[:min_epochs] for vl in all_val_losses]
    all_val_accuracies = [va[:min_epochs] for va in all_val_accuracies]

    # Calcolo della media su tutte le epoche
    mean_history = {
        "val_loss": np.mean(np.array(all_val_losses), axis=0).tolist(),
        "val_acc": np.mean(np.array(all_val_accuracies), axis=0).tolist(),
        "epoch": list(range(1, min_epochs + 1))
    }

    mean_accuracy = np.mean(fold_accuracies)

    return mean_accuracy, fold_accuracies, mean_history



