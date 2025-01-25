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
    Applies Min-Max scaling to a numpy array.

    Args:
        X (numpy.ndarray): Input data (2D array).
        feature_range (tuple): Desired scaling range (default: [-1, 1]).

    Returns:
        numpy.ndarray: Scaled data in the specified range.
        numpy.ndarray: Original minimum values of the features.
        numpy.ndarray: Original maximum values of the features.
    """
    min_val, max_val = feature_range  # Target range

    X_min = np.min(X, axis=0)  # Minimum values of the columns (features)
    X_max = np.max(X, axis=0)  # Maximum values of the columns (features)

    # Avoid division by zero in case of constant features
    X_scaled = (X - X_min) / (X_max - X_min + 1e-8)  # Normalize to [0, 1]
    # Rescale to the desired range
    X_scaled = X_scaled * (max_val - min_val) + min_val
    return X_scaled, X_min, X_max


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


# Perform Preprocessing on Training data
# 1. removing id
# 2. one hot encoding on each column
# 3. split training data to X and Y
def preprocessTrainingClassificationData(trainData):
    # remove the id column
    trainData = removeId(trainData)

    # apply one-hot encoding on all the columns except the first column which is the target
    columns_to_encode = trainData.columns[1:]
    encoded_columns = {}
    for col in columns_to_encode:
        one_hot_encoded, category_to_index_map = one_hot_encode(trainData[col])
        encoded_columns[col] = pd.DataFrame(
            one_hot_encoded,
            columns=[f"{col}_{val}" for val in category_to_index_map.keys()],
        )

    # Combine one-hot encoded data with the target column
    one_hot_encoded_data = pd.concat(
        [trainData["target"]] + [encoded_columns[col] for col in columns_to_encode],
        axis=1,
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


# Perform Preprocessing on Testing data
# 1. removing id
# 2. one hot encoding on each column
# 3. split training data to X and Y
def preprocessTestingClassificationData(testData):
    # Remove ID from test data
    testData = removeId(testData)

    # Apply one-hot encoding to test data
    columns_to_encode = testData.columns[1:]  # Exclude 'target'
    encoded_columns = {
        col: pd.DataFrame(one_hot_encode(testData[col])[0]) for col in columns_to_encode
    }
    # Combine one-hot encoded data with the target column
    one_hot_test_monk1 = pd.concat(
        [testData["target"], pd.concat(encoded_columns.values(), axis=1)], axis=1
    )
    # split the testing data to features and target
    test_X, test_Y = splitToFeaturesAndTargetClassification(one_hot_test_monk1)

    # returning test_X,test_Y,
    return (
        np.array(test_X, dtype=np.float64),
        np.array(test_Y, dtype=np.float64),
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
    plt.show(block=False)

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


def custom_cross_validation_classification(
    model,
    X_tr,
    y_tr,
    epoch,
    batch_size,
    num_folds=5,
):
    """
    Perform custom k-fold cross-validation for a classification model.

    Parameters:
    -----------
    model : object
        The neural network model to be trained and evaluated.
    X_tr : array-like
        Training input data.
    y_tr : array-like
        Training target labels.
    epoch : int
        Number of epochs to train the model.
    batch_size : int
        Size of each mini-batch for training.
    num_folds : int, optional (default=5)
        Number of folds for cross-validation.

    Returns:
    --------
    mean_accuracy : float
        Mean accuracy across all folds.
    fold_accuracies : list of float
        List of accuracies for each fold.
    mean_history : dict
        Dictionary containing the mean validation loss and accuracy across folds for each epoch.
    """
    # Convert input data to numpy arrays for consistency
    X_tr, y_tr = np.array(X_tr), np.array(y_tr)

    # Initialize StratifiedKFold for cross-validation
    # StratifiedKFold ensures that each fold has the same proportion of classes as the original dataset
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Lists to store accuracies and training history for each fold
    fold_accuracies = []
    fold_history = []

    # Iterate over each fold
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_tr, y_tr)):
        print(f"Fold {fold + 1}/{num_folds}")

        # Split the data into training and testing sets for the current fold
        X_train, X_test = X_tr[train_idx], X_tr[test_idx]
        y_train, y_test = y_tr[train_idx], y_tr[test_idx]

        # Reset the model's weights to ensure training starts from scratch for each fold
        model.reset_weights()

        # Train the model on the training data for the current fold
        history = model.fit(
            X_train,
            y_train.reshape(
                -1, 1
            ),  # Reshape y_train to match model's expected input shape
            X_test,
            y_test,
            epochs=epoch,
            batch_size=batch_size,
        )

        # Make predictions on the test set for the current fold
        predictions = model.predict(X_test)

        # Calculate accuracy for the current fold
        accuracy = np.mean(predictions.flatten() == y_test.flatten())

        # Store the accuracy and training history for the current fold
        fold_accuracies.append(accuracy)
        fold_history.append(history)

    # Extract validation metrics from all folds
    all_val_losses = [history["test_loss"] for history in fold_history]
    all_val_accuracies = [history["test_acc"] for history in fold_history]

    # Find the minimum number of epochs across all folds
    # This ensures that all folds have the same number of epochs for averaging
    min_epochs = min(len(vl) for vl in all_val_losses)

    # Truncate the validation losses and accuracies to the minimum number of epochs
    all_val_losses = [vl[:min_epochs] for vl in all_val_losses]
    all_val_accuracies = [va[:min_epochs] for va in all_val_accuracies]

    # Calculate the mean validation loss and accuracy across all folds for each epoch
    mean_history = {
        "test_loss": np.mean(
            np.array(all_val_losses), axis=0
        ).tolist(),  # Mean validation loss
        "test_acc": np.mean(
            np.array(all_val_accuracies), axis=0
        ).tolist(),  # Mean validation accuracy
        "epoch": list(range(1, min_epochs + 1)),  # Epoch numbers
    }

    # Calculate the mean accuracy across all folds
    mean_accuracy = np.mean(fold_accuracies)

    return mean_accuracy, fold_accuracies, mean_history
