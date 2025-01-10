from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedShuffleSplit


def custom_cross_validation(model, X, y, epoch=None, num_folds=5, ):
    """
    Perform stratified k-fold cross-validation

    Parameters:    - model: model object.
    - X: samples.
    - y: target.
    - num_folds: Number of cross-validation folds.
    - epoch: number of epoch
    Returns:
    - fold_accuracies: List of accuracy scores for each fold.
    - mean_accuracy: Mean accuracy across all folds.
    """

    X, y = np.array(X), np.array(y)

    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    fold_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold + 1}/{num_folds}")

        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print('train size: ', len(X_train))
        print('test size: ', len(X_test))

        # Train the model
        if epoch is not None:
            model.fit(X_train, y_train.reshape(-1, 1), epochs=epoch)
        else:
            model.fit(X_train, y_train.reshape(-1, 1))

            # Evaluate the model on the test set
        predictions = model.predict(X_test)
        accuracy = np.mean(predictions.flatten() == y_test.flatten())

        print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")
        print('--------------------------------------------')
        fold_accuracies.append(accuracy)

    # Calculate mean accuracy
    mean_accuracy = np.mean(fold_accuracies)

    # Return fold accuracies and the mean accuracy
    return mean_accuracy, fold_accuracies


def customClassificationReport(trueValue, predictedValues):
    print("Classification report:\n", metrics.classification_report(trueValue, predictedValues, zero_division=0))

    cm = confusion_matrix(y_true=trueValue, y_pred=predictedValues)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    disp.figure_.suptitle("Confusion Matrix")
    plt.show()

    print('Accuracy: ', accuracy_score(trueValue, predictedValues))
    print('Precision: ', precision_score(trueValue, predictedValues, average='weighted', zero_division=0))
    print('Recall: ', recall_score(trueValue, predictedValues, average='weighted', zero_division=0))
    print('F1: ', f1_score(trueValue, predictedValues, average='weighted', zero_division=0))


def one_hot_encode(data):
    # Step 1: Trova le categorie uniche
    unique_categories = list(set(data))
    category_to_index = {category: idx for idx, category in enumerate(unique_categories)}

    # Step 2: Crea i vettori one-hot
    one_hot_encoded = []
    for item in data:
        # Crea un vettore zero lungo quanto il numero di categorie uniche
        one_hot_vector = [0] * len(unique_categories)
        # Imposta il valore corretto su 1
        one_hot_vector[category_to_index[item]] = 1
        one_hot_encoded.append(one_hot_vector)

    return one_hot_encoded, category_to_index


def preprocessData(data):
    # remove the id column
    data = removeId(data)

    # apply one-hot encoding
    columns_to_encode = data.columns[1:]
    # Perform one-hot encoding for each column
    encoded_columns = {}
    category_mappings = {}
    for col in columns_to_encode:
        one_hot_encoded, mapping = one_hot_encode(data[col])
        encoded_columns[col] = pd.DataFrame(one_hot_encoded,
                                            columns=[f"{col}_{val}" for val in mapping.keys()])
        category_mappings[col] = mapping

    # Combine one-hot encoded data with the target column
    one_hot_monk = pd.concat(
        [data['target']] + [encoded_columns[col] for col in columns_to_encode],
        axis=1
    )

    # split the data to training and validation
    split_train_set, split_validation_set = splitData(one_hot_monk, 'target')

    # split the data to features and target
    train_X, train_Y = splitToFeaturesAndTarget(split_train_set)

    # validation set
    validation_X, validation_Y = splitToFeaturesAndTarget(split_validation_set)

    return np.array(train_X, dtype=np.float32), np.array(train_Y).reshape(-1, 1), \
        np.array(validation_X, dtype=np.float32), np.array(validation_Y).reshape(-1, 1)


# function to split data to features and target
def splitToFeaturesAndTarget(data):
    X = data.drop('target', axis=1).values.tolist()
    Y = data["target"].values.tolist()
    return X, Y


# function to split the data to training and validation while preserving the proportion of a specific target in the
# dataset
def splitData(data, feature):
    # split=1 returns 1 training set and one validation set
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    # for loops based on number of splits
    for train_index, validation_index in split.split(data, data[feature]):
        split_train_set = data.loc[train_index]
        split_validation_set = data.loc[validation_index]
    return split_train_set, split_validation_set


# function to remove the id column from the data
def removeId(data):
    return data.drop('ID', axis=1, errors='ignore')


# Scoring function for the neural network
def accuracy_score_custom(nn_model, X, y):
    predictions = nn_model.predict(X)
    predictions = (predictions > 0.5).astype(int)
    return np.mean(predictions == y)
