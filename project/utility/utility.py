from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedShuffleSplit, train_test_split
import numpy as np
import pandas as pd
from sklearn import metrics
from project.utility.Enum import TaskType
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay

# function to remove the id column from the data
def removeId(data):
    return data.drop('ID', axis=1, errors='ignore')

#----------------------------CLASSIFICATION-----------------------------------

def custom_cross_validation_class(model, X, y, epoch=None, num_folds=5):
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

def preprocessClassData(data):
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
    Y = data['target'].values.tolist()
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

# Scoring function for the neural network
def accuracy_score_custom(nn_model, X, y):
    predictions = nn_model.predict(X)
    predictions = (predictions > 0.5).astype(int)
    return np.mean(predictions == y)


#----------------------------REGRESSION-----------------------------------

def custom_cross_validation_regr(model, X, y, epoch=None, num_folds=5, metric='mse'):
    """
    Perform k-fold cross-validation for a regression model.

    Parameters:    
    - model: model object.
    - X: input features.
    - y: target values.
    - num_folds: Number of cross-validation folds.
    - epoch: number of epochs for training.
    - metric: evaluation metric ('mse' or 'mae'). Default is 'mse'.
    
    Returns:
    - fold_scores: List of scores (MSE or MAE) for each fold.
    - mean_score: Mean score across all folds.
    """

    X, y = np.array(X), np.array(y)

    # Initialize k-fold (not StratifiedKFold, since it's a regression problem)
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    fold_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}/{num_folds}")
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print('Train size:', len(X_train))
        print('Test size:', len(X_test))

        # Train the model
        if epoch is not None:
            model.fit(X_train, y_train.reshape(-1, 3), epochs=epoch)
        else:
            model.fit(X_train, y_train.reshape(-1, 3))  
        
        # Evaluate the model on the test set
        predictions = model.predict(X_test)

        # Calculate the chosen regression metric
        if metric == 'mse':
            score = mean_squared_error(y_test, predictions)
        elif metric == 'mae':
            score = mean_absolute_error(y_test, predictions)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        print(f"Fold {fold + 1} {metric.upper()}: {score:.4f}")
        print('--------------------------------------------')
        fold_scores.append(score)

    # Calculate mean score
    mean_score = np.mean(fold_scores)

    # Return fold scores and the mean score
    return mean_score, fold_scores

def customRegressionReport(trueValues, predictedValues, target_names):
    # Print individual regression metrics
    mae = mean_absolute_error(trueValues, predictedValues)
    mse = mean_squared_error(trueValues, predictedValues)
    rmse = mse ** 0.5
    r2 = r2_score(trueValues, predictedValues)
    
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    
    # Visualization: scatter plots for predictions vs true values
    if target_names is not None:
        for i, target_name in enumerate(target_names):
            plt.figure(figsize=(6, 6))
            # Min and max for scatter plots
            min_val = min(trueValues[:, i].min(), predictedValues[:, i].min())
            max_val = max(trueValues[:, i].max(), predictedValues[:, i].max())
            # Plot true values
            plt.scatter(trueValues[:, i], trueValues[:, i], alpha=0.5, color="green", label="True Values")
            # Plot predicted values
            plt.scatter(trueValues[:, i], predictedValues[:, i], alpha=0.5, color="blue")
            # Line y = x
            plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")  
            plt.xlabel("True Values")
            plt.ylabel("Predicted Values")
            plt.title(f"True vs Predicted: {target_name}")
            plt.grid()
            plt.show()
    else:
        # General scatter plot if no target names are provided
        plt.figure(figsize=(6, 6))
        # Min and max for scatter plots
        min_val = min(trueValues.min(), predictedValues.min())
        max_val = max(trueValues.max(), predictedValues.max())
        plt.scatter(trueValues, predictedValues, alpha=0.5, color="blue")
        plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")  # Line y = x
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title("True vs Predicted")
        plt.grid()
        plt.show()

def preprocessRegrData(data, target_columns=['TARGET_x', 'TARGET_y', 'TARGET_z']):
    #remove the id column
    data = removeId(data)

    #split the data to training and validation
    split_train_set, split_validation_set = splitRegrData(data, target_columns, test_size=0.2, random_state=42)
    #split the data to features and target
    train_X, train_Y = splitToFeaturesAndTargetRegr(split_train_set, target_columns)

    #validation set
    validation_X, validation_Y = splitToFeaturesAndTargetRegr(split_validation_set, target_columns)
  
    return np.array(train_X), np.array(train_Y), np.array(validation_X), np.array(validation_Y)

def splitRegrData(data, target_columns, test_size=0.2, random_state=42):
    """
    Split data into training and validation sets for multi-target regression.
    
    Parameters:
    - data: Input DataFrame with features and targets.
    - target_columns: List of target columns (e.g., ['TARGET_x', 'TARGET_y', 'TARGET_z']).
    - test_size: Proportion of the dataset to include in the validation set.
    - random_state: Seed for reproducibility.

    Returns:
    - split_train_set: Training set.
    - split_validation_set: Validation set.
    """
    
    # Extract features (X) and targets (y)
    X = data.drop(columns=target_columns)
    y = data[target_columns]

    # Perform a simple train-test split (without stratification for continuous targets)
    split_train_set, split_validation_set = train_test_split(
        data, test_size=test_size, random_state=random_state, stratify=None
    )

    return split_train_set, split_validation_set

# function to split data to features and target
def splitToFeaturesAndTargetRegr(data, target_columns):
    X = data.drop(target_columns, axis=1).values.tolist()
    Y = data[target_columns].values.tolist()
    return X,Y


