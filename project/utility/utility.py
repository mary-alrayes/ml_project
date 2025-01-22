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


# -----------------------------------common utility ------------------------------------

# Apply moving average smoothing to mitigate noise introduced by one-hot encoding and class balancing
def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


# function to remove the id column from the data
def removeId(data):
    return data.drop("ID", axis=1, errors="ignore")


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
    split_train_set, split_validation_set = (
        splitDataToTrainingAndValidationForClassification(
            one_hot_encoded_data, "target"
        )
    )

    # split the training data to features and target
    train_X, train_Y = splitToFeaturesAndTargetClassification(split_train_set)

    # split the validation data to features and target
    validation_X, validation_Y = splitToFeaturesAndTargetClassification(
        split_validation_set
    )

    # returning train_X,train_Y, Val_X,Val_Y
    return (
        np.array(train_X, dtype=np.float32),
        np.array(train_Y).reshape(-1, 1),
        np.array(validation_X, dtype=np.float32),
        np.array(validation_Y).reshape(-1, 1),
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
        str(precision_score(
            trueValue, predictedValues, average="weighted", zero_division=0
        ))[:4],
    )
    print(
        "Recall: ",
        str(recall_score(trueValue, predictedValues, average="weighted", zero_division=0))[:4],
    )
    print(
        "F1: ",
        str(f1_score(trueValue, predictedValues, average="weighted", zero_division=0))[:4],
    )


# Accuracy scoring function
def accuracy_score_custom_for_grid_search(nn_model, X, y):
    predictions = nn_model.predict(X)
    return np.mean(predictions == y)


## function to perform cross validation for classification
def custom_cross_validation_classification(
    model,
    X_tr,
    y_tr,
    epoch,
    batch_size,
    num_folds=3,
):
    """
    Perform stratified k-fold cross-validation
    we are performing cross validation (k-fold) on (X_train, y_train) set.

    Parameters:
    - model: model object.
    - X_tr: data.
    - y_tr: target.
    - epoch: number of epochs for the training
    - batch_size: the size of the batch size. Default is 5.
    - num_folds: Number of cross-validation folds.

    Returns:
    - mean_accuracy: Mean accuracy across all folds.
    - fold_accuracies: List of accuracy scores for each fold.

    """

    X_tr, y_tr = np.array(X_tr), np.array(y_tr)

    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    # an array to store the accuracies for each fold
    fold_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_tr, y_tr)):
        print(f"Fold {fold + 1}/{num_folds}")

        # Split (X_train,y_train) data to training and testing for each fold
        X_train, X_test = X_tr[train_idx], X_tr[test_idx]
        y_train, y_test = y_tr[train_idx], y_tr[test_idx]
        print("train size: ", len(X_train))
        print("test size: ", len(X_test))

        # Train the model on the training data
        model.fit(
            X_train,
            y_train.reshape(-1, 1),
            epochs=epoch,
            batch_size=batch_size,
        )

        # Evaluate the model on the test set for each fold
        predictions = model.predict(X_test)
        # Calculate accuracy for the fold
        accuracy = np.mean(predictions.flatten() == y_test.flatten())

        print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")
        print("--------------------------------------------")

        # append the fold accuracy to fold_accuracies
        fold_accuracies.append(accuracy)

    # Calculate mean accuracy
    mean_accuracy = np.mean(fold_accuracies)

    # Return mean accuracy and fold accuracies
    return mean_accuracy, fold_accuracies


# ----------------------------REGRESSION-----------------------------------

def splitDataToTrainingAndValidationForRegression(
    data,
):
    """
    Split data into training and validation sets for multi-target regression.

    Parameters:
    - data: Input DataFrame with features and targets.

    Returns:
    - split_train_set: Training set.
    - split_validation_set: Validation set.
    """

    # Convert to NumPy array
    data_array = np.array(data)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Shuffle the data
    shuffled_indices = np.random.permutation(len(data_array))

    # Calculate the split index
    split_index = int(len(data_array) * (1 - 0.15))

    # Split the data
    train_indices = shuffled_indices[:split_index]
    validation_indices = shuffled_indices[split_index:]

    split_train_set = data_array[train_indices]
    split_validation_set = data_array[validation_indices]

    # Convert to DataFrames
    split_train_set_df = pd.DataFrame(split_train_set, columns=data.columns)
    split_validation_set_df = pd.DataFrame(split_validation_set, columns=data.columns)

    return split_train_set_df, split_validation_set_df

def splitDataToTrainingAndAssessmentForRegression(
    data,
):
    """
    Split data into training and assessment sets for multi-target regression.

    Parameters:
    - data: Input DataFrame with features and targets.

    Returns:
    - split_train_set: Training set.
    - split_assessment_set: Assessment set.
    """

    # Convert to NumPy array
    data_array = np.array(data)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Shuffle the data
    shuffled_indices = np.random.permutation(len(data_array))

    # Calculate the split index
    split_index = int(len(data_array) * (1 - 0.15))

    # Split the data
    train_indices = shuffled_indices[:split_index]
    assessment_indices = shuffled_indices[split_index:]

    split_train_set = data_array[train_indices]
    split_assessment_set = data_array[assessment_indices]

    # Convert to DataFrames
    split_train_set_df = pd.DataFrame(split_train_set, columns=data.columns)
    split_assessment_set_df = pd.DataFrame(split_assessment_set, columns=data.columns)

    return split_train_set_df, split_assessment_set_df


# function to split data to features and target
# pass the name of the target columns
def splitToFeaturesAndTargetRegression(data, target_columns):
    print("data: ", data)
    X = data.drop(target_columns, axis=1).values.tolist()
    Y = data[target_columns].values.tolist()
    return X, Y


## function to perform MinMax normalization
def min_max_normalization(data, min_vals=None, max_vals=None):
    data_normalized = data.copy()
    # Select numeric columns only for normalization
    numeric_data = data_normalized.select_dtypes(include=['float64', 'int64'])

    if min_vals is None or max_vals is None:
        # Calculate min and max from the training data
        min_vals = numeric_data.min()
        max_vals = numeric_data.max()

    # Normalize each numeric column separately using the min-max formula
    normalized_data = (numeric_data - min_vals) / (max_vals - min_vals)

    # Rejoin with non-numeric columns if needed
    non_numeric_data = data.select_dtypes(exclude=['float64', 'int64'])
    final_data = pd.concat([non_numeric_data, normalized_data], axis=1)

    return final_data, min_vals, max_vals


## function to denormalize the predictions values for the ML-CUP24-TS.csv file
def min_max_denormalization(predictions, data, target_columns):
    """
    Denormalizes the predicted values back to the original scale.

    Parameters:
    - predictions: The normalized predicted values.
    - data: The original data (used to get min/max values for denormalization).
    - target_columns: List of target columns (e.g., ['TARGET_x', 'TARGET_y', 'TARGET_z']).
    """

    # Initialize a copy of the predictions array
    denorm_predictions = predictions.copy()

    # Select the columns of interest for denormalization
    target_data = data[target_columns]

    # Denormalize the predictions for each target column
    for idx, target_column in enumerate(target_columns):
        min_value = target_data[target_column].min()
        max_value = target_data[target_column].max()
        denorm_predictions[:, idx] = predictions[:, idx] * (max_value - min_value) + min_value

    return denorm_predictions

def denormalize_zscore(predictions, data, target_columns):
    """
    Denormalizes the predicted values back to the original scale using Z-score normalization.

    Parameters:
    - predictions: The normalized predicted values.
    - means: Means used for normalization.
    - stds: Standard deviations used for normalization.
    """

    # Initialize a copy of the predictions array
    denorm_predictions = predictions.copy()

    # Select the columns of interest for denormalization
    target_data = data[target_columns]
    for idx, target_column in enumerate(target_columns):
        mean = target_data[target_column].mean()
        std = target_data[target_column].std()
        denorm_predictions[:, idx] = predictions[:, idx] * std + mean

    return denorm_predictions

## function to perform Zscore normalization
def zscore_normalization(data, means=None, stds=None):
    # Create a copy of the data to avoid modifying the original DataFrame
    data_normalized = data.copy()

    # Select only numeric columns
    numeric_data = data_normalized.select_dtypes(include=['float64', 'int64'])
    columns_to_normalize = numeric_data.columns

    # Calculate means and standard deviations if not provided
    if means is None or stds is None:
        means = numeric_data.mean(axis=0)
        stds = numeric_data.std(axis=0)

    # Avoid division by zero for constant columns
    stds_replaced = stds.replace(0, 1)

    # Apply Z-score normalization
    data_normalized[columns_to_normalize] = (numeric_data - means) / stds_replaced

    return data_normalized, means, stds


# Perform Preprocessing on the data
# 1. removing id
# 2. applying normalization
# 3. split the data to training and validation
# 4. split training data to X and Y
# 5. split validation data to X and Y
def preprocessRegrData(data, standard, target_columns=["TARGET_x", "TARGET_y", "TARGET_z"]):
    # remove the id column
    data = removeId(data)
    # split the data to training and validation
    split_train_set, split_validation_set = splitDataToTrainingAndValidationForRegression(data)
    # split the data to training and assessment
    split_train_set, split_assessment_set = splitDataToTrainingAndAssessmentForRegression(split_train_set)
    
    # use z-score normalization
    if standard:

        # Normalize the training set and get its means and stds
        train_set, train_means, train_stds = zscore_normalization(split_train_set)
        # Normalize the validation set using the training set's means and stds
        split_validation_set, _, _ = zscore_normalization(split_validation_set, means=train_means, stds=train_stds)
        # Normalize the assessment set using the training set's means and stds
        split_assessment_set, _, _ = zscore_normalization(split_assessment_set, means=train_means, stds=train_stds)
        
        # split the training set to features and target
        train_X, train_Y = splitToFeaturesAndTargetRegression(train_set, target_columns)
        # split the validation set to features and target
        validation_X, validation_Y = splitToFeaturesAndTargetRegression(split_validation_set, target_columns)
        # split the assessment set to features and target
        assessment_X, assessment_Y = splitToFeaturesAndTargetRegression(split_assessment_set, target_columns)

    # use min-max normalization
    else:

        # Normalize the training set
        train_set, train_min, train_max = min_max_normalization(split_train_set)
        # Normalize the validation set using the training set's min and max
        split_validation_set, _, _ = min_max_normalization(split_validation_set, min_vals=train_min, max_vals=train_max)
        #Normalize the assessment set using the training set's min and max
        split_assessment_set, _, _ = min_max_normalization(split_assessment_set, min_vals=train_min, max_vals=train_max)

        # split the training set to features and target
        train_X, train_Y = splitToFeaturesAndTargetRegression(train_set, target_columns)
        # split the validation set to features and target
        validation_X, validation_Y = splitToFeaturesAndTargetRegression(split_validation_set, target_columns)
        # split the assessment set to features and target
        assessment_X, assessment_Y = splitToFeaturesAndTargetRegression(split_assessment_set, target_columns)

    return split_train_set, np.array(train_X), np.array(train_Y), np.array(validation_X), np.array(validation_Y), np.array(assessment_X), np.array(assessment_Y)

def preprocessRegressionTestData(data, test_X, standard=True, target_columns=['TARGET_x', 'TARGET_y', 'TARGET_z']):
    # remove the id column
    test_X = removeId(test_X)

    # Extract feature columns (exclude target columns)
    feature_columns = data.drop(columns=target_columns).columns

    if standard:
        # Normalize the training set and get its means and stds
        data[feature_columns], train_means, train_stds = zscore_normalization(data[feature_columns])

        # Normalize the test set using the training set's means and stds
        test_X[feature_columns], _, _ = zscore_normalization(test_X[feature_columns], means=train_means, stds=train_stds)
    else:
        # Normalize the training set and get its min and max values
        data[feature_columns], train_min, train_max = min_max_normalization(data[feature_columns])

        # Normalize the test set using the training set's min and max values
        test_X[feature_columns], _, _ = min_max_normalization(test_X[feature_columns], min_vals=train_min, max_vals=train_max)

    return np.array(test_X)

# custom function to give a full report for regression
# takes the true values of the target , the predicted values, and the target columns names
# it gives the MAE, MSE, RMSE, R2, MEE
def customRegressionReport(trueValues, predictedValues, target_names):
    # Print individual regression metrics
    mae = mean_absolute_error(trueValues, predictedValues)
    mse = mean_squared_error(trueValues, predictedValues)
    rmse = mse**0.5
    r2 = r2_score(trueValues, predictedValues)
    mee = np.mean(np.sqrt(np.sum((trueValues - predictedValues) ** 2, axis=1)))

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    print(f"Mean Euclidean Error (MEE): {mee:.4f}")

    # Visualization: scatter plots for predictions vs true values
    if target_names is not None:
        for i, target_name in enumerate(target_names):
            plt.figure(figsize=(6, 6))
            # Min and max for scatter plots
            min_val = min(trueValues[:, i].min(), predictedValues[:, i].min())
            max_val = max(trueValues[:, i].max(), predictedValues[:, i].max())
            # Plot true values
            plt.scatter(
                trueValues[:, i],
                trueValues[:, i],
                alpha=0.5,
                color="green",
                label="True Values",
            )
            # Plot predicted values
            plt.scatter(
                trueValues[:, i], predictedValues[:, i], alpha=0.5, color="blue"
            )
            # Line y = x
            plt.plot(
                [min_val, max_val], [min_val, max_val], color="red", linestyle="--"
            )
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
        plt.plot(
            [min_val, max_val], [min_val, max_val], color="red", linestyle="--"
        )  # Line y = x
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title("True vs Predicted")
        plt.grid()
        plt.show()


### function to perform cross validation for regression
def custom_cross_validation_regression(
    model,
    X_tr,
    y_tr,
    epoch,
    batch_size,
    num_folds=5,
    metric=RegressionMetrics.MSE,
):
    """
    Perform k-fold cross-validation for a regression model.

    Parameters:
    - model: model object.
    - X_tr: data.
    - y_tr: target.
    - epoch: number of epochs for the training
    - batch_size: the size of the batch size
    - num_folds: Number of cross-validation folds. Default is 5.
    - metric: evaluation metric ('mse' or 'mae'). Default is 'mse'.

    Returns:
    - mean_score: Mean score across all folds.
    - fold_scores: List of scores (MSE or MAE) for each fold.

    """

    X_train, y_train = np.array(X_tr), np.array(y_tr)

    # Initialize stratified k-fold
    skf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # an array to store the score for each fold
    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_tr)):
        print(f"Fold {fold + 1}/{num_folds}")

        # Split (X_train,y_train) data to training and testing for each fold
        X_train, X_test = X_tr[train_idx], X_tr[test_idx]
        y_train, y_test = y_tr[train_idx], y_tr[test_idx]
        print("train size: ", len(X_tr))
        print("test size: ", len(X_test))

        # Train the model on the training data
        model.fit(
            X_train,
            y_train.reshape(-1, 3),
            epochs=epoch,
            batch_size=batch_size,
        )

        # Evaluate the model on the test set for each fold
        predictions = model.predict(X_test)

        # Calculate the chosen regression metric
        if metric == RegressionMetrics.MSE:
            score = mean_squared_error(y_test, predictions)
        elif metric == RegressionMetrics.MAE:
            score = mean_absolute_error(y_test, predictions)

        print(f"Fold {fold + 1} {metric.value}: {score:.4f}")
        print("--------------------------------------------")
        # append the fold score loss to fold_scores
        fold_scores.append(score)

    # Calculate mean score
    mean_score = np.mean(fold_scores)

    # Return  the mean score and the fold scores
    return mean_score, fold_scores

def averaging_ensemble(models, X):
    """
    Perform an averaging ensemble for regression models.

    Parameters:
    - models: List of trained regression models.
    - X: Input data.

    Returns:
    - predictions: Averaged predictions from the ensemble.
    """

    # Get predictions from all models
    predictions = [model.predict(X) for model in models]

    # Average of the predictions
    averaged_predictions = np.mean(predictions, axis=0)

    return averaged_predictions

# save the results in a csv file
def save_predictions_to_csv(data, file_name):
    data = pd.DataFrame(data, columns=['TARGET_x', 'TARGET_y', 'TARGET_z'])
    data.to_csv(file_name, index=False)