import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from project.CustomNN import CustomNeuralNetwork
from project.utility.Enum import RegularizationType, ActivationType, TaskType, InitializationType
from project.utility.Search import Search
from project.utility.utility import (
    balanceData,
    one_hot_encode,
    customClassificationReport,
    removeId,
    preprocessClassificationData,
    splitToFeaturesAndTargetClassification,
    min_max_scaling,
)

if __name__ == "__main__":
    monk2_train = "monk/monks-2.train"
    monk2_test = "monk/monks-2.test"

    # Load training data
    monk2_train_data = pd.read_csv(monk2_train, sep=" ", header=None)
    monk2_train_data = monk2_train_data.drop(monk2_train_data.columns[0], axis=1)
    monk2_train_data.rename(columns={
        1: "target", 2: "a1", 3: "a2", 4: "a3", 5: "a4", 6: "a5", 7: "a6", 8: "ID"
    }, inplace=True)

    # Load testing data
    monk2_test_data = pd.read_csv(monk2_test, sep=" ", header=None)
    monk2_test_data = monk2_test_data.drop(monk2_test_data.columns[0], axis=1)
    monk2_test_data.rename(columns={
        1: "target", 2: "a1", 3: "a2", 4: "a3", 5: "a4", 6: "a5", 7: "a6", 8: "ID"
    }, inplace=True)

    print("MONK2")
    print("Train data")
    print(monk2_train_data.head())
    print("Test data")
    print(monk2_test_data.head())

    # Balance the training data
    monk2_train_data = balanceData(monk2_train_data)

    monk2_train_X, monk2_train_Y, monk2_validation_X, monk2_validation_Y = preprocessClassificationData(monk2_train_data)

    monk2_train_X, X_min, X_max = min_max_scaling(monk2_train_X, feature_range=(-1, 1))
    monk2_validation_X = (monk2_validation_X - X_min) / (X_max - X_min + 1e-8)
    monk2_validation_X = monk2_validation_X * (1 - (-1)) + (-1)

    param_grid = {
        "learning_rate": [0.2],
        "momentum": [0.7],
        "lambd": [0.0],
        "decay": [0.0],
        "dropout": [0],
        "batch_size": [1]
    }

    search = Search(
        model=CustomNeuralNetwork,
        param_grid=param_grid,
        activation_type=ActivationType.SIGMOID,
        regularization_type=RegularizationType.L2,
        initialization=InitializationType.GAUSSIAN,
        nesterov=False,
        decay=0.0,
        dropout=0.0
    )

    print("Performing Grid Search...")
    best_params, best_score = search.grid_search_classification(
        monk2_train_X, monk2_train_Y, epoch=200, neurons=[3], output_size=1,
    )

    nn2 = CustomNeuralNetwork(
        input_size=monk2_train_X.shape[1],
        hidden_layers=[3],
        output_size=1,
        activationType=ActivationType.SIGMOID,
        learning_rate=best_params["learning_rate"],
        momentum=best_params["momentum"],
        lambd=best_params["lambd"],
        regularizationType=RegularizationType.L2,
        task_type=TaskType.CLASSIFICATION,
        initialization=InitializationType.GAUSSIAN,
        dropout_rate=best_params["dropout"],
        decay=best_params["decay"],
        nesterov=True
    )

    X_final_train = np.vstack((monk2_train_X, monk2_validation_X))
    Y_final_train = np.vstack((monk2_train_Y, monk2_validation_Y))

    history_final = nn2.fit(
        X_final_train, Y_final_train, X_final_train, Y_final_train,
        epochs=200, batch_size=best_params["batch_size"]
    )

    # Plot Training and Validation Loss (MSE)
    plt.figure()
    plt.plot(history_final["epoch"], history_final["train_loss"], label="Training Loss (MSE)", color="red",
             linestyle="-")
    plt.plot(history_final["epoch"], history_final["val_loss"], label="Validation Loss (MSE)", color="blue",
             linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.title("MONK2 - Final Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Training and Validation Accuracy
    plt.figure()
    plt.plot(history_final["epoch"], history_final["train_acc"], label="Training Accuracy", color="red", linestyle="-")
    plt.plot(history_final["epoch"], history_final["val_acc"], label="Validation Accuracy", color="blue",
             linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("MONK2 - Final Training and Validation Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Validation predictions
    print("Predicting validation set")
    monk1_validation_nn_predictions = nn2.predict(monk2_validation_X)
    customClassificationReport(monk2_validation_Y, monk1_validation_nn_predictions)

    # -------------------------------------------------TEST------------------------------------------------------------

    print("Real Testing")

    # Remove ID from test data
    monk2_test_data = removeId(monk2_test_data)
    columns_to_encode = monk2_test_data.columns[1:]
    encoded_columns = {col: pd.DataFrame(one_hot_encode(monk2_test_data[col])[0]) for col in columns_to_encode}
    one_hot_test_monk2 = pd.concat([monk2_test_data["target"], pd.concat(encoded_columns.values(), axis=1)], axis=1)

    monk2_real_test_X, monk2_real_test_Y = splitToFeaturesAndTargetClassification(one_hot_test_monk2)
    monk2_real_test_X = (monk2_real_test_X - X_min) / (X_max - X_min + 1e-8)
    monk2_real_test_X = monk2_real_test_X * (1 - (-1)) + (-1)
    monk2_real_test_Y = np.array(monk2_real_test_Y, dtype=np.float64)

    monk2_real_test_predictions_nn = nn2.predict(monk2_real_test_X)
    mse_test = customClassificationReport(monk2_real_test_Y, monk2_real_test_predictions_nn)
    mse_train = np.mean(history_final['train_loss'])

    print(f"MSE(TR): {mse_train}, MSE(TS): {mse_test}")
