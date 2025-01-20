import numpy as np
from matplotlib import pyplot as plt

from project.CustomNN import CustomNeuralNetwork
import pandas as pd
from sklearn.utils import resample
from project.utility.Enum import RegularizationType, ActivationType, TaskType
from project.utility.Search import Search
from project.utility.utility import (
    balanceData,
    one_hot_encode,
    customClassificationReport,
    removeId,
    accuracy_score_custom_for_grid_search,
    preprocessClassificationData,
    splitToFeaturesAndTargetClassification,
)

if __name__ == "__main__":
    monk3_train = "monk/monks-3.train"
    monk3_test = "monk/monks-3.test"

    # Load training data
    monk3_train_data = pd.read_csv(
        monk3_train,
        sep=" ",
        header=None,
    )
    # Drop the first column (not needed for training)
    monk3_train_data = monk3_train_data.drop(monk3_train_data.columns[0], axis=1)
    # Rename columns according to dataset description
    monk3_train_data.rename(
        columns={
            1: "target",
            2: "a1",
            3: "a2",
            4: "a3",
            5: "a4",
            6: "a5",
            7: "a6",
            8: "ID",
        },
        inplace=True,
    )

    # Load testing data
    monk3_test_data = pd.read_csv(
        monk3_test,
        sep=" ",
        header=None,
    )
    # Drop the first column (not needed for testing)
    monk3_test_data = monk3_test_data.drop(monk3_test_data.columns[0], axis=1)
    # Rename columns according to dataset description
    monk3_test_data.rename(
        columns={
            1: "target",
            2: "a1",
            3: "a2",
            4: "a3",
            5: "a4",
            6: "a5",
            7: "a6",
            8: "ID",
        },
        inplace=True,
    )

    # Print loaded data for verification
    print("MONK3")
    print("Train data")
    print(monk3_train_data.head())
    print("Test data")
    print(monk3_test_data.head())

    # balancing data cause the target column is not balanced
    monk3_train_data = balanceData(monk3_train_data)

    # --------------------------------------------------MONK3-----------------------------------------------------------
    print("-------------------------------------")
    print("\nReal Testing\n")

    # reshape train_X, train_Y, validation_X
    monk3_train_X, monk3_train_Y, monk3_validation_X, monk3_validation_Y = (
        preprocessClassificationData(monk3_train_data)
    )

    monk3_train_X = monk3_train_X.reshape(monk3_train_X.shape[0], -1)
    monk3_train_Y = monk3_train_Y.reshape(-1, 1)

    monk3_validation_X = np.array(monk3_validation_X)
    monk3_validation_Y = np.array(monk3_validation_Y)
    monk3_validation_X = monk3_validation_X.reshape(monk3_validation_X.shape[0], -1)

    print(f"val X shape: {monk3_validation_X.shape}")

    # reshape train_X, train_Y, validation_X
    X = monk3_train_X.reshape(monk3_train_X.shape[0], -1)
    y = monk3_train_Y.reshape(-1, 1)

    monk3_validation_X = np.array(monk3_validation_X)
    monk3_validation_Y = np.array(monk3_validation_Y)
    monk3_validation_X = monk3_validation_X.reshape(monk3_validation_X.shape[0], -1)

    print(f"train X shape: {X.shape[1]}")
    print(
        "Nomi delle colonne di X:",
        X.columns if hasattr(X, "columns") else "X non è un DataFrame",
    )

    print(f"train Y shape: {y.shape}")

    print(f"val X shape: {monk3_validation_X.shape}")

    # Define the parameter grid
    param_grid = {
        "learning_rate": [0.1 / (10**i) for i in range(10)],
        "momentum": [x / 100 for x in range(80, 90)],
        "lambd": [x / 100000000 for x in range(1, 10)],
    }
    # Best Parameters: {'learning_rate': 0.1, 'momentum': 0.8, 'lambd': 1e-08},
    # Initialize the Search class for grid search
    search = Search(CustomNeuralNetwork, param_grid, accuracy_score_custom, activation_type=ActivationType.TANH,
                    regularization_type=RegularizationType.L2, task_type=TaskType.CLASSIFICATION, nesterov=False,
                    decay=0.0001)

    # Perform grid search on the learning rate
    print("Performing Grid Search...")
    best_params, best_score = search.grid_search_classification(
        X, y, epoch=100, batchSize=16, neurons=[4], output_size=1
    )
    print(f"Best Parameters:\n {best_params}, Best Score: {best_score}")

    # Define the network with dynamic hidden layers
    nn3 = CustomNeuralNetwork(input_size=X.shape[1],
                              hidden_layers=[2],
                              output_size=1,
                              activationType=ActivationType.TANH,
                              learning_rate=best_params['learning_rate'],
                              momentum=best_params['momentum'],
                              lambd=best_params['lambd'],
                              regularizationType=RegularizationType.L2,
                              taskType=TaskType.CLASSIFICATION,
                              nesterov=False,
                              decay=0.0001
                              )
    nn3 = CustomNeuralNetwork(
        input_size=X.shape[1],
        hidden_layers=[4],
        output_size=1,
        activationType=ActivationType.SIGMOID,
        learning_rate=best_params["learning_rate"],
        momentum=best_params["momentum"],
        lambd=best_params["lambd"],
        regularizationType=RegularizationType.L2,
        task_type=TaskType.CLASSIFICATION,
    )

    # Train the network
    history = nn3.fit(
        X, y, monk3_validation_X, monk3_validation_Y, epochs=100, batch_size=16
    )

    # Plot a single graph with Loss and Training Accuracy
    plt.figure()

    # Plot Training Loss
    plt.plot(
        history["epoch"],
        history["train_loss"],
        label="Training Loss",
        color="blue",
        linestyle="-",
    )

    # Plot Training Accuracy
    plt.plot(
        history["epoch"],
        history["train_acc"],
        label="Training Accuracy",
        color="orange",
        linestyle="--",
    )

    # Plot Validation Loss
    plt.plot(
        history["epoch"],
        history["val_loss"],
        label="Validation Loss",
        color="yellow",
        linestyle="-",
    )

    # Plot Validation Accuracy
    plt.plot(
        history["epoch"],
        history["val_acc"],
        label="Validation Accuracy",
        color="green",
        linestyle="--",
    )

    # Configure the plot
    plt.xlabel("Epochs")  # X-axis as the recorded epochs
    plt.ylabel("Value")  # Shared y-axis label
    plt.title("MONK3 - Training Loss and Accuracy Over Recorded Epochs")
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()

    # Validation predictions
    print("Predicting validation set")
    monk3_validation_nn_predictions = nn3.predict(monk3_validation_X)
    customClassificationReport(monk3_validation_Y, monk3_validation_nn_predictions)

    # -------------------------------------------------TEST------------------------------------------------------------
    # Rimuovi l'ID dal dataset
    monk3_test_data = removeId(monk3_test_data)

    # Applicazione del One-Hot Encoding
    columns_to_encode = monk3_test_data.columns[1:]  # Escludi la colonna 'target'
    encoded_columns = {}
    category_mappings = {}

    for col in columns_to_encode:
        one_hot_encoded, mapping = one_hot_encode(monk3_test_data[col])
        encoded_columns[col] = pd.DataFrame(
            one_hot_encoded
        )  # Assicurati che sia un DataFrame
        category_mappings[col] = mapping

    # Concatenazione delle colonne codificate con la colonna target
    encoded_columns_df = pd.concat(encoded_columns.values(), axis=1)
    one_hot_test_monk3 = pd.concat(
        [monk3_test_data["target"], encoded_columns_df], axis=1
    )

    # Verifica che tutte le colonne abbiano la stessa lunghezza
    assert all(
        encoded_columns_df[col].shape[0] == len(one_hot_test_monk3)
        for col in encoded_columns_df.columns
    ), "Le colonne codificate non hanno la stessa lunghezza!"

    monk3_real_test_X, monk3_real_test_Y = splitToFeaturesAndTargetClassification(
        one_hot_test_monk3
    )

    # Conversione a numpy array
    try:
        monk3_real_test_X = np.array(
            monk3_real_test_X, dtype=np.float64
        )  # Assicurati che siano numerici
    except ValueError as e:
        print("Errore nella conversione dei dati di Features in array numpy:", e)

    monk3_real_test_Y = np.array(monk3_real_test_Y, dtype=np.float64)

    if monk3_real_test_X.ndim == 1:
        monk3_real_test_X = monk3_real_test_X.reshape(-1, 1)

    # Stampa delle dimensioni per debug
    print(f"Train X shape: {monk3_real_test_X.shape}")
    print(f"Train Y shape: {monk3_real_test_Y.shape}")

    monk3_real_test_predictions_nn = nn3.predict(monk3_real_test_X)
    customClassificationReport(monk3_real_test_Y, monk3_real_test_predictions_nn)
