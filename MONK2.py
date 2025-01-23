import numpy as np
from matplotlib import pyplot as plt

from project.CustomNN import CustomNeuralNetwork
import pandas as pd
from sklearn.utils import resample
from project.utility.Enum import RegularizationType, ActivationType, TaskType, InitializationType
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
    monk2_train = "monk/monks-2.train"
    monk2_test = "monk/monks-2.test"

    # Load training data
    monk2_train_data = pd.read_csv(
        monk2_train,
        sep=" ",
        header=None,
    )
    # Drop the first column (not needed for training)
    monk2_train_data = monk2_train_data.drop(monk2_train_data.columns[0], axis=1)
    # Rename columns according to dataset description
    monk2_train_data.rename(
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
    monk2_test_data = pd.read_csv(
        monk2_test,
        sep=" ",
        header=None,
    )
    # Drop the first column (not needed for testing)
    monk2_test_data = monk2_test_data.drop(monk2_test_data.columns[0], axis=1)
    # Rename columns according to dataset description
    monk2_test_data.rename(
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
    print("MONK2")
    print("Train data")
    print(monk2_train_data.head())
    print("Test data")
    print(monk2_test_data.head())

    # balancing data cause the target column is not balanced
    print("Class distribution before balancing:", monk2_train_data["target"].value_counts())
    monk2_train_data = balanceData(monk2_train_data)
    print("Class distribution after balancing:", monk2_train_data["target"].value_counts())

    # --------------------------------------------------MONK2-----------------------------------------------------------

    # reshape train_X, train_Y, validation_X
    monk2_train_X, monk2_train_Y, monk2_validation_X, monk2_validation_Y = (
        preprocessClassificationData(monk2_train_data)
    )

    monk2_train_X = monk2_train_X.reshape(monk2_train_X.shape[0], -1)
    monk2_train_Y = monk2_train_Y.reshape(-1, 1)

    monk2_validation_X = np.array(monk2_validation_X)
    monk2_validation_Y = np.array(monk2_validation_Y)
    monk2_validation_X = monk2_validation_X.reshape(monk2_validation_X.shape[0], -1)

    print(f"val X shape: {monk2_validation_X.shape}")

    # reshape train_X, train_Y, validation_X
    X = monk2_train_X.reshape(monk2_train_X.shape[0], -1)
    y = monk2_train_Y.reshape(-1, 1)

    monk2_validation_X = np.array(monk2_validation_X)
    monk2_validation_Y = np.array(monk2_validation_Y)
    monk2_validation_X = monk2_validation_X.reshape(monk2_validation_X.shape[0], -1)

    print(f"train X shape: {X.shape[1]}")
    print(
        "Nomi delle colonne di X:",
        X.columns if hasattr(X, "columns") else "X non è un DataFrame",
    )

    print(f"train Y shape: {y.shape}")

    print(f"val X shape: {monk2_validation_X.shape}")

    # Define the parameter grid
    param_grid = {
        "learning_rate": [0.5],
        "momentum": [0.8],
        "lambd": [0.0],
        "decay": [0.0],
        "dropout": [0],
        "batch_size": [2]
    }
    # Best Parameters: {'learning_rate': 0.1, 'momentum': 0.9, 'lambd': 0.0, batch_size}, Best Score: 1.0000

    # Initialize the Search class for grid search
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

    # Perform grid search on the learning rate
    print("Performing Grid Search...")
    best_params, best_score = search.grid_search_classification(
        X, y, epoch=200, neurons=[3], output_size=1,
    )
    print(f"Best Parameters:\n {best_params}, Best Score: {best_score}")

    # Define the network with dynamic hidden layers
    nn2 = CustomNeuralNetwork(
        input_size=X.shape[1],
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

    # Unisci il training set e il validation set
    X_final_train = np.vstack((monk2_train_X, monk2_validation_X))
    Y_final_train = np.vstack((monk2_train_Y, monk2_validation_Y))

    print(f"Final training data shape: {X_final_train.shape}")
    print(f"Final training labels shape: {Y_final_train.shape}")

    # Re-addestra la rete neurale sull'intero set di dati
    history_final = nn2.fit(
        X_final_train, Y_final_train, X_final_train, Y_final_train,  # Utilizzo di tutto il dataset
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
    monk2_validation_nn_predictions = nn2.predict(monk2_validation_X)
    customClassificationReport(monk2_validation_Y, monk2_validation_nn_predictions)

    # -------------------------------------------------TEST------------------------------------------------------------

    print("-------------------------------------")
    print("\nReal Testing\n")

    # Rimuovi l'ID dal dataset
    monk2_test_data = removeId(monk2_test_data)

    # Applicazione del One-Hot Encoding
    columns_to_encode = monk2_test_data.columns[1:]  # Escludi la colonna 'target'
    encoded_columns = {}
    category_mappings = {}

    for col in columns_to_encode:
        one_hot_encoded, mapping = one_hot_encode(monk2_test_data[col])
        encoded_columns[col] = pd.DataFrame(
            one_hot_encoded
        )  # Assicurati che sia un DataFrame
        category_mappings[col] = mapping

    # Concatenazione delle colonne codificate con la colonna target
    encoded_columns_df = pd.concat(encoded_columns.values(), axis=1)
    one_hot_test_monk2 = pd.concat(
        [monk2_test_data["target"], encoded_columns_df], axis=1
    )

    # Verifica che tutte le colonne abbiano la stessa lunghezza
    assert all(
        encoded_columns_df[col].shape[0] == len(one_hot_test_monk2)
        for col in encoded_columns_df.columns
    ), "Le colonne codificate non hanno la stessa lunghezza!"

    monk2_real_test_X, monk2_real_test_Y = splitToFeaturesAndTargetClassification(
        one_hot_test_monk2
    )

    # Conversione a numpy array
    try:
        monk2_real_test_X = np.array(
            monk2_real_test_X, dtype=np.float64
        )  # Assicurati che siano numerici
    except ValueError as e:
        print("Errore nella conversione dei dati di Features in array numpy:", e)

    monk2_real_test_Y = np.array(monk2_real_test_Y, dtype=np.float64)

    if monk2_real_test_X.ndim == 1:
        monk2_real_test_X = monk2_real_test_X.reshape(-1, 1)

    # Stampa delle dimensioni per debug
    print(f"Train X shape: {monk2_real_test_X.shape}")
    print(f"Train Y shape: {monk2_real_test_Y.shape}")

    monk2_real_test_predictions_nn = nn2.predict(monk2_real_test_X)
    mse_test = customClassificationReport(monk2_real_test_Y, monk2_real_test_predictions_nn)
    mse_train = history_final['train_loss']
    mse_train = np.mean(mse_train)

    print(f"MSE(TR) : {mse_train}, MSE(TS): {mse_test}")
