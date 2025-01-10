import numpy as np
from matplotlib import pyplot as plt

from project.CustomNN import CustomNeuralNetwork
import pandas as pd
from sklearn.utils import resample
from project.utility.Enum import RegularizationType, ActivationType
from project.utility.Search import Search
from project.utility.utility import one_hot_encode, customClassificationReport, preprocessData, accuracy_score_custom, \
    removeId, splitToFeaturesAndTarget

if __name__ == "__main__":
    monk1_train = 'monk/monks-1.train'
    monk1_test = 'monk/monks-1.test'

    # Load training data
    monk1_train_data = pd.read_csv(monk1_train, sep=' ', header=None, )
    # Drop the first column (not needed for training)
    monk1_train_data = monk1_train_data.drop(monk1_train_data.columns[0], axis=1)
    # Rename columns according to dataset description
    monk1_train_data.rename(columns={1: 'target',
                                     2: 'a1',
                                     3: 'a2',
                                     4: 'a3',
                                     5: 'a4',
                                     6: 'a5',
                                     7: 'a6',
                                     8: 'ID'
                                     }, inplace=True)

    # Load testing data
    monk1_test_data = pd.read_csv(monk1_test, sep=' ', header=None, )
    # Drop the first column (not needed for testing)
    monk1_test_data = monk1_test_data.drop(monk1_test_data.columns[0], axis=1)
    # Rename columns according to dataset description
    monk1_test_data.rename(columns={1: 'target',
                                    2: 'a1',
                                    3: 'a2',
                                    4: 'a3',
                                    5: 'a4',
                                    6: 'a5',
                                    7: 'a6',
                                    8: 'ID'
                                    }, inplace=True)

    # Print loaded data for verification
    print('MONK1')
    print('Train data')
    print(monk1_train_data.head())
    print('Test data')
    print(monk1_test_data.head())

    # Separate majority and minority classes
    majority_class = monk1_train_data[monk1_train_data['target'] == 1]
    minority_class = monk1_train_data[monk1_train_data['target'] == 0]

    # Oversample the minority class to match the majority class size
    minority_class = resample(minority_class,
                              replace=True,  # Sample with replacement
                              n_samples=len(majority_class),  # Match majority class size
                              random_state=62)  # For reproducibility

    # Combine the oversampled minority class with the majority class
    monk1_train_data = pd.concat([majority_class, minority_class])

    # Shuffle the balanced dataset
    monk1_train_data = monk1_train_data.sample(frac=1, random_state=62).reset_index(drop=True)

    # Print the balanced dataset for verification
    print('Balancing completed:')
    print(monk1_train_data['target'].value_counts())

    # The balanced dataset is now ready for training
    # Use monk1_train_data for training your model

    # --------------------------------------------------MONK1-----------------------------------------------------------

    # reshape train_X, train_Y, validation_X
    monk1_train_X, monk1_train_Y, monk1_validation_X, monk1_validation_Y = preprocessData(monk1_train_data)

    monk1_train_X = monk1_train_X.reshape(monk1_train_X.shape[0], -1)
    monk1_train_Y = monk1_train_Y.reshape(-1, 1)

    monk1_validation_X = np.array(monk1_validation_X)
    monk1_validation_Y = np.array(monk1_validation_Y)
    monk1_validation_X = monk1_validation_X.reshape(monk1_validation_X.shape[0], -1)

    print(f"val X shape: {monk1_validation_X.shape}")

    # reshape train_X, train_Y, validation_X
    X = monk1_train_X.reshape(monk1_train_X.shape[0], -1)
    y = monk1_train_Y.reshape(-1, 1)

    monk1_validation_X = np.array(monk1_validation_X)
    monk1_validation_Y = np.array(monk1_validation_Y)
    monk1_validation_X = monk1_validation_X.reshape(monk1_validation_X.shape[0], -1)

    print(f"train X shape: {X.shape[1]}")
    print("Nomi delle colonne di X:", X.columns if hasattr(X, 'columns') else "X non è un DataFrame")

    print(f"train Y shape: {y.shape}")

    print(f"val X shape: {monk1_validation_X.shape}")

    # Define the parameter grid
    param_grid = {
        'learning_rate': [0.02, 0.05, 0.1],
        'momentum': [0.85, 0.9, 0.95],
        'lambd': [0.001, 0.003, 0.005, 0.01, 0.05, 0.1]
    }

    # Initialize the Search class for grid search
    search = Search(CustomNeuralNetwork, param_grid, accuracy_score_custom, activation_type=ActivationType.SIGMOID,
                    regularization_type=RegularizationType.L2)

    # Perform grid search on the learning rate
    print("Performing Grid Search...")
    best_params, best_score = search.grid_search(X, y, epoch=400, neurons=[3])
    print(f"Best Parameters:\n {best_params}, Best Score: {best_score}")

    # Define the network with dynamic hidden layers
    nn1 = CustomNeuralNetwork(input_size=X.shape[1],
                              hidden_layers=[3],
                              output_size=1,
                              activationType=ActivationType.SIGMOID,
                              learning_rate=best_params['learning_rate'],
                              momentum=best_params['momentum'],
                              lambd=best_params['lambd'],
                              regularizationType=RegularizationType.L2
                              )

    # Train the network
    history = nn1.fit(X, y, epochs=400, batch_size=8)

    # Plot a single graph with Loss and Training Accuracy
    plt.figure()

    # Plot Training Loss
    plt.plot(history['epoch'], history['train_loss'], label='Training Loss', color='blue', linestyle='-')

    # Plot Training Accuracy
    plt.plot(history['epoch'], history['train_acc'], label='Training Accuracy', color='orange', linestyle='--')

    # Configure the plot
    plt.xlabel('Epochs')  # X-axis as the recorded epochs
    plt.ylabel('Value')  # Shared y-axis label
    plt.title('Training Loss and Accuracy Over Recorded Epochs')
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()

    # Validation predictions
    print('Predicting validation set')
    monk1_validation_nn_predictions = nn1.predict(monk1_validation_X)
    customClassificationReport(monk1_validation_Y, monk1_validation_nn_predictions)

    # -------------------------------------------------TEST------------------------------------------------------------
    # Rimuovi l'ID dal dataset
    monk1_test_data = removeId(monk1_test_data)

    # Applicazione del One-Hot Encoding
    columns_to_encode = monk1_test_data.columns[1:]  # Escludi la colonna 'target'
    encoded_columns = {}
    category_mappings = {}

    for col in columns_to_encode:
        one_hot_encoded, mapping = one_hot_encode(monk1_test_data[col])
        encoded_columns[col] = pd.DataFrame(one_hot_encoded)  # Assicurati che sia un DataFrame
        category_mappings[col] = mapping

    # Concatenazione delle colonne codificate con la colonna target
    encoded_columns_df = pd.concat(encoded_columns.values(), axis=1)
    one_hot_test_monk1 = pd.concat([monk1_test_data['target'], encoded_columns_df], axis=1)

    # Verifica che tutte le colonne abbiano la stessa lunghezza
    assert all(encoded_columns_df[col].shape[0] == len(one_hot_test_monk1) for col in encoded_columns_df.columns), \
        "Le colonne codificate non hanno la stessa lunghezza!"

    monk1_real_test_X, monk1_real_test_Y = splitToFeaturesAndTarget(one_hot_test_monk1)

    # Conversione a numpy array
    try:
        monk1_real_test_X = np.array(monk1_real_test_X, dtype=np.float64)  # Assicurati che siano numerici
    except ValueError as e:
        print("Errore nella conversione dei dati di Features in array numpy:", e)

    monk1_real_test_Y = np.array(monk1_real_test_Y, dtype=np.float64)

    if monk1_real_test_X.ndim == 1:
        monk1_real_test_X = monk1_real_test_X.reshape(-1, 1)

    # Stampa delle dimensioni per debug
    print(f"Train X shape: {monk1_real_test_X.shape}")
    print(f"Train Y shape: {monk1_real_test_Y.shape}")

    monk1_real_test_predictions_nn = nn1.predict(monk1_real_test_X)
    customClassificationReport(monk1_real_test_Y, monk1_real_test_predictions_nn)
