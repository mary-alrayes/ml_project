import numpy as np
from matplotlib import pyplot as plt

from project.CustomNN import CustomNeuralNetwork
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error
from project.utility.Enum import RegularizationType, ActivationType, TaskType
from project.utility.Search import Search
from project.utility.utility import customRegressionReport, preprocessRegrData, save_predictions_to_csv, denormalize_zscore, min_max_denormalization, preprocessRegressionTestData

if __name__ == "__main__":
    train = "cup/ML-CUP24-TR.csv"
    test = "cup/ML-CUP24-TS.csv"

    # Read the training dataset
    train_data = pd.read_csv(train, comment="#", header=None)

    # Add column headers based on the format
    columns = [
        "ID",
        "INPUT1",
        "INPUT2",
        "INPUT3",
        "INPUT4",
        "INPUT5",
        "INPUT6",
        "INPUT7",
        "INPUT8",
        "INPUT9",
        "INPUT10",
        "INPUT11",
        "INPUT12",
        "TARGET_x",
        "TARGET_y",
        "TARGET_z",
    ]

    train_data.columns = columns

    # Read the test dataset
    test_data = pd.read_csv(test, comment="#", header=None)

    # Add column headers based on the format
    columns = [
        "ID",
        "INPUT1",
        "INPUT2",
        "INPUT3",
        "INPUT4",
        "INPUT5",
        "INPUT6",
        "INPUT7",
        "INPUT8",
        "INPUT9",
        "INPUT10",
        "INPUT11",
        "INPUT12",
    ]

    test_data.columns = columns

    # Print the training and test data
    print("CUP")
    print("train shape: ", train_data.shape, "\n train: \n", train_data.head())
    print("test shape: \n", test_data.shape, "\n test: \n", test_data.head())

    train_X, train_Y, validation_X, validation_Y = preprocessRegrData(
        train_data, standard=True, target_columns=["TARGET_x", "TARGET_y", "TARGET_z"]
    )
    print("train_X\n: ", train_X.shape, "train_Y\n: ", train_Y.shape)
    print("validation_X\n: ", validation_X.shape, "validation_Y\n:  ", validation_Y.shape)
    
    #reshape train_X, train_Y, validation_X  
    X = train_X
    y = train_Y.reshape(-1, 3)

    print("train X shape: ", X.shape)
    print("train Y shape: ", y.shape)

    # Define the parameter grid
    param_grid = {
        'learning_rate': [0.001, 0.002], #[0.1 / (10**i) for i in range(10)],  # Learning rate values
        'momentum': [0.85, 0.9], #[x/100 for x in range(83, 89)],              # Momentum values
        'lambd': [0.005, 0.01], #[0.2 / (10**i) for i in range(7)] + [x/10 for x in range(1, 10)], #+ [x/100 for x in range(1, 10)],              # Regularization lambda values
        'hidden_layers': [[x, y] for x in range(24, 26) for y in range(15, 18)],   # Number of neurons in the hidden layer
    }

    # Initialize the Search class for grid search
    search = Search(CustomNeuralNetwork, param_grid, activation_type=ActivationType.ELU, 
                    regularization_type=RegularizationType.L2, nesterov=True, decay=0.0005)

    # Perform grid search on the learning rate
    print("Performing Grid Search...")
    best_params, best_score = search.grid_search_regression(
        X, y, epoch=150, output_size=3, batchSize=20
    )
    print("Best Parameters:\n ", best_params, "Best Score: ", best_score)

    # Define the network with dynamic hidden layers
    nn1 = CustomNeuralNetwork(input_size=X.shape[1],
                              hidden_layers=best_params['hidden_layers'],
                              output_size=3,
                              activationType=ActivationType.ELU,
                              learning_rate=best_params['learning_rate'],
                              momentum=best_params['momentum'],
                              lambd=best_params['lambd'],
                              regularizationType=RegularizationType.L2,
                              task_type=TaskType.REGRESSION,
                              nesterov=True,
                              decay=0.0005
                              )
    # Train the network
    history = nn1.fit(X, y, X_val=validation_X, y_val=validation_Y, epochs=150, batch_size=20)

    # Plot a single graph with Loss and Training Accuracy
    plt.figure()

    # Plot Training Loss
    plt.plot(history['epoch'], history['train_loss'], label='Training Loss', color='blue', linestyle='-')

    # Plot Validation Loss
    plt.plot(history['epoch'], history['val_loss'], label='Validation Loss', color='green', linestyle='-')

    # Plot Training Accuracy
    plt.plot(history['epoch'], history['train_me'], label='Training me', color='orange', linestyle='--')
    
    # Plot Validation Accuracy
    plt.plot(history['epoch'], history['val_me'], label='Validation me', color='red', linestyle='--')

    # Configure the plot
    plt.xlabel("Epochs")  # X-axis as the recorded epochs
    plt.ylabel("Value")  # Shared y-axis label
    plt.title("Training Loss and Accuracy Over Recorded Epochs, elu Activation, L2 Regularization, decay=0.0005, nesterov, xavier initialization, 16 batch")  # Plot title
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()

    # Validation predictions
    print("Predicting validation set: elu Activation, L2 Regularization, decay=0.0005, nesterov, xavier initialization, 16 batch")  
    validation_nn_predictions = nn1.predict(validation_X)
    customRegressionReport(validation_Y, validation_nn_predictions, target_names=['TARGET_x', 'TARGET_y', 'TARGET_z'])
    
    print('Predicting denormalized validation set')
    #Denormalize the validation predictions
    validation_Y_denorm = denormalize_zscore(validation_Y, data=train_data, target_columns=['TARGET_x', 'TARGET_y', 'TARGET_z'])
    validation_nn_pred_denorm = denormalize_zscore(validation_nn_predictions, data=train_data, target_columns=['TARGET_x', 'TARGET_y', 'TARGET_z'])
    customRegressionReport(validation_Y_denorm, validation_nn_pred_denorm, target_names=['TARGET_x', 'TARGET_y', 'TARGET_z'])


#-------------------------------Test Set Predictions---------------------------------
    #test preprocessing
    test_X = preprocessRegressionTestData(train_data, test_data, standard=True, target_columns=['TARGET_x', 'TARGET_y', 'TARGET_z'])
    print('Predicting test set')
    
    # Test predictions
    test_nn_predictions = nn1.predict(test_X)
    print("test_nn_predictions: ", test_nn_predictions.shape)
    
    # Denormalize the test predictions
    test_nn_predictions_denorm = denormalize_zscore(test_nn_predictions, data=train_data, target_columns=['TARGET_x', 'TARGET_y', 'TARGET_z'])
    print("test_nn_predictions_denorm: ", test_nn_predictions_denorm.shape)
    
    # Save the test predictions to a CSV file
    save_predictions_to_csv(test_nn_predictions_denorm, file_name="predictions.csv")
