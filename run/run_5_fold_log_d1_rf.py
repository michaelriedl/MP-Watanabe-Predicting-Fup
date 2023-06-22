import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Define the hyperparameters for the RF based on the publication
N_ESTIMATORS = 500
MAX_FEATURES = 261

# Set the random seed to use
RANDOM_SEED = 0

# Set the number of folds
NUM_FOLDS = 5


def main():
    # Fix the seeds of the RNGs
    np.random.seed(RANDOM_SEED)

    # Set the file name of the data
    fup_watanabe_file_name = "../data/raw/fup_watanabe.csv"

    # Set the name of the down-selected descriptor file to generate
    watanabe_desc_down_file_name = "../data/final/fup_watanabe_d1_down.csv"

    # Set the results directory
    results_dir = "../results/run_5_fold_log_d1_rf/"

    # Create the results directory if it does not exist
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # Load the data
    fup_watanabe_df = pd.read_csv(fup_watanabe_file_name, skiprows=1)

    # Filter the training and test sets
    fup_train_df = fup_watanabe_df[fup_watanabe_df["Test set or Training set"] == "Tr"]

    # Create the random forest regressor
    rf = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, max_features=MAX_FEATURES, n_jobs=-1
    )

    # Load the down-selected features
    desc_down_df = pd.read_csv(watanabe_desc_down_file_name)

    # Run the 5-fold cross-validation
    for fold in range(NUM_FOLDS):
        # Get the training data
        X_train = desc_down_df[
            desc_down_df["No."].isin(fup_train_df["No."][fup_train_df["Split"] != fold])
        ]
        X_train = X_train.drop("No.", axis=1).to_numpy()
        y_train = fup_train_df["fup_log"][fup_train_df["Split"] != fold].to_numpy()

        # Get the testing data
        X_test = desc_down_df[
            desc_down_df["No."].isin(fup_train_df["No."][fup_train_df["Split"] == fold])
        ]
        X_test = X_test.drop("No.", axis=1).to_numpy()
        y_test = fup_train_df["fup_log"][fup_train_df["Split"] == fold].to_numpy()

        # Fit the data
        rf.fit(X_train, y_train)

        # Get the predictions on the test set
        y_pred = rf.predict(X_test)

        # Calculate and print the test metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        mae = np.mean(np.abs(y_test - y_pred))
        print(f"Fold {fold} results:")
        print(f"R^2: {r2:0.3f}")
        print(f"RMSE: {rmse:0.3f}")
        print(f"MAE: {mae:0.3f}\n")

        # Write the test metrics to a file
        if fold == 0:
            mode = "w"
        else:
            mode = "a"
        with open(os.path.join(results_dir, "test_metrics.txt"), mode) as f:
            f.write(f"Fold {fold} results:\n")
            f.write(f"R^2: {r2}\n")
            f.write(f"RMSE: {rmse}\n")
            f.write(f"MAE: {mae}\n\n")


if __name__ == "__main__":
    main()
