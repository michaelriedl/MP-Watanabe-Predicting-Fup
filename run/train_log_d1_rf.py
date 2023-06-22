import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from mordred import Calculator, descriptors
from padelpy import padeldescriptor
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
from sklearn.metrics import r2_score

# Define the hyperparameters for the RF based on the publication
N_ESTIMATORS = 500
MAX_FEATURES = 261

# Set the random seed to use
RANDOM_SEED = 0

# Define the low value threshold
LOW_VALUE_THRESHOLD = np.log10(0.05)

# Define the fingerprint types to use
FINGERPRINT_MAP = {
    "AtomPairs2D": "../data/fingerprint_templates/AtomPairs2DFingerprinter.xml",
    "CDKextended": "../data/fingerprint_templates/ExtendedFingerprinter.xml",
    "KlekotaRoth": "../data/fingerprint_templates/KlekotaRothFingerprinter.xml",
}


def main():
    # Fix the seeds of the RNGs
    np.random.seed(RANDOM_SEED)

    # Set the file name of the data
    fup_watanabe_file_name = "../data/raw/fup_watanabe.csv"

    # Set the name of the descriptor file to generate
    watanabe_desc_file_name = "../data/intermediate/fup_watanabe_mordred.csv"

    # Set the name of the down-selected descriptor file to generate
    watanabe_desc_down_file_name = "../data/final/fup_watanabe_d1_down.csv"

    # Set the temporary SMILES file name
    temp_smi_file_name = "../data/temp.smi"

    # Set the results directory
    results_dir = "../results/train_log_d1_rf/"

    # Create the results directory if it does not exist
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # Load the data
    fup_watanabe_df = pd.read_csv(fup_watanabe_file_name, skiprows=1)

    # Create the temporary SMILES file
    temp_df = pd.DataFrame(columns=["SMILES", "No."])
    temp_df["SMILES"] = fup_watanabe_df["canonical_smiles"].to_list()
    temp_df["No."] = fup_watanabe_df["No."].to_list()
    temp_df.to_csv(temp_smi_file_name, sep="\t", index=False, header=False)

    # Create the descriptor calculator and calculate the descriptors
    if not os.path.isfile(watanabe_desc_file_name):
        # Create the calculator
        desc_calc = Calculator(descriptors, ignore_3D=True, version="1.0.0")
        # Convert the SMILES to molecules
        mols_list = [
            Chem.MolFromSmiles(smi) for smi in fup_watanabe_df["canonical_smiles"]
        ]
        # Calculate the descriptors
        desc_df = desc_calc.pandas(mols_list)
        # Filter out the failed descriptors
        keep_cols = []
        for key, value in desc_df.dtypes.to_dict().items():
            if value != object:
                keep_cols.append(key)
        desc_df = desc_df[keep_cols]
        # Add in the number index for matching
        desc_df.insert(0, "No.", fup_watanabe_df["No."].to_list())
        # Save the descriptors
        desc_df.to_csv(watanabe_desc_file_name, index=False)
    # Load the descriptors
    desc_df = pd.read_csv(watanabe_desc_file_name)

    # Create the fingerprints
    fp_dfs = {}
    for fp_type, fp_file in FINGERPRINT_MAP.items():
        # Set the output file name
        fp_out_file_name = f"../data/intermediate/fup_watanabe_{fp_type.lower()}.csv"
        # Check if the file exists
        if not os.path.isfile(fp_out_file_name):
            # Create the fingerprints
            padeldescriptor(
                mol_dir=temp_smi_file_name,
                d_file=fp_out_file_name,
                descriptortypes=fp_file,
                detectaromaticity=True,
                standardizenitro=True,
                standardizetautomers=True,
                removesalt=True,
                fingerprints=True,
                threads=8,
            )
        # Load the fingerprints
        fp_dfs[fp_type] = pd.read_csv(fp_out_file_name)

    # Merge the fingerprints with the descriptors
    for fp_type, fp_df in fp_dfs.items():
        fp_df = fp_df.rename(columns={"Name": "No."})
        desc_df = desc_df.merge(fp_df, on="No.", how="left")

    # Filter the training and test sets
    fup_train_df = fup_watanabe_df[fup_watanabe_df["Test set or Training set"] == "Tr"]
    fup_test_df = fup_watanabe_df[fup_watanabe_df["Test set or Training set"] == "Te"]

    # Create the random forest regressor
    rf = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, max_features=MAX_FEATURES, n_jobs=-1
    )

    # Down-select the features
    if not os.path.isfile(watanabe_desc_down_file_name):
        # Create the Boruta feature selection
        feat_selector = BorutaPy(
            rf, n_estimators="auto", verbose=2, random_state=RANDOM_SEED
        )

        # Get the training data
        X_train = desc_df[desc_df["No."].isin(fup_train_df["No."])]
        X_train = X_train.drop("No.", axis=1).to_numpy()
        y_train = fup_train_df["fup_log"].to_numpy()

        # Fit the feature selector
        feat_selector.fit(X_train, y_train)

        # Filter the features
        feat_cols = list(desc_df.columns)
        feat_cols.remove("No.")
        keep_feat_cols = [x for x, y in zip(feat_cols, feat_selector.support_) if y]
        desc_down_df = desc_df[["No."] + keep_feat_cols]
        desc_down_df.to_csv(watanabe_desc_down_file_name, index=False)
    # Load the down-selected features
    desc_down_df = pd.read_csv(watanabe_desc_down_file_name)

    # Get the training data
    X_train = desc_down_df[desc_down_df["No."].isin(fup_train_df["No."])]
    X_train = X_train.drop("No.", axis=1).to_numpy()
    y_train = fup_train_df["fup_log"].to_numpy()

    # Get the testing data
    X_test = desc_down_df[desc_down_df["No."].isin(fup_test_df["No."])]
    X_test = X_test.drop("No.", axis=1).to_numpy()
    y_test = fup_test_df["fup_log"].to_numpy()

    # Fit the data
    rf.fit(X_train, y_train)

    # Get the predictions on the test set
    y_pred = rf.predict(X_test)

    # Calculate and print the test metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mae = np.mean(np.abs(y_test - y_pred))
    print("Full dataset:")
    print(f"R^2: {r2:0.3f}")
    print(f"RMSE: {rmse:0.3f}")
    print(f"MAE: {mae:0.3f}\n")

    r2_low = r2_score(
        y_test[y_test < LOW_VALUE_THRESHOLD], y_pred[y_test < LOW_VALUE_THRESHOLD]
    )
    rmse_low = np.sqrt(
        np.mean(
            (
                y_test[y_test < LOW_VALUE_THRESHOLD]
                - y_pred[y_test < LOW_VALUE_THRESHOLD]
            )
            ** 2
        )
    )
    mae_low = np.mean(
        np.abs(
            y_test[y_test < LOW_VALUE_THRESHOLD] - y_pred[y_test < LOW_VALUE_THRESHOLD]
        )
    )
    print("Lower range:")
    print(f"R^2: {r2_low:0.3f}")
    print(f"RMSE: {rmse_low:0.3f}")
    print(f"MAE: {mae_low:0.3f}\n")

    # Write the test metrics to a file
    with open(os.path.join(results_dir, "test_metrics.txt"), "w") as f:
        f.write("Full dataset:\n")
        f.write(f"R^2: {r2:0.3f}\n")
        f.write(f"RMSE: {rmse:0.3f}\n")
        f.write(f"MAE: {mae:0.3f}\n\n")

        f.write("Lower range:\n")
        f.write(f"R^2: {r2_low:0.3f}\n")
        f.write(f"RMSE: {rmse_low:0.3f}\n")
        f.write(f"MAE: {mae_low:0.3f}\n\n")

    # Plot the results
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(y_test, y_pred, s=10)
    ax.plot([-3.25, 0.25], [-3.25, 0.25], "r--")
    plt.xlim([-3.25, 0.25])
    plt.ylim([-3.25, 0.25])
    ax.set_xlabel(r"True $f_{u,p}$ (log)")
    ax.set_ylabel(r"Predicted $f_{u,p}$ (log)")
    plt.savefig(
        os.path.join(results_dir, "watanabe_parity.png"),
        bbox_inches="tight",
        dpi=300,
    )

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(
        y_test[y_test < LOW_VALUE_THRESHOLD], y_pred[y_test < LOW_VALUE_THRESHOLD], s=10
    )
    ax.plot([-3.25, 0.25], [-3.25, 0.25], "r--")
    plt.xlim([-3.25, -1])
    plt.ylim([-3.25, 0])
    ax.set_xlabel(r"True $f_{u,p}$ (log)")
    ax.set_ylabel(r"Predicted $f_{u,p}$ (log)")
    plt.savefig(
        os.path.join(results_dir, "watanabe_parity_low.png"),
        bbox_inches="tight",
        dpi=300,
    )

    # Save the trained model
    joblib.dump(rf, os.path.join(results_dir, "rf_model.joblib"))

    # Remove the temporary SMILES file
    os.remove(temp_smi_file_name)


if __name__ == "__main__":
    main()
