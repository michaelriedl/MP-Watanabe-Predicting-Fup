import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from mordred import Calculator, descriptors
from padelpy import padeldescriptor
from sklearn.metrics import r2_score

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

    # Set the file name of the model to load
    model_file_name = "../results/train_log_d1_rf/rf_model.joblib"

    # Set the name of the down-selected descriptor file to load
    watanabe_desc_down_file_name = "../data/final/fup_watanabe_d1_down.csv"

    # Set the file name of the OPERA data
    fup_opera_file_name = "../data/raw/fup_opera_no_overlap.csv"

    # Set the name of the descriptor file to generate
    opera_desc_file_name = "../data/intermediate/fup_opera_mordred.csv"

    # Set the file name for the valid data
    fup_opera_valid_file_name = "../data/final/fup_opera_no_overlap_valid.csv"

    # Set the temporary SMILES file name
    temp_smi_file_name = "../data/temp.smi"

    # Set the results directory
    results_dir = "../results/eval_log_d1_rf_opera/"

    # Create the results directory if it does not exist
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # Load the data
    desc_down_watanabe_df = pd.read_csv(watanabe_desc_down_file_name)
    fup_opera_df = pd.read_csv(fup_opera_file_name)

    # Create the temporary SMILES file
    temp_df = pd.DataFrame(columns=["SMILES", "No."])
    temp_df["SMILES"] = fup_opera_df["Canonical_QSARr"].to_list()
    temp_df["No."] = fup_opera_df["No."].to_list()
    temp_df.to_csv(temp_smi_file_name, sep="\t", index=False, header=False)

    # Create the descriptor calculator and calculate the descriptors
    if not os.path.isfile(opera_desc_file_name):
        # Create the calculator
        desc_calc = Calculator(descriptors, ignore_3D=True, version="1.0.0")
        # Convert the SMILES to molecules
        mols_list = [Chem.MolFromSmiles(smi) for smi in fup_opera_df["Canonical_QSARr"]]
        # Calculate the descriptors
        desc_opera_df = desc_calc.pandas(mols_list)
        # Add in the number index for matching
        desc_opera_df.insert(0, "No.", fup_opera_df["No."].to_list())
        # Save the descriptors
        desc_opera_df.to_csv(opera_desc_file_name, index=False)
    # Load the descriptors
    desc_opera_df = pd.read_csv(opera_desc_file_name)

    # Create the fingerprints
    fp_dfs = {}
    for fp_type, fp_file in FINGERPRINT_MAP.items():
        # Set the output file name
        fp_out_file_name = f"../data/intermediate/fup_opera_{fp_type.lower()}.csv"
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
        desc_opera_df = desc_opera_df.merge(fp_df, on="No.", how="left")

    # Filter down to the descriptors from Watanabe
    keep_cols = list(desc_down_watanabe_df.columns)
    keep_cols.remove("No.")
    desc_down_opera_df = desc_opera_df[keep_cols]

    # Drop rows that have invalid descriptors
    keep_list = []
    for _, row in desc_down_opera_df.iterrows():
        if not row.str.contains("invalid").any():
            keep_list.append(True)
        else:
            keep_list.append(False)
    desc_down_opera_df = desc_down_opera_df[keep_list]
    fup_opera_df = fup_opera_df[keep_list]
    fup_opera_df.to_csv(fup_opera_valid_file_name, index=False)
    print(f"Number of dropped rows: {len(keep_list) - sum(keep_list)}\n")

    # Load the random forest model
    rf = joblib.load(model_file_name)

    # Get the testing data
    X_test = desc_down_opera_df.to_numpy()
    y_test = fup_opera_df["log10_fup"].to_numpy()

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

    # Remove the temporary SMILES file
    os.remove(temp_smi_file_name)


if __name__ == "__main__":
    main()
