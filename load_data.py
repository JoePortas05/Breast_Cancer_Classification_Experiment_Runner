import pandas as pd


# Method to load the dataset
def load_dataset():
    data_file_path = "breast-cancer-wisconsin.data"

    # The columns in the data
    columns = [
        "sample_code_number",
        "clump_thickness",
        "uniformity_of_cell_size",
        "uniformity_of_cell_shape",
        "marginal_adhesion",
        "single_epithelial_cell_size",
        "bare_nuclei",
        "bland_chromatin",
        "normal_nucleoli",
        "mitoses",
        "class",
    ]

    # The features in the data
    features = [
        "clump_thickness",
        "uniformity_of_cell_size",
        "uniformity_of_cell_shape",
        "marginal_adhesion",
        "single_epithelial_cell_size",
        "bare_nuclei",
        "bland_chromatin",
        "normal_nucleoli",
        "mitoses",
    ]

    # Read the data and replace ?s with NA
    df = pd.read_csv(data_file_path, header=None, names=columns)
    df = df.replace("?", pd.NA)

    # Convert feature values to numeric
    for column in features:
        numeric = pd.to_numeric(df[column], errors="coerce")
        df[column] = numeric

    # Get the features frame and classes and return them
    X = df[features].copy()
    y = df["class"].map({2: 0, 4: 1}).astype(int)

    return X, y
