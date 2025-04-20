import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Function to load the CSV file
def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Function to inspect the dataset and handle columns dynamically
def process_data(df):
    # Handle missing values (NaN)
    df = df.dropna()  # Drop rows with NaN values
    
    # Print the first few rows of the dataset
    print("Dataset Preview:")
    print(df.head())
    
    # Display columns and types
    print("\nColumns and Data Types:")
    print(df.dtypes)

    # Ask the user to define features and target column
    target_column = input("\nPlease enter the target column name: ")
    feature_columns = input("\nPlease enter the feature columns (comma separated): ").split(',')

    # Check if the target column and feature columns exist
    if target_column not in df.columns:
        print(f"Error: '{target_column}' not found in the dataset.")
        return None, None
    if any(col not in df.columns for col in feature_columns):
        print(f"Error: Some feature columns not found in the dataset.")
        return None, None

    # Encoding categorical columns
    label_encoders = {}
    for col in feature_columns + [target_column]:
        if df[col].dtype == 'object':  # Apply Label Encoding only to categorical data
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le  # Save the encoder for future use if needed

    return df[feature_columns], df[target_column], label_encoders

# Main function to execute the process
def main():
    # Get the file path from the user
    file_path = input("Please enter the path to your CSV file: ")
    
    # Load the dataset
    df = load_dataset(file_path)

    if df is None:
        return  # Exit if data loading failed

    # Process the data: Get features, target, and label encoders
    features, target, label_encoders = process_data(df)

    if features is None:
        return  # Exit if data processing failed

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)

    # Model training
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Model evaluation
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred, average='weighted'))
    print('Recall:', recall_score(y_test, y_pred, average='weighted'))
    print('F1 Score:', f1_score(y_test, y_pred, average='weighted'))
    print("\nPredictions:", y_pred)

# Run the main function
main()