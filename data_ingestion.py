import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def load_data():
    local_file = "Housing.csv"  # Local path to your data file
    return pd.read_csv(local_file)

def preprocess_data(data):
    # Identify categorical columns (columns with dtype 'object' are usually categorical)
    categorical_cols = data.select_dtypes(include=['object']).columns

    # Apply label encoding or one-hot encoding to categorical columns
    for col in categorical_cols:
        if data[col].nunique() == 2:  # If there are only two unique values, use LabelEncoder
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
        else:
            data = pd.get_dummies(data, columns=[col], drop_first=True)  # One-hot encode for more than 2 categories

    return data

if __name__ == "__main__":
    data = load_data()
    print("Data loaded successfully.")

    # Preprocess data
    data = preprocess_data(data)

    # Make sure the 'price' column exists and is numeric
    if 'price' not in data.columns:
        raise ValueError("'price' column not found in the dataset")

    # Splitting data into features and target
    X = data.drop('price', axis=1)
    y = data['price']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(n_estimators=100, max_depth=None)
    model.fit(X_train, y_train)

    # Predict on the test set
    predictions = model.predict(X_test)
    print(f"Generated {len(predictions)} predictions.")
