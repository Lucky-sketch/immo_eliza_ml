import joblib
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb

def train():
    """
    Train an XGBoost regression model to predict real estate prices.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    This function performs the following steps:
    1. Load and preprocess the dataset.
    2. Train an XGBoost regression model.
    3. Evaluate the model on the training and test sets.
    4. Save the trained model and artifacts.

    
    Examples
    --------
    train() 

    """
    # Load the data
    data = pd.read_csv("immo_eliza_ml/data/properties.csv")

    # Define numerical and categorical features
    num_features = ["nbr_frontages", 'nbr_bedrooms', "latitude", "longitude", "total_area_sqm",
                     'surface_land_sqm','terrace_sqm','garden_sqm']
    cat_features = ["province", 'heating_type', 'state_building',
                    "property_type", "epc", 'locality', 'subproperty_type','region', "fl_terrace", 'fl_garden', 'fl_swimming_pool']

    # Select features
    X = data[num_features + cat_features]
    y = data["price"]

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Encode categorical features
    enc = OneHotEncoder()
    enc.fit(X_train[cat_features])
    X_train_features = enc.transform(X_train[cat_features]).toarray()
    X_train_features_df = pd.DataFrame(
        X_train_features, columns=enc.get_feature_names_out()
    )
    X_test_features = enc.transform(X_test[cat_features]).toarray()
    X_test_features_df = pd.DataFrame(
        X_test_features, columns=enc.get_feature_names_out()
    )

    # Combine numerical and encoded categorical features
    X_train = pd.concat([X_train[num_features].reset_index(drop=True), X_train_features_df], axis=1)
    X_test = pd.concat([X_test[num_features].reset_index(drop=True), X_test_features_df], axis=1)

    # Print the features being used
    print(f"Features: \n {X_train.columns.tolist()}")

    # Define the best parameters found with hyperparameter tuning
    best_params = {
        'n_estimators': 200,
        'max_depth': 7,
        'learning_rate': 0.1,
        'subsample': 0.9,
        'colsample_bytree': 1.0,
        'gamma': 5,
        'reg_alpha': 0,
        'reg_lambda': 1.0,
    }

    # Define the XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)

    # Train the XGBoost model
    model.fit(X_train, y_train)

    # Evaluate the performance of the model
    train_score = r2_score(y_train, model.predict(X_train))
    test_score = r2_score(y_test, model.predict(X_test))
    print(f"Boost: Train R² score: {train_score}")
    print(f"Boost: Test R² score: {test_score}")

    # Save the trained model and encoding information
    artifacts = {
        "features": {
            "numerical_features": num_features,
            "categorical_features": cat_features,
        },
        "model": model,
        'enc': enc,
    }
    joblib.dump(artifacts, "immo_eliza_ml/models/artifacts.joblib", compress=3)

if __name__ == "__main__":
    train()
