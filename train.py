import joblib
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
import xgboost as xgb


def train():
    """Trains a linear regression model on the full dataset and stores output."""
    # Load the data
    data = pd.read_csv("data/properties.csv")
    # IQR Outlier deletion

    for column in [
        # "Price",
        "primary_energy_consumption_sqm",
        "nbr_bedrooms",
        "total_area_sqm",
        "nbr_frontages",
    ]:
        previous_count = data.shape[0]

        # IQR
        # Calculate the upper and lower limits
        Q1 = data[column].quantile(0.1)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        upper = Q3 + 1.5 * IQR

        # Create arrays of Boolean values indicating the outlier rows
        upper_array = np.where(data[column] >= upper)[0]

        # Removing the outliers
        numerical_data_IQR = data.drop(index=data.index[upper_array])

        print(
            f"\nRows removed from {column}:",
            previous_count - numerical_data_IQR.shape[0],
        )
        print("upper outliers:", len(upper_array))

        categorical_features = [
            "property_type",
            "subproperty_type",
            "locality",
            "equipped_kitchen",
            "state_building",
            "heating_type",
            "epc",
        ]
        numerical_features = [
            "construction_year",
            "total_area_sqm",
            "surface_land_sqm",
            "nbr_frontages",
            "fl_furnished",
            "nbr_bedrooms",
            "fl_open_fire",
            "fl_terrace",
            "terrace_sqm",
            "primary_energy_consumption_sqm",
            "fl_floodzone",
            "fl_double_glazing",
            "cadastral_income",
        ]
        # Define features to use
        X = data
        y = data["price"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=505
        )

    # Impute missing values using SimpleImputer
    imputer = SimpleImputer(
        strategy="mean"
    )  # You can also use 'median', 'most_frequent', or 'constant'

    # Fit the imputer to the data
    imputer.fit(X_train[numerical_features])

    # Transform the data by replacing NaN values with the imputed values
    X_train[numerical_features] = imputer.transform(X_train[numerical_features])
    X_test[numerical_features] = imputer.transform(X_test[numerical_features])

    enc = OneHotEncoder()
    enc.fit(
        X_train[categorical_features]
    )  # Note the double brackets to create a DataFrame with a single column
    X_train_features = enc.transform(X_train[categorical_features]).toarray()
    X_train_features_df = pd.DataFrame(
        X_train_features, columns=enc.get_feature_names_out()
    )
    X_test_features = enc.transform(X_test[categorical_features]).toarray()
    X_test_features_df = pd.DataFrame(
        X_test_features, columns=enc.get_feature_names_out()
    )

    scaler = MinMaxScaler()
    scaler.fit(X_train[numerical_features])
    scaled_train = scaler.transform(X_train[numerical_features])
    scaled_test = scaler.transform(X_test[numerical_features])
    scaled_train_df = pd.DataFrame(scaled_train, columns=scaler.get_feature_names_out())
    scaled_test_df = pd.DataFrame(scaled_test, columns=scaler.get_feature_names_out())

    X_train = pd.concat(
        [
            scaled_train_df.reset_index(drop=True),
            pd.DataFrame(X_train_features_df),
        ],
        axis=1,
    )
    X_test = pd.concat(
        [
            scaled_test_df.reset_index(drop=True),
            pd.DataFrame(X_test_features_df),
        ],
        axis=1,
    )

    print(f"Features: \n {X_train.columns.tolist()}")
    # Define the model
    model3 = xgb.XGBRegressor(reg_lambda=5, reg_alpha=1, objective='reg:squarederror', 
                            n_estimators=500, min_child_weight=14, max_depth=12, 
                            learning_rate=0.03, gamma=0.1, colsample_bytree=0.8, booster='gbtree')


    # Train the model
    model3.fit(X_train, y_train)
    # Predict
    train_score = r2_score(y_train, model3.predict(X_train))
    test_score = r2_score(y_test, model3.predict(X_test))
    print(f"Boost: Train R² score: {train_score}")
    print(f"Boost: Test R² score: {test_score}")
    # Train the model for LinearRegression
    model2 = LinearRegression()
    model2.fit(X_train, y_train)

    train_score_line = r2_score(y_train, model2.predict(X_train))
    test_score_line = r2_score(y_test, model2.predict(X_test))
    print(f"Line: Train R² score: {train_score_line}")
    print(f"Line: Test R² score: {test_score_line}")

    # Save the model
    artifacts = {
        "features": {
            "numerical_features": numerical_features,
            "categorical_features": categorical_features,
        },
        "imputer": imputer,
        "enc": enc,
        "scaler": scaler,
        "model2": model2,
        "model3": model3,
    }
    joblib.dump(artifacts, "models/artifacts.joblib", compress=3)

if __name__ == "__main__":
    train()
