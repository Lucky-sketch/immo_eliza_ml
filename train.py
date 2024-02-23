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

    # for column in [
    #     # "Price",
    #     "primary_energy_consumption_sqm",
    #     "nbr_bedrooms",
    #     "total_area_sqm",
    #     "nbr_frontages",
    # ]:
    #     previous_count = data.shape[0]

    #     # IQR
    #     # Calculate the upper and lower limits
    #     Q1 = data[column].quantile(0.1)
    #     Q3 = data[column].quantile(0.75)
    #     IQR = Q3 - Q1
    #     upper = Q3 + 1.5 * IQR

    #     # Create arrays of Boolean values indicating the outlier rows
    #     upper_array = np.where(data[column] >= upper)[0]

    #     # Removing the outliers
    #     numerical_data_IQR = data.drop(index=data.index[upper_array])

    #     print(
    #         f"\nRows removed from {column}:",
    #         previous_count - numerical_data_IQR.shape[0],
    #     )
    #     print("upper outliers:", len(upper_array))

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
        'latitude',
        'longitude',
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
    #catefory in nums
    enc = OneHotEncoder()
    enc.fit(
        X_train[categorical_features]
    )  
    X_train_features = enc.transform(X_train[categorical_features]).toarray()
    X_train_features_df = pd.DataFrame(
        X_train_features, columns=enc.get_feature_names_out()
    )
    X_test_features = enc.transform(X_test[categorical_features]).toarray()
    X_test_features_df = pd.DataFrame(
        X_test_features, columns=enc.get_feature_names_out()
    )

    X_train = pd.concat(
        [
            X_train[numerical_features].reset_index(drop=True),
            X_train_features_df,
        ],
        axis=1,
    )
    X_test = pd.concat(
        [
            X_test[numerical_features].reset_index(drop=True),
            X_test_features_df,
        ],
        axis=1,
    )
        
    print(f"Features: \n {X_train.columns.tolist()}")

    # Define the boost model
    model = xgb.XGBRegressor(
        reg_lambda=3,  # Adjust regularization parameters
        reg_alpha=0.5,
        objective='reg:squarederror',
        n_estimators=1000,
        min_child_weight=25,  # Increase minimum child weight
        max_depth=5,  # Decrease maximum depth
        learning_rate=0.01,  # Decrease learning rate
        gamma=0.1,
        colsample_bytree=0.8,
        booster='gbtree'
    )


    # Train the boost model
    model.fit(X_train, y_train)

    train_score = r2_score(y_train, model.predict(X_train))
    test_score = r2_score(y_test, model.predict(X_test))
    print(f"Boost: Train R² score: {train_score}")
    print(f"Boost: Test R² score: {test_score}")


    # Save the model
    artifacts = {
        "features": {
            "numerical_features": numerical_features,
            "categorical_features": categorical_features,
        },
        "model": model,
        'enc': enc,
    }
    joblib.dump(artifacts, "models/artifacts.joblib", compress=3)

if __name__ == "__main__":
    train()
