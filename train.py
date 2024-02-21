import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor



def train():
    """Trains a linear regression model on the full dataset and stores output."""
    # Load the data
    data = pd.read_csv("data/properties.csv")

    categorical_features = ["property_type", "subproperty_type", "province", "equipped_kitchen", "state_building", "heating_type"]
    numerical_features = ['construction_year', 'total_area_sqm', 'surface_land_sqm', 'nbr_frontages', 'fl_furnished', 'nbr_bedrooms', 'fl_open_fire', 'fl_terrace', 'terrace_sqm', 'primary_energy_consumption_sqm', "fl_furnished", 'fl_floodzone', 'fl_double_glazing']
    # Define features to use
    X = data
    y = data["price"]

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=505
    )

    # Impute missing values using SimpleImputer
    imputer = SimpleImputer(strategy='mean')  # You can also use 'median', 'most_frequent', or 'constant'

    # Fit the imputer to the data
    imputer.fit(X_train[numerical_features])

    # Transform the data by replacing NaN values with the imputed values
    X_train[numerical_features] = imputer.transform(X_train[numerical_features])
    X_test[numerical_features] = imputer.transform(X_test[numerical_features])

    enc = OneHotEncoder()
    enc.fit(X_train[categorical_features])  # Note the double brackets to create a DataFrame with a single column
    X_train_features = enc.transform(X_train[categorical_features]).toarray()
    X_train_features_df = pd.DataFrame(X_train_features, columns=enc.get_feature_names_out())
    X_test_features = enc.transform(X_test[categorical_features]).toarray()
    X_test_features_df = pd.DataFrame(X_test_features, columns=enc.get_feature_names_out())

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

    # # Train the model
    # model = RandomForestRegressor(n_estimators=200, min_samples_split=80, random_state=1)
    # model.fit(X_train, y_train)

    # # Evaluate the model
    # train_score = r2_score(y_train, model.predict(X_train))
    # test_score = r2_score(y_test, model.predict(X_test))
    # print(f"Train R² score: {train_score}")
    # print(f"Test R² score: {test_score}")
    from sklearn.model_selection import GridSearchCV

# Define the grid of hyperparameters
    param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
    }

    # Initialize the grid search with the RandomForestRegressor and parameter grid
    grid_search = GridSearchCV(RandomForestRegressor(random_state=1), param_grid, cv=5, scoring='r2')

    # Perform grid search on the training data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters and the best R² score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best Hyperparameters:", best_params)
    print("Best R² Score:", best_score)


    # # Save the model
    # artifacts = {
    #     "features": {
    #         "numerical_features": numerical_features,
    #         "categorical_features": categorical_features,
    #     },
    #     "imputer": imputer,
    #     "enc": enc,
    #     "scaler": scaler,
    #     "model": model,
        
    # }
    # joblib.dump(artifacts, "models/artifacts.joblib")


if __name__ == "__main__":
    train()
