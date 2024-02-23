import click
import joblib
import pandas as pd


@click.command()
@click.option("-i", "--input-dataset", help="path to input .csv dataset", required=True)
@click.option("-m", "--input-model", default = "boost", help="name of the model that you want to use")
@click.option(
    "-o",
    "--output-dataset",
    default="output/predictions.csv",
    help="full path where to store predictions",
    required=True,
)
def predict(input_dataset, input_model, output_dataset):
    """Predicts house prices from 'input_dataset', stores it to 'output_dataset'."""
    ### -------- DO NOT TOUCH THE FOLLOWING LINES -------- ###
    # Load the data
    data = pd.read_csv(input_dataset)
    ### -------------------------------------------------- ###

    # Load the model artifacts using joblib
    artifacts = joblib.load("models/artifacts.joblib")

    # Unpack the artifacts
    numerical_features = artifacts["features"]["numerical_features"]
    categorical_features = artifacts["features"]["categorical_features"]
    imputer = artifacts["imputer"]
    enc = artifacts["enc"]
    model = artifacts["model2"]
    model2 = artifacts["model3"]
    scaler = artifacts["scaler"]



    # Apply imputer and encoder on data
    data_cat = enc.transform(data[categorical_features]).toarray()
    data = pd.concat(
        [
            data[numerical_features].reset_index(drop=True),
            pd.DataFrame(data_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )
    # Make predictions
    if "line" in input_model:
        predictions = model.predict(data)
    else:
        predictions = model2.predict(data)

    data[numerical_features]  = imputer.transform(data[numerical_features])
    data_scaled = pd.DataFrame(scaler.transform(data[numerical_features]), columns = scaler.get_feature_names_out())

    # Combine the numerical and one-hot encoded categorical columns
    data = pd.concat(
        [
            pd.DataFrame(data_scaled.reset_index(drop=True)),
            pd.DataFrame(data_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    # Make predictions
    if "line" in input_model:
        predictions = model.predict(data)
    else:
        predictions = model2.predict(data)

    ### -------- DO NOT TOUCH THE FOLLOWING LINES -------- ###
    # Save the predictions to a CSV file (in order of data input!)
    pd.DataFrame({"predictions": predictions}).to_csv(output_dataset, index=False)

    # Print success messages
    click.echo(click.style("Predictions generated successfully!", fg="green"))
    click.echo(f"Saved to {output_dataset}")
    click.echo(
        f"Nbr. observations: {data.shape[0]} | Nbr. predictions: {predictions.shape[0]}"
    )
    ### -------------------------------------------------- ###


if __name__ == "__main__":
    # how to run on command line:
    # python .\predict.py -i "data\input.csv" -o "output\predictions.csv"
    predict()
