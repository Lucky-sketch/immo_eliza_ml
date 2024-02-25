## Model Card

### Project Context

Given the prepared dataset, I was researching and exploring ways to predict prices for real estate based on the features of those properties.

### Data

Input dataset contains 75,000 properties with 30 columns describing different features. In accordance with best machine learning practices, I divided this dataset into 80% train data and 20% test data. Since I chose XGBoost regressor in the end, which takes care of preprocessing and for which outliers don't make a difference, I didn't do much preprocessing except for one hot encoder to transform categories into numbers.

```python
enc = OneHotEncoder()
enc.fit(X_train[categorical_features])

X_train_features = enc.transform(X_train[categorical_features]).toarray()
X_train_features_df = pd.DataFrame(
    X_train_features, columns=enc.get_feature_names_out()
)

X_test_features = enc.transform(X_test[categorical_features]).toarray()
X_test_features_df = pd.DataFrame(
    X_test_features, columns=enc.get_feature_names_out()
)
```

### Model Details

I have tested 3 models: linear regression, random forest, and XGBoost. Due to the better performance of ensemble models and the nature of the data, which cannot be explained by linear patterns, I chose to pursue XGBoost. The performance may vary based on how I regulated these models.

### Performance

With the XGBoost model, I achieved 75% accuracy on the train data and 73% on test data. In the visualization below, blue dots represent actual prices and red ones represent predicted.

![predictions](https://github.com/Lucky-sketch/immo_eliza_ml/assets/53155116/aceaa0e4-114e-495b-ba4d-311d38a5460e)


### Limitations

There are 2 major limitations while working with XGBoost:

1)It is very prone to overfitting so it is imperative to adjust hyperparameters;
2)Very sensitive to hyperparameters, which are also not that easy to adjust.

### Usage

To use the script, install all dependencies inside the requirements.txt file:

```bash
pip install requirements.txt
```

The train.py script trains the model and encoder, which are then dumped to artifact.joblib. predict.py loads these artifacts and uses them. To use predict.py, run the following command in the command line:

```bash
 python3 predict.py -i "path_to_your_data" -o "path_where_you_want_to_store_predictions"
```


### Maintainers

Pray to the god of coding and you will be fine, but you may contact Guido van Rossum, he created Python
