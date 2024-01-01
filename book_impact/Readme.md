Data Downloaded: books_task.csv
create dir : book_impact/data/processd and book_impact/data/predictions

Steps:
- scripts/train_text_encoder.py: Loads and reads text columns from csv data. Concatenated columns(text and description) for the use case and saved a pipelinemodel for encoding (tokenising and encoding the text)
- scripts/data_prerocessing.py: Handles Categorical data and encode text data after loading encoder and saves the necessary feature columns ti parquet file.
- scripts/model_hyperparameter_tuning.py: Hyper parameter tuning on different worker count. Sampled 50% of total data for running (due to time constraint). Input both categorical and** text features**  Models : RandomForestRegressor(maxDepth=5, numTrees=40) and DecisionTreeRegressor(maxDepth=5, maxBins=32). Logging the Best model, params and result in mlflow
- scripts/train.py: Read parquet file and fit the model on training data (splitted in previous step of hyper param tuning) Also saves the final model
- scripts/inference.py: Loads the model and test data. Apply all the transformations on test data and predicts the model output
