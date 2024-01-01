**Data Analysis**:
- Columns: Title, description, authors, publisher, publishedDate, categories, Impact
- Row value with max % missing: 38.0
- Column: publisher have long tail, so bucketizing to "others" category: Top 2808 publishers and others category
- Missing Rows: 10.32%

**Feature Engineering**
- Created publishedDate_year and publishedDate_month from publishedDate
- Concatenated Title and description as sometimes Title is missing and sometimes description. Created new column title_desc
- One hot encoded categorical features
- Tokenised, and tf idf encoding on text features

**Model**
- Train Test split (10% test data)
- Model Hyper parameter tuning on Regressor model. Used DecisionTree and Random forest
- With Categorical(metadata features only) Random forest was working beter. Including text based featues : DecisionTree
  


Data Downloaded: books_task.csv

create dir : book_impact/data/processd and book_impact/data/predictions

pip install -r requirements.txt

**Steps**:

- **scripts/train_text_encoder.py**: Loads and reads text columns from csv data. Concatenated columns(text and description) for the use case and saved a pipelinemodel for encoding (tokenising and encoding the text)
  
- **scripts/data_prerocessing.py**: Handles Categorical data and encode text data after loading encoder and saves the necessary feature columns ti parquet file.
  
- **scripts/model_hyperparameter_tuning.py**: Hyper parameter tuning on different worker count. Sampled 50% of total data for running (due to time constraint). Input both categorical and** text features**  Models : RandomForestRegressor(maxDepth=5, numTrees=40) and DecisionTreeRegressor(maxDepth=5, maxBins=32). Logging the Best model, params and result in mlflow
  
- **scripts/train.py**: Read parquet file and fit the model on training data (splitted in previous step of hyper param tuning) Also saves the final model
  
- **scripts/inference.py**: Loads the model and test data. Apply all the transformations on test data and predicts the model output
