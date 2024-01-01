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
- 
