Pyspark Ml Modeling

Use Case: 
- Developing a model to predict the 'Impact' of a book, a composite score achieved post-publication, using its attributes and metadata.
  
Dataset downloaded:
- https://drive.google.com/file/d/1VXp7KDhIvh2B7nnsBdfeMY4SwMNrPtCs/view?usp=sharing

Project Outline:
- EDA (notebook)
- Set up and ran Spark locally: Simulated different worker configurations.
- MLFlow tracking of metric and performance
- 



Performance Analysis:

- Num of worker: 2 -> Duration: 10.6min, MAE: 40.62
- <img width="1428" alt="image" src="https://github.com/harshita23sharma/big_data_ml/assets/16293041/71f2bd02-c799-4298-996e-1fdbe83e2d1b">


- Num of worker :1 -> Duration : 20 mins, MAE: 40.58
<img width="1434" alt="image" src="https://github.com/harshita23sharma/big_data_ml/assets/16293041/6bea8c8a-275d-48af-9699-d81f6367977c">

