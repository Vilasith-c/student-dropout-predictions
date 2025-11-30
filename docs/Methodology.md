# Methodology

This section describes the dataset, preprocessing steps, experimental design, and machine learning models used for predicting student dropout and academic success.

## 1. Dataset

The experiments use the **“Predict Students' Dropout and Academic Success”** dataset from the UCI Machine Learning Repository. The dataset contains **4,424** records corresponding to undergraduate students from a European higher education institution and **36 input attributes** covering demographic, socio-economic, and academic information, along with a three-class target variable `Target` with values **Dropout**, **Enrolled**, and **Graduate**. The dataset has been widely adopted in prior research on student dropout prediction and academic success. 

## 2. Problem formulation

The task is formulated as a **multi-class classification problem**. Given a student’s features at or near enrollment, the goal is to predict the final outcome class (`Dropout`, `Enrolled`, or `Graduate`).  The study focuses on comparing different supervised learning algorithms and examining how light hyperparameter tuning affects their predictive performance.

## 3. Data preparation

The original data file (`data.csv`) is a semicolon-separated values file obtained from the UCI repository and stored under `data/raw/data.csv`. 

Data preparation steps:

1. **Loading and parsing**  
   - The CSV file is loaded using `pandas.read_csv` with `sep=";"`, resulting in a 4,424 × 37 DataFrame (36 predictors + `Target`). 

2. **Feature and target separation**  
   - The target column `Target` is separated from the remaining 36 predictor variables
