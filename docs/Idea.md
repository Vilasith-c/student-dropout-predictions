# Project Idea: Student Dropout Prediction

## 1. Motivation

Student dropout is a major problem for higher education institutions because it leads to financial loss for colleges and wasted time and effort for students. Early identification of at-risk students can help universities provide timely interventions and improve overall academic success. [web:22][web:24]

This project aims to build and compare machine learning models that can predict whether a student is likely to **Dropout**, **Remain Enrolled**, or **Graduate**, using historical academic and demographic data from a real higher-education institution. [web:11][web:60]

## 2. Dataset

- **Name:** Predict Students' Dropout and Academic Success  
- **Source:** UCI Machine Learning Repository (with mirrors on Kaggle). [web:11][web:12][web:14]  
- **Size:** 4,424 records and 36 input features, plus one target column `Target`. [web:11][web:15]  
- **Target Variable:** Multi-class label with three outcomes:
  - `Dropout`
  - `Enrolled`
  - `Graduate` [web:11][web:15]  

This dataset has been used in several published works on student dropout and academic success, making it suitable and credible for an undergraduate research paper. [web:60][web:21]

## 3. Problem Statement

**Goal:**  
Given a student’s demographic, socio-economic, and academic information at or near enrollment, predict whether the student will **Dropout**, remain **Enrolled**, or **Graduate**, and compare the performance of different machine learning algorithms on this task. [web:11][web:22]

**Research Questions:**  

1. Among Logistic Regression, Random Forest, and XGBoost, which algorithm provides the best predictive performance for student dropout prediction on the UCI dataset? [web:22][web:24]  
2. How much improvement can be obtained by moving from **baseline models** (default hyperparameters) to **lightly tuned models** for Random Forest and XGBoost? [web:21][web:149]

## 4. Proposed Methods

Three supervised classification models will be studied:

1. **Logistic Regression**  
   - Simple, interpretable baseline model. [web:22]

2. **Random Forest**  
   - Ensemble of decision trees, robust for tabular data and useful for feature importance analysis. [web:56][web:118]

3. **XGBoost**  
   - Gradient boosting algorithm known for strong performance on structured/tabular datasets. [web:24][web:135]

The study will be conducted in two stages:

- **Baseline stage:** Train each model with reasonable default parameters and standard preprocessing to establish baseline performance. [web:22][web:151]  
- **Tuning stage:** Apply light hyperparameter tuning for Random Forest and XGBoost (e.g., `n_estimators`, `max_depth`, `learning_rate`, `subsample`) and compare the gains over the baselines. [web:118][web:121][web:149]

## 5. Experimental Workflow

1. **Data Loading**
   - Load the CSV file from `data/raw/data.csv` (semicolon-separated) into a pandas DataFrame. [web:11][web:15]

2. **Preprocessing**
   - Encode any categorical variables into numeric form (label encoding or similar).  
   - Separate features (`X`) and target (`y = Target`).  
   - Standardize numeric features when training Logistic Regression (via `StandardScaler`). [attached_file:1][web:110]  

3. **Train–Test Split**
   - Split the dataset into **80% training** and **20% testing** sets using stratified sampling to preserve class proportions. [web:22][web:26]

4. **Baseline Model Training**
   - Train:
     - Logistic Regression (with scaling),
     - Random Forest (default or simple parameters),
     - XGBoost (default or simple parameters).  
   - Evaluate each model on the test set using:
     - Accuracy,
     - Precision, Recall, F1-score per class,
     - Macro and weighted F1-scores. [web:22][web:122][web:131]

5. **Tuned Model Training (Planned)**
   - For Random Forest, explore a small grid of hyperparameters such as:
     - `n_estimators` (e.g., 200–500),
     - `max_depth`,
     - `min_samples_split`,
     - `max_features`. [web:118][web:149]  
   - For XGBoost, explore:
     - `n_estimators`,
     - `max_depth`,
     - `learning_rate`,
     - `subsample`,
     - `colsample_bytree`. [web:121][web:135]  
   - Compare baseline vs tuned models using the same metrics.

6. **Comparison and Analysis**
   - Create tables comparing metrics across all models (baseline and tuned).  
   - Analyze feature importance from Random Forest and XGBoost to identify key factors associated with dropout and success. [web:56][web:129]

## 6. Expected Outcomes

- A clear baseline comparison of Logistic Regression, Random Forest, and XGBoost on the UCI dropout dataset. [web:11][web:22]  
- Quantitative evidence showing whether light hyperparameter tuning significantly improves performance over baseline settings. [web:24][web:149]  
- Visualizations such as confusion matrices and feature importance plots that can be used directly in the Results section of the research paper. [web:22][web:56]

## 7. Publication Target (Tentative)

- **Primary:** IEEE or similar conferences in computer science/education technology that accept applied machine learning studies using educational data. [web:22][web:24]  
- **Backup:** Scopus / UGC-CARE indexed journals on educational data mining or applied ML, depending on supervisor guidance. [web:21][web:28]
