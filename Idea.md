# Project Idea: Student Dropout Prediction

## 1. Motivation

Student dropout is a major problem for higher education institutions because it leads to financial loss for colleges and wasted time and effort for students. Early identification of at-risk students can help universities provide timely interventions and improve overall academic success.

This project aims to build and compare machine learning models that can predict whether a student is likely to **Dropout**, **Remain Enrolled**, or **Graduate**, using historical academic and demographic data.

## 2. Dataset

- **Name:** Predict Students' Dropout and Academic Success  
- **Source:** UCI Machine Learning Repository / Kaggle  
- **Size:** 4,424 records and 36 features (demographic, socio-economic, and academic variables). 
- **Target Variable:** Multi-class label with three outcomes:
  - `Dropout`
  - `Enrolled`
  - `Graduate` 

This dataset is clean, well-documented, and suitable for classification tasks, making it appropriate for an undergraduate research project.

## 3. Problem Statement

**Goal:**  
Given a student’s demographic and academic information at an early stage, predict whether the student will **Dropout**, remain **Enrolled**, or **Graduate**, and compare the performance of different machine learning algorithms on this task.

**Research Question:**  
Which machine learning algorithm among Logistic Regression, Random Forest, and XGBoost provides the best predictive performance for early student dropout prediction on the UCI “Predict Students' Dropout and Academic Success” dataset?

## 4. Proposed Methods

Three supervised classification models will be trained and evaluated:

1. **Logistic Regression**  
   - Acts as a simple, interpretable baseline model.

2. **Random Forest**  
   - Ensemble of decision trees, typically strong on tabular data and useful for feature importance analysis. 

3. **XGBoost**  
   - Gradient boosting algorithm known for high performance on structured/tabular datasets and competitions. 

All models will be implemented in Python using standard machine learning libraries.

## 5. Experimental Workflow

1. **Data Loading**
   - Load the dataset from a CSV file into a pandas DataFrame. 

2. **Preprocessing**
   - Handle missing values if present.
   - Encode categorical variables (label or one-hot encoding as appropriate).
   - Separate features (X) and target (y).
   - Optionally address class imbalance using class weights or simple resampling. 

3. **Train–Test Split**
   - Split the data into **80% training** and **20% testing** sets with stratification on the target variable.

4. **Model Training**
   - Train Logistic Regression, Random Forest, and XGBoost models on the training data with reasonable default or lightly tuned hyperparameters. 

5. **Evaluation**
   - Evaluate each model on the test set using:
     - Accuracy
     - Precision
     - Recall
     - F1-score (macro / weighted) 
   - Generate confusion matrices for each model.

6. **Comparison and Analysis**
   - Create a comparison table of all metrics for the three models.
   - Identify the best-performing model.
   - Analyze feature importances (from Random Forest and XGBoost) to understand which factors most influence dropout risk. 

## 6. Expected Outcomes

- A clear comparison of Logistic Regression, Random Forest, and XGBoost on student dropout prediction using a real-world dataset. 
- Identification of the best-performing algorithm and discussion of why it performs better.  
- Visualizations (confusion matrices, feature importance plots) that can be directly used in the research paper’s **Results** and **Discussion** sections.  

## 7. Publication Target (Tentative)

- Primary: IEEE or similar computer science/education technology conferences that accept applied machine learning studies.
- Backup: A Scopus / UGC-CARE indexed journal focusing on educational data mining or applied ML, depending on guidance from the academic supervisor. 
