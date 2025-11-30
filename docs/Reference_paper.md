# Reference Papers for Student Dropout Prediction

This document lists key reference papers for **“Comparative Analysis of Machine Learning Algorithms for Early Prediction of Student Dropout”**, with links and how each paper supports this work.

---

## 1. Core dataset and problem formulation

### 1.1 UCI dataset and original case study

- **Title:** Early prediction of student’s performance in higher education: a case study  
- **Authors:** Martins et al.  
- **Context:** Origin study behind the UCI “Predict Students’ Dropout and Academic Success” dataset (4,424 students, 36 attributes, 3-class target). [web:11][web:15][web:65][web:67]  
- **Link (dataset page + citation):**  
  - UCI: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success [web:11][web:15]  
- **Relation to this project:**  
  - Defines the **same dataset and target classes** used here, making it the primary data reference.  
  - This project extends the case study by running a clean comparison of Logistic Regression, Random Forest, and XGBoost (baseline and tuned) on the same problem.

---

## 2. High-reputation general ML dropout studies

### 2.1 Machine learning optimization for dropout prediction

- **Title:** Student dropout prediction through machine learning optimization  
- **Venue:** Scientific Reports (Nature portfolio). [web:24]  
- **Link:** https://www.nature.com/articles/s41598-025-93918-1 [web:24]  
- **Relation:**  
  - Demonstrates that **optimized ML models** can improve dropout prediction performance and highlights the role of feature importance.  
  - Supports the decision to investigate **hyperparameter tuning** after establishing baselines.

### 2.2 Empirical study of dropout prediction models

- **Title:** Predicting student dropouts with machine learning: An empirical study  
- **Venue:** Peer-reviewed journal on higher-education analytics (ScienceDirect). [web:22]  
- **Link:** https://www.sciencedirect.com/science/article/pii/S0160791X24000228 [web:22]  
- **Relation:**  
  - Compares multiple ML algorithms on educational data and emphasizes evaluation with Accuracy, Precision, Recall, and F1-score.  
  - Justifies the multi-model comparison and the use of macro/weighted F1 in this project.

### 2.3 Open-access dropout prediction tool

- **Title:** Student Dropout Prediction  
- **Venue:** Open-access article indexed in PubMed Central. [web:26]  
- **Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC7334184/ [web:26]  
- **Relation:**  
  - Illustrates a practical tool using classical ML models for predicting first-year dropout.  
  - Provides motivation and context for applying interpretable models like Logistic Regression as baselines.

---

## 3. Papers using the same or very similar datasets

### 3.1 XGBoost on UCI dropout dataset

- **Title:** Predict Students' Dropout and Academic Success with XGBoost  
- **Venue:** Journal of Education and Computer Applications (open access). [web:53][web:60]  
- **Links:**  
  - Article: https://jeca.aks.or.id/jeca/article/view/13 [web:53]  
  - PDF: https://jeca.aks.or.id/jeca/article/download/13/6/61 [web:60]  
- **Relation:**  
  - Uses the **same UCI dataset** and trains XGBoost for the three-class dropout/success task.  
  - Serves as an external benchmark for the XGBoost results in this project (baseline and tuned).

### 3.2 Random Forest and XGBoost for higher-education dropout

- **Title:** Prediction of Higher Education Student Dropout based on Machine Learning Techniques  
- **Venue:** Engineering, Technology & Applied Science Research (ETASR). [web:28][web:142]  
- **Link:** https://etasr.com/index.php/ETASR/article/view/8644 [web:28]  
- **Relation:**  
  - Compares Random Forest and XGBoost on a dataset of similar size and context (higher-education dropout).  
  - Supports choosing these two models as strong contenders and motivates exploring hyperparameter tuning.

---

## 4. Broader ML techniques and tuning

### 4.1 General ML techniques for dropout prediction

- **Title:** Student Dropout Prediction Using Machine Learning Techniques  
- **Venue:** International Journal of Intelligent Systems and Applications in Engineering. [web:64]  
- **Link:** https://ijisae.org/index.php/IJISAE/article/view/2276 [web:64]  
- **Relation:**  
  - Evaluates several classifiers for dropout prediction.  
  - Provides context for classifier selection and motivates comparing multiple models rather than relying on one.

### 4.2 Oversampling and optimization approaches

- **Title:** Optimizing Dropout Prediction in University Using Oversampling Techniques and Machine Learning Models  
- **Venue:** International Journal of Emerging Technologies in Learning or similar. [web:149]  
- **Link:** https://www.ijiet.org/vol14/IJIET-V14N8-2133.pdf [web:149]  
- **Relation:**  
  - Shows that **data-level methods and hyperparameter optimization** can increase predictive performance.  
  - Relevant when discussing future work beyond simple hyperparameter tuning.

---

## 5. Advanced / future-work references

### 5.1 Intuitionistic fuzzy sets + XGBoost

- **Title:** Predictive Modeling of Student Dropout Using Intuitionistic Fuzzy Sets and XGBoost  
- **Venue:** ACM / Springer proceedings. [web:59][web:145]  
- **Link:** https://dl.acm.org/doi/10.1145/3696271.3696288 [web:59]  
- **Relation:**  
  - Extends XGBoost with fuzzy logic to handle uncertainty.  
  - Can be cited in the **Future Work** section as an example of more advanced hybrid models.

### 5.2 Early prediction in higher education

- **Title:** Early Prediction of Student Dropout in Higher Education using Machine Learning  
- **Venue:** Educational Data Mining (EDM) 2024. [web:62][web:100]  
- **Link:** https://educationaldatamining.org/edm2024/proceedings/2024.EDM-short-papers.32/index.html [web:62]  
- **Relation:**  
  - Focuses on early-semester prediction using ML and compares performance at different time points.  
  - Supports framing this project as **early prediction** and motivates discussing when predictions should be made.

---

## 6. How these references support this work

- They establish that student dropout prediction is an important and active field where ML is widely used. [web:22][web:24][web:26]  
- They justify the selection of Logistic Regression, Random Forest, and XGBoost as core models and provide external performance benchmarks. [web:22][web:28][web:60]  
- They show that hyperparameter tuning and more advanced techniques can increase accuracy and F1, which this project explores at a modest level and discusses in relation to complexity. [web:24][web:56][web:149]
