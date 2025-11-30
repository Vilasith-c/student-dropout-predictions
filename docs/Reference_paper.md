# Reference Papers for Student Dropout Prediction

This document lists key reference papers for the project **“Comparative Analysis of Machine Learning Algorithms for Early Prediction of Student Dropout”**, along with download links and how each paper relates to this work.

---

## 1. Core dataset and problem formulation

### 1.1 Early prediction case study (UCI dataset source)

- **Title:** Early prediction of student’s performance in higher education: a case study  
- **Authors:** Martins, M. V., Toledo, D., Machado, J., Baptista, L., Realinho, V.  
- **Venue:** Book chapter / conference paper in an intelligent systems/education context (UCI “Predict Students’ Dropout and Academic Success” dataset origin). [web:11][web:19][web:67]  
- **Link (dataset page with citation info):**  
  - UCI Repository: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success [web:11][web:15]  
- **How it relates to this project:**  
  - Defines the **same higher‑education dataset** (≈4,424 students, 35–36 variables) and the **three‑class target** (Dropout, Enrolled, Graduate) that this project uses. [web:11][web:15]  
  - Provides the **official dataset citation** and original problem framing; this project extends their work by performing a focused comparison of Logistic Regression, Random Forest, and XGBoost on the same task.

---

## 2. High‑reputation general ML dropout studies

### 2.1 Machine learning optimization for dropout prediction (Nature portfolio)

- **Title:** Student dropout prediction through machine learning optimization  
- **Venue:** Scientific Reports (Nature portfolio) – open access. [web:24]  
- **Link (HTML/PDF):**  
  - https://www.nature.com/articles/s41598-025-93918-1 [web:24]  
- **How it relates to this project:**  
  - Confirms that **machine learning is a strong approach** for dropout/failure prediction and explores **optimized models and feature importance**, which motivates using powerful models like Random Forest and XGBoost in this project. [web:24]  
  - Can be cited in the **Introduction** and **Related Work** to show that dropout prediction is an active and important research topic and to suggest **future work** on hyperparameter optimization beyond baseline models.

### 2.2 Empirical comparison of ML models for dropout (ScienceDirect)

- **Title:** Predicting student dropouts with machine learning: An empirical study  
- **Venue:** Peer‑reviewed journal on education / information systems (ScienceDirect). [web:22]  
- **Link (HTML/PDF via publisher):**  
  - https://www.sciencedirect.com/science/article/pii/S0160791X24000228 [web:22]  
- **How it relates to this project:**  
  - Provides an **empirical comparison of multiple ML algorithms** for dropout prediction using educational data (transcripts, demographics, LMS logs). [web:22]  
  - Supports the decision to **compare several classifiers** (LogReg, RF, XGBoost) and to evaluate them using metrics beyond accuracy (e.g., precision, recall, F1). [web:22]  

### 2.3 Open‑access dropout prediction tool (PMC)

- **Title:** Student Dropout Prediction  
- **Venue:** Open‑access article indexed in PubMed Central. [web:26]  
- **Link (full article PDF/HTML):**  
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC7334184/ [web:26]  
- **How it relates to this project:**  
  - Describes a **practical tool** that uses several ML models for predicting first‑year student dropout and discusses interventions. [web:26]  
  - Useful for **motivation** (why early prediction matters) and for justifying the selection of classical ML models as strong baselines.

---

## 3. Papers closely aligned with this project’s idea

### 3.1 XGBoost on the same UCI dataset

- **Title:** Predict Students' Dropout and Academic Success with XGBoost  
- **Venue:** Journal of Education and Computer Applications (JECA) or similar; open‑access PDF. [web:53][web:60]  
- **Link (HTML):**  
  - Article page: https://jeca.aks.or.id/jeca/article/view/13 [web:53]  
- **Link (PDF download):**  
  - PDF: https://jeca.aks.or.id/jeca/article/download/13/6/61 [web:60]  
- **How it relates to this project:**  
  - Uses **the same UCI “Predict Students’ Dropout and Academic Success” dataset** and applies **XGBoost** to classify Dropout / Enrolled / Graduate. [web:11][web:53][web:60]  
  - Provides a **direct performance benchmark** for XGBoost on this dataset; this project compares XGBoost against Logistic Regression and Random Forest under a unified preprocessing and evaluation pipeline.

### 3.2 RF and XGBoost for higher‑education dropout

- **Title:** Prediction of Higher Education Student Dropout based on [Machine Learning Methods]  
- **Venue:** Engineering, Technology & Applied Science Research (ETASR) or similar journal. [web:28]  
- **Link (HTML/PDF):**  
  - Article page (PDF download inside): https://etasr.com/index.php/ETASR/article/view/8644 [web:28]  
- **How it relates to this project:**  
  - Evaluates **Random Forest and XGBoost** (and possibly other models) for **higher‑education dropout prediction**, often with datasets of similar size (~4,000+ records). [web:28]  
  - Strong support for choosing **RF and XGBoost as key models** and for reporting standard metrics (accuracy, precision, recall, F1) as this project plans.

### 3.3 Student dropout prediction using ML techniques (broad comparison)

- **Title:** Student Dropout Prediction Using Machine Learning Techniques  
- **Venue:** International Journal of Intelligent Systems and Applications in Engineering (IJISAE) or similar. [web:64]  
- **Link (HTML/PDF):**  
  - Article page: https://ijisae.org/index.php/IJISAE/article/view/2276 [web:64]  
- **How it relates to this project:**  
  - Compares **multiple ML classifiers** on student data and examines which features and algorithms perform best. [web:64]  
  - Helps position this project as a **focused comparative study** (LogReg vs RF vs XGBoost) on a specific, widely used dataset.

---

## 4. Advanced / future‑work oriented references

### 4.1 Intuitionistic fuzzy sets + XGBoost (ACM)

- **Title:** Predictive Modeling of Student Dropout Using Intuitionistic Fuzzy Sets and XGBoost  
- **Venue:** ACM‑published conference/journal article. [web:59]  
- **Link (HTML/PDF via ACM):**  
  - https://dl.acm.org/doi/fullHtml/10.1007/978-3-030-52237-7_11 (or similar DOI path) [web:59]  
- **How it relates to this project:**  
  - Extends XGBoost with **intuitionistic fuzzy sets** to handle uncertainty in dropout prediction. [web:59]  
  - Can be cited in **Related Work / Future Work** to show that more advanced hybrid approaches exist beyond the baseline models used here.

### 4.2 Early prediction in higher education (EDM)

- **Title:** Early Prediction of Student Dropout in Higher Education using [Machine Learning]  
- **Venue:** Educational Data Mining (EDM) 2024 short paper. [web:62]  
- **Link (proceedings entry with PDF):**  
  - https://educationaldatamining.org/edm2024/proceedings/2024.EDM-short-papers.32/index.html [web:62]  
- **How it relates to this project:**  
  - Focuses on **early‑semester prediction** using ML (often including boosting models) and evaluates predictive performance at different time points. [web:62]  
  - Supports the **“early prediction”** motivation and can be referenced when discussing the potential use of this project’s models for interventions.

---

## 5. How these references support the new paper

- **Motivation and importance:**  
  - Nature/Scientific Reports, ScienceDirect, and PMC papers show that **student dropout prediction is a significant and active research area** that benefits from machine learning. [web:22][web:24][web:26]  

- **Dataset and problem definition:**  
  - Martins et al.’s case study and the UCI dataset entry define the **exact dataset, features, and three‑class target** used in this project, ensuring a strong and citable foundation. [web:11][web:15][web:19][web:67]  

- **Model choice and metrics:**  
  - Multiple papers demonstrate the effectiveness of **tree‑based and boosting models** (Random Forest, XGBoost) for dropout prediction and emphasize evaluating models with **accuracy, precision, recall, and F1‑score**, aligning directly with this project’s methodology. [web:22][web:24][web:28][web:53][web:56][web:64]  

- **Positioning and future work:**  
  - Advanced methods (optimization, fuzzy logic, ensembles) provide **natural extensions** beyond this project’s baseline comparisons, giving material for the **Discussion and Future Work** sections. [web:24][web:55][web:59][web:62]  
