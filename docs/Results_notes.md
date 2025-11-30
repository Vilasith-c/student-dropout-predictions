# Results Notes – Student Dropout Prediction

This file records intermediate and final results from experiments for later use in the IEEE paper.

---

## 1. Dataset summary

- **Dataset:** Predict Students' Dropout and Academic Success (UCI). 
- **Shape:** 4,424 rows × 37 columns (36 features + `Target`). 
- **Target distribution (full dataset):**
  - Graduate ≈ 49.9%  
  - Dropout ≈ 32.1%  
  - Enrolled ≈ 17.9%  
  indicating a mildly imbalanced three-class classification problem.

---

## 2. Experimental setup

- **Train–test split:** 80% train, 20% test, stratified by `Target`. 
- **Preprocessing:**
  - CSV loaded with `sep=";"`.  
  - Any object-type features label-encoded (dataset is mostly numeric already).
  - For Logistic Regression, features standardized using `StandardScaler` in a scikit-learn `Pipeline`. [attached_file:1]
- **Evaluation metrics:**  
  - Overall accuracy.  
  - Per-class Precision, Recall, F1-score.  
  - Macro and weighted F1-scores.

---

## 3. Baseline models (default / simple settings)

### 3.1 Logistic Regression (with StandardScaler)

- **Model:** `LogisticRegression(max_iter=2000, solver="lbfgs")` in a pipeline with `StandardScaler`. [attached_file:1]
- **Test set size:** 885 samples. 

Per-class metrics:

| Class     | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| Dropout  | 0.79      | 0.77   | 0.78     | 284     |
| Enrolled | 0.52      | 0.33   | 0.41     | 159     |
| Graduate | 0.80      | 0.93   | 0.86     | 442     |

Overall:

- Accuracy: **0.7684 ≈ 0.77**  
- Macro F1: **0.6826 ≈ 0.68**  
- Weighted F1: **0.7531 ≈ 0.75**

### 3.2 Random Forest (baseline)

- **Model:** `RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)`.  

Per-class metrics:

- Dropout: precision 0.82, recall 0.75, F1 0.78  
- Enrolled: precision 0.58, recall 0.38, F1 0.46  
- Graduate: precision 0.79, recall 0.93, F1 0.85  

Overall:

- Accuracy: **0.7740 ≈ 0.77**  
- Macro F1: **0.7001 ≈ 0.70**  
- Weighted F1: **0.7611 ≈ 0.76**

### 3.3 XGBoost (baseline)

- **Model:** `XGBClassifier(objective="multi:softmax", num_class=3, n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss", random_state=42, n_jobs=-1)`. 

Per-class metrics:

- Dropout: precision 0.80, recall 0.75, F1 0.77  
- Enrolled: precision 0.54, recall 0.44, F1 0.48  
- Graduate: precision 0.81, recall 0.90, F1 0.85  

Overall:

- Accuracy: **0.7684 ≈ 0.77**  
- Macro F1: **0.7036 ≈ 0.70**  
- Weighted F1: **0.7614 ≈ 0.76**

### 3.4 Baseline comparison table

| Model                | Accuracy | Macro F1 | Weighted F1 |
|----------------------|----------|----------|-------------|
| Logistic Regression  | 0.7684   | 0.6826   | 0.7531      |
| Random Forest        | 0.7740   | 0.7001   | 0.7611      |
| XGBoost              | 0.7684   | 0.7036   | 0.7614      |

Interpretation:

- All three models achieve similar overall accuracy (~0.77).  
- Random Forest has the **highest accuracy and weighted F1**, while XGBoost has the **best macro F1**, indicating slightly better balance across classes. 
- Logistic Regression is a strong baseline but slightly underperforms the tree-based models on macro/weighted F1.

---

## 4. Planned tuned models (TODO)

To study the impact of hyperparameter tuning:

- **Random Forest (tuned):**
  - Try small grids for:
    - `n_estimators`: e.g., [300, 500],
    - `max_depth`: e.g., [None, 10, 20],
    - `min_samples_split`: e.g., [2, 5],
    - `max_features`: e.g., ["sqrt", "log2"].
  - Record new Accuracy, Macro F1, Weighted F1 and compare to baseline.

- **XGBoost (tuned):**
  - Try small grids for:
    - `n_estimators`: e.g., [300, 500],
    - `max_depth`: e.g., [4, 6],
    - `learning_rate`: e.g., [0.05, 0.1],
    - `subsample`: e.g., [0.8, 1.0],
    - `colsample_bytree`: e.g., [0.6, 0.8].   
  - Record metrics and compare to baseline.

A final table will summarize **Baseline vs Tuned** for each model, and the paper will discuss whether the observed performance gains justify the extra complexity.

## 5. Figures generated

- Confusion matrices:
  - `cm_logistic_regression.png`
  - `cm_random_forest.png`
  - `cm_xgboost.png`
- Feature importance:
  - `fi_random_forest.png`
  - `fi_xgboost.png`

These plots are stored under `reports/figures/` and will be referenced in the paper to illustrate class-wise performance and the most influential features.