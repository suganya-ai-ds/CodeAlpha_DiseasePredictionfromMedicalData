# 🎯 MEDSCOPE AI — Technical Interview Preparation & Q&A

> **Preparation Guide:** In-depth interview questions, theoretical foundations, architectural defenses, and machine learning principles implemented in MEDSCOPE AI.

---

### Q1: Why did you separate the preprocessing pipeline and fit it only on the training set?
**Answer:**  
Fitting preprocessing transformers (such as `SimpleImputer` or `StandardScaler`) on the entire dataset prior to splitting introduces **data leakage** (specifically distribution leakage). The mean and standard deviation of the test set would leak into the training representations, yielding overly optimistic evaluation metrics that fail in real-world clinical inference. In MEDSCOPE AI, all transformations are encapsulated in Scikit-Learn `Pipeline` and `ColumnTransformer` objects that are strictly `.fit()` on the training split only and merely `.transform()` the held-out test data.

---

### Q2: Why did you evaluate both ROC-AUC and Precision-Recall curves?
**Answer:**  
While ROC-AUC is standard for evaluating overall discrimination ability across all classification thresholds, it can produce an overly optimistic assessment when class distributions are imbalanced. The false positive rate (FPR = FP / (FP + TN)) can stay low simply because the true negative count (TN) is large. Precision-Recall curves focus strictly on the minority (positive) class, providing a much more sensitive evaluation of model performance in skewed clinical disease prevalence scenarios.

---

### Q3: Why did Logistic Regression outperform Random Forest on Breast Cancer and Heart Disease?
**Answer:**  
1. **Sample Size & Feature Geometry:** The Wisconsin Diagnostic dataset has 30 continuous features and 569 instances. The features represent normalized cell geometry measurements where the class boundary is largely linear and separable in hyper-dimensional space. Random Forest can over-partition small sample regions with axis-aligned splits, leading to slight variance inflation.
2. **Occam's Razor & Regularization:** Regularized Logistic Regression ($L_2$ penalty) prevents overfitting on small medical cohorts (e.g., $N=303$ for Cleveland Heart Disease) and delivers smoother probability calibration than tree ensembles without extensive calibration tuning.

---

### Q4: How do you handle missing values in clinical data responsibly?
**Answer:**  
In MEDSCOPE AI:
- Missingness is audited and reported transparently in `data_quality.json`.
- Missing values in numerical features (e.g., `ca` in heart disease) are imputed using the **median** computed solely from training folds to be robust against outliers.
- In categorical features, missingness is imputed using the most frequent category or an explicit `'missing'` category.
- Clinically impossible zero values (such as zero blood pressure or zero glucose in diabetes datasets) are explicitly analyzed before treating them as missing, rather than applying automated indiscriminate replacements.

---

### Q5: How did you validate unseen categorical values during inference?
**Answer:**  
In our `ColumnTransformer` preprocessor in `src/preprocessing.py`, `OneHotEncoder` is instantiated with `handle_unknown='ignore'`. This ensures that if a user or novel patient input contains an unseen categorical level during runtime inference, the encoder assigns zeros to the corresponding one-hot columns rather than throwing an unhandled runtime exception.

---

### Q6: Why did you deduplicate the Early Stage Diabetes dataset?
**Answer:**  
The raw Sylhet Diabetes dataset contains 520 rows, but audit revealed multiple identical survey responses across all 16 features. If identical records appear in both the training and test splits, the test set is no longer an independent evaluation (duplicate record leakage). By removing duplicates and retaining 251 unique respondent profiles, we ensured that test metrics accurately reflect out-of-sample generalization.

---

### Q7: Can model feature importance be interpreted as medical causation?
**Answer:**  
**No, absolutely not.** In MEDSCOPE AI, we include a mandatory explainability disclaimer. Feature importance (whether Random Forest Gini impurity reduction or Logistic Regression coefficients) reflects mathematical correlation with the target within the specific training sample distribution. It does not establish biological causality or physiological mechanism. Confounders, observational bias, and correlation between symptoms can shift feature weights without altering the underlying pathology.

---

### Q8: What automated tests did you implement and why?
**Answer:**  
We engineered 49 automated unit and integration tests using `pytest` across 5 dedicated test modules:
1. `test_data_validation.py`: Verifies error handling on empty data, missing target columns, single-class targets, and NaN detection.
2. `test_preprocessing.py`: Verifies schema column presence, pipeline transformation output shapes, and unknown category handling.
3. `test_metrics.py`: Verifies boundary conditions ($[0, 1]$ bounds on accuracy, F1, ROC-AUC) and confusion matrix dimensions.
4. `test_prediction.py`: Validates user inputs against required schemas, rejecting non-numeric or infinite values.
5. `test_app_helpers.py`: Tests model artifact loading, directory setups, and missing-file exception handling.

---

### Q9: How is the application prepared for cloud deployment?
**Answer:**  
The application is pre-configured for Streamlit Community Cloud or Docker containers:
- Explicit `.streamlit/config.toml` configuring port and production theming.
- Lightweight, locked `requirements.txt` with compatible versions for Python 3.10–3.13.
- `.gitignore` preventing commit of cache files, virtual environments, or heavy temporary artifacts.
- Lightweight serialized model pipelines ($<1\text{ MB}$ each) compatible with git and server memory constraints.
