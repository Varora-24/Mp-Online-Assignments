# Adult Census Income Classification Project

**Dataset**: Adult Census Income Dataset (Source: UCI Machine Learning Repository / Kaggle)  
**Objective**: Build and evaluate a complete machine learning classification pipeline to predict whether an individual's annual income exceeds $50,000 based on demographic and employment features.

## Dataset Understanding
The dataset consists of 48,842 observations from the 1994 Census database. The target variable is binary: >50K or <=50K.
It exhibits a significant class imbalance with ~76% earning <=50K and 24% earning >50K.

## Data Cleaning & Feature Engineering
- **Missing Values**: Missing values are represented as `?`. They were replaced with standard NaNs and imputed using mode imputation.
- **Encoding**: Categorical features were encoded using `LabelEncoder`.
- **Scaling**: Continuous features were standardized using `StandardScaler`.

## Performance Evaluation
Five distinct classification models were tested:

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.8261 | 0.6724 | 0.4532 | 0.5413 | 0.8573 |
| Decision Tree | 0.8143 | 0.6121 | 0.5641 | 0.5871 | 0.7482 |
| Random Forest | 0.8529 | 0.7312 | 0.5894 | 0.6527 | 0.8991 |
| KNN (K=5) | 0.8245 | 0.6419 | 0.5281 | 0.5794 | 0.8314 |
| SVM (RBF Kernel) | 0.8431 | 0.7291 | 0.5103 | 0.6004 | 0.8845 |

## Conclusion
- **Top Performer**: Random Forest demonstrated the strongest overall capability, achieving the highest Accuracy (85.29%) and a superior ROC-AUC score of 0.8991. Its ensemble nature effectively reduces variance and captures complex interactions.
- **The Recall Challenge**: Across all algorithms, Recall remains lower than Precision due to the 76/24 class imbalance. Future optimization could introduce minority oversampling (SMOTE) or adjusted class weights.
