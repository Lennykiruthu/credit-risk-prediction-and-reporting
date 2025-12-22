A comprehensive machine learning system for predicting loan default risk with real-time prediction capabilities and explainable AI using SHAP analysis.

## 🎯 Project Overview 
This project builds an end-to-end machine learning pipeline that: 
- Predicts the probability of loan default for credit risk assessment 
- Provides interpretable explanations for each prediction using SHAP values 
- Offers a user-friendly Streamlit interface for real-time underwriting decisions

**Key Features:** 
- 📊 Comprehensive EDA and feature engineering 
- 🤖 Multiple ML approaches (Classical, Neural Networks, Ensembling) 
- 🔍 SHAP-based model explainability 
- 🚀 Production-ready Streamlit deployment 
- 📈 Achieved on average 70% ROC-AUC score on test sets

## 📁 Front-end Examples

![credit-risk-prediction-and-reporting](loan_page_001.png)
![credit-risk-prediction-and-reporting](loan_page_002.png)

 
--- 
## 🚀 Quick Start 
### Prerequisites 
```bash 
Python 3.8+ 
pip or conda 
``` 
### Installation 
1. **Clone the repository** 
```bash 
git clone https://github.com/yourusername/loan-default-prediction.git 
cd loan-default-prediction 
``` 
2. **Install dependencies** 
```bash 
pip install -r requirements.txt 
``` 
3. **Run the Streamlit app** 
```bash 
streamlit run credit-app.py
```
 The app will open in your browser at `http://localhost:8501` 
 
 ---
## 📊 Dataset 
**Source:** [Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
**Description:** 
- `Samples:` 2.3 Million loan applications 
- `Features:` 151 features including FICO scores, annual income, loan amount, term, loan status, DTI etc. 
- `Target:` Binary classification (Default = 1, Paid = 0) 
- `Class distribution:` 20% defaults, 80% non-defaults

**Engineered Features**
- `credit_history_years`: Length of the borrower’s credit history in years (derived from `earliest_cr_line`)    
- `income_to_loan_ratio`: Ratio of annual income to loan amount, indicating repayment capacity
- `account_utilization`: Proportion of open credit accounts relative to total accounts    
- `fico_score`: Average FICO credit score computed from `fico_range_low` and `fico_range_high`
- `avg_credit_per_account`: Average revolving balance per open credit account
- `deliquency_rate`: Ratio of delinquent accounts to total accounts, reflecting past repayment behavior

**Target Variable Construction: Loan Default Indicator**

|Feature Name|Description|
|---|---|
|`loan_status`|Original categorical loan outcome provided by the Lending Club dataset, indicating the repayment state of each loan.|
|`default_flag`|Binary target variable derived from `loan_status`, representing borrower default risk. A value of **1** indicates default or high credit risk, while **0** indicates non-default or low credit risk.|
**Mapping Logic for `default_flag`**

| Loan Status Category                                 | Default Flag        | Rationale                                                        |
| ---------------------------------------------------- | ------------------- | ---------------------------------------------------------------- |
| Charged Off                                          | 1                   | Loan has been written off as a loss, indicating failure to repay |
| Default                                              | 1                   | Explicit default designation                                     |
| Late (31–120 days)                                   | 1                   | Severe delinquency strongly associated with default risk         |
| Late (16–30 days)                                    | 1                   | Early-stage delinquency with elevated default probability        |
| Does not meet the credit policy. Status: Charged Off | 1                   | Charged-off loan despite failing credit policy requirements      |
| Fully Paid                                           | 0                   | Loan successfully repaid in full                                 |
| Does not meet the credit policy. Status: Fully Paid  | 0                   | Loan repaid despite not meeting credit policy standards          |
| Current                                              | Excluded / Cautious | Loan is active; final outcome unknown                            |
| In Grace Period                                      | Excluded / Cautious | Temporary delinquency that may self-correct                      |


---
## 🔬 Methodology
### **Classical ML Models** 

| Model               | ROC-AUC | Accuracy | Best F1-Score | Best Binary Threshold |
| ------------------- | ------- | -------- | ------------- | --------------------- |
| Logistic Regression | 0.6903  | 0.64     | 0.429         | 0.504                 |
| XGBoost             | 0.6917  | 0.64     | 0.430         | 0.312                 |
| LightGBM            | 0.6984  | 0.63     | 0.435         | 0.238                 |
| CatBoost            | 0.6968  | 0.64     | 0.434         | 0.248                 |
| Random Forest       | 0.6807  | 0.63     | 0.422         | 0.453                 |
| Neural Network      | 0.7068  | 0.65     | 0.442         | 0.505                 |

- The meta-model of the classical models highlighted that cat boost classifier and lightgbm classifier push our predictions towards default and non-default more accurately:
	![credit-risk-prediction-and-reporting](classical-meta-model.png)
### **Neural Network** 
- **Architecture**: Fully connected feed-forward neural network with **64 and 32 hidden units**
- **Activation Functions**: ReLU for hidden layers, Sigmoid for output layer
- **Loss Function**: Binary cross-entropy
- **Optimizer**: Adam optimizer (learning rate = 1e-5)
- **Output**: Single neuron producing default probability

### **Ensemble Model** 
- Voting/Stacking ensemble of top 2 classical models and the neural network
- **Final Performance:** ROC-AUC = 0.6935, Accuracy = 63% 

 ### **Model Interpretation (SHAP)** 
 - TreeExplainer for LightGBM model 
 - Global feature importance analysis 
 - Local explanations for individual predictions 
 - Waterfall plots for real-time decision support

---
## 🧠 Key Insights & Learnings

### **1. Performance Ceiling Is Real**

Across classical ML, neural networks, and ensembles, model performance consistently plateaued around **ROC-AUC ≈ 0.69–0.71**, even after confirming no data leakage. Increasing model complexity yielded minimal gains, highlighting that **feature signal—not model choice—was the main limiting factor**.

---

### **2. Credit Lending Is Inherently Difficult**

With a realistic **20/80 default imbalance**, even well-tuned models struggled to strongly separate defaulters from non-defaulters. Many borrowers who default appear statistically similar to those who repay, reflecting real-world uncertainty rather than modeling failure.

---

### **3. A Small Set of Features Dominates Predictions**

SHAP analysis showed consistent reliance on:

- Debt-to-income ratio
- Income
- Account utilization
- Loan term
- FICO score (or strong proxies)
- 
Even when FICO was removed, predictions changed only marginally, indicating **high feature redundancy and correlation**.

---

### **4. Complex Models Learn Similar Boundaries**

Neural networks, tree-based models, and ensembles relied on **nearly identical feature patterns**, explaining why ensembling and deep learning offered limited uplift over simpler models.

---

### **5. Thresholds Matter More Than Accuracy**

Optimal classification thresholds varied significantly by model. A fixed 0.5 cutoff is not operationally meaningful in credit lending—**risk decisions must align with portfolio risk tolerance**, not raw accuracy.

---

### **6. Production Credit Models Require Monitoring**

Static models are insufficient in real lending environments. Effective systems require:
- Feature and prediction drift monitoring
- Periodic retraining
- Portfolio-level risk tracking
---

### **7. Explainability Is Essential**

SHAP explanations were critical for validating model behavior and supporting underwriting decisions. In regulated financial settings, **interpretability matters more than marginal AUC improvements**.

---

### **Final Takeaway**

> Credit risk modeling is constrained by real-world uncertainty, not algorithms. Meaningful improvements come from better data, monitoring, and decision frameworks—not just more complex models.
