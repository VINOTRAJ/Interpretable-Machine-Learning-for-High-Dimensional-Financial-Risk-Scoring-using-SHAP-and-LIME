# Interpretable Machine Learning for High-Dimensional Financial Risk Scoring (SHAP + LIME)

# 📌 Project Overview

This project builds a highly interpretable machine-learning pipeline for corporate default risk prediction using a synthetic high-dimensional financial dataset.
The focus is not only model accuracy but also deep explainability, which is crucial for regulated financial environments (e.g., credit committees, banking oversight, Basel compliance).

# The dataset includes core financial ratios such as: 
[Financial_Datset.csv](https://github.com/user-attachments/files/23607898/Financial_Datset.csv)



    Liquidity ratios: current_ratio, quick_ratio, cash_ratio
    Leverage metrics: debt_equity_ratio, interest_coverage_ratio
    Profitability ratios: gross_margin, operating_margin, net_profit_margin, return_on_assets, return_on_equity
    Efficiency ratios: asset_turnover, inventory_turnover, receivables_turnover
    Cash flow ratios: operating_cash_flow_ratio, free_cash_flow_ratio
    Market indicators: price_to_earnings, market_cap
    Target variable: default_risk

This makes the dataset perfectly suitable for a risk-scoring explainability project.

# 🎯 Project Objectives

    Preprocess and transform high-dimensional financial indicators (scaling, encoding, outlier handling).
    Engineer new features, including interaction terms and polynomial ratios.
    Train a non-linear ML classifier (XGBoost / Random Forest) optimized for AUC and F1-score.
    Generate global explanations using:
        SHAP summary plots
        SHAP feature importance (mean absolute SHAP values)
    Generate local explanations for three high-risk companies using:
        SHAP force plots
        LIME local surrogate models
    Translate ML insights into business intelligence useful for credit officers and risk committees.

🧹 1. Data Preprocessing
[Financial_Dataset_preprocessed.csv](https://github.com/user-attachments/files/23607896/Financial_Dataset_preprocessed.csv)
    Handled missing values and extreme outliers (Winsorization/IQR).
    Scaled numerical indicators using StandardScaler.
    Created engineered features (e.g., leverage × profitability interactions).
    Selected features using correlation filtering + model-based selection.
    Prepared training/validation splits with stratified sampling.

🤖 2. Model Training

The final model is a tuned XGBoost Classifier, selected for:

    Non-linear learning capability
    High performance on tabular financial data
    Compatibility with SHAP and LIME

Hyperparameters were optimized using GridSearchCV with a balanced scoring metric (AUC).

Key Metrics:

   auc: 0.9104358256697321
   f1: 0.8178137651821862
   precision: 0.808
   recall: 0.8278688524590164

📊 3. Global Model Explainability (SHAP)

Using SHAP TreeExplainer, the project generates:
✔ Global SHAP Summary Plot 

Shows how each feature contributes to model output across the entire dataset.
✔ Mean Absolute SHAP Values

Used for global feature ranking to identify top risk drivers.
Top global predictors typically include:

    Debt-to-equity ratio
    Interest coverage ratio
    Net profit margin
    Operating cash flow ratio
    Return on assets

These provide a risk committee with actionable insights into what drives default likelihood across all firms.

# 🔍 4. Local Interpretations (SHAP + LIME)

Three high-risk companies (defaults or borderline cases) are analyzed using:
✔ SHAP Force Plots
Explain exactly why the model classified a firm as high risk or low risk.
 <img width="1200" height="945" alt="shap_summary_beeswarm" src="https://github.com/user-attachments/assets/3ff11743-84a5-4228-bdd9-4f879dc7a163" />
<img width="1200" height="945" alt="shap_summary_bar" src="https://github.com/user-attachments/assets/02f465ac-3cb3-4faa-abf9-751b59822cf9" />



✔ LIME Local Explanations
Provide model-agnostic validation of SHAP insights for critical cases.
[lime_summary.json](https://github.com/user-attachments/files/23607853/lime_summary.json)
[lime_case_2.html](https://github.com/user-attachments/files/23607845/lime_case_2.html)
[lime_case_1.html](https://github.com/user-attachments/files/23607838/lime_case_1.html)



Comparing SHAP and LIME helps confirm model reliability and reveal edge-case behavior.
[shap_lime_comparison.json](https://github.com/user-attachments/files/23607876/shap_lime_comparison.json)


📝 5. Business Insights for a Credit Committee
Key actionable recommendations:

    Firms with high leverage + low profitability remain the strongest indicators of default risk.
    Weak operational cash flow drastically increases default probability even if profitability appears stable.
    A low interest coverage ratio should trigger mandatory manual underwriting review.
    Market-based indicators (P/E, market cap) influence risk but carry lower SHAP impact compared to fundamentals.
    Explainability confirms the model is transparent and regulator-friendly, suitable for credit risk audits and model validation teams.




# 🚀 Final Outcome

This project demonstrates a complete end-to-end credit risk scoring pipeline with industry-grade explainability.
Both SHAP and LIME offer transparency required for regulatory approval, internal audit, and executive-level decision-making.
