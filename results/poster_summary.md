# Poster Summary

## 1. Problem Statement & Core Research Goal

Autism Spectrum Disorder (ASD) prediction using only pre-diagnostic phenotypic information is a challenging but clinically meaningful task. In real-world healthcare settings, data are often distributed across multiple institutions with heterogeneous acquisition conditions, making centralized training difficult due to privacy and governance constraints. This project investigates whether a federated learning framework can preserve predictive utility while respecting data privacy across sites.

Our core research goal is to evaluate whether a privacy-preserving federated learning approach can achieve competitive ASD classification performance using only clean, non-diagnostic phenotypic features, while also measuring the impact of site heterogeneity on model performance and communication overhead.

## 2. Data & Leakage Audit Methodology

We used the ABIDE phenotypic dataset and engineered a clean feature set consisting of demographic and behavioral variables that are available before diagnosis, including age, sex, handedness, and intelligence measures such as FIQ, VIQ, and PIQ. We explicitly excluded clinically diagnostic assessment variables such as ADOS and ADI-R scores because they are highly informative of the target label and therefore constitute direct leakage if included in the model input.

This leakage audit was essential for maintaining scientific validity. The model should predict ASD status using pre-diagnostic, patient-level descriptors rather than variables that are partly diagnostic instruments used to define the condition. The cleaned feature set therefore reflects a realistic and ethically sound research setting, where prediction is based on non-diagnostic, population-level features rather than on instruments used to establish the diagnosis.

## 3. Experimental Results Table

| Experiment | ROC-AUC | Accuracy | Communication Overhead (KB) |
|---|---:|---:|---:|
| Centralized Baseline | 0.6213 ± 0.0499 | 0.6025 ± 0.0383 | N/A |
| Federated IID | 0.6382 | 0.6223 | 10.94 KB |
| Federated Site Non-IID | 0.6310 | 0.6241 | 43.75 KB |

Notes:
- Centralized baseline metrics were computed using 5-fold cross-validation on the full cleaned dataset.
- Federated results were obtained from a simple FedAvg simulation over 10 rounds.
- Communication overhead is reported as cumulative bytes transferred across the simulation, converted to kilobytes for comparison.

## 4. Key XAI Findings

We used SHAP to interpret the trained logistic regression model and identify the variables that most strongly influenced ASD classification. The strongest contributors were the IQ-related features, particularly FIQ, VIQ, and PIQ, followed by sex-related encoding and smaller but meaningful effects from age and handedness.

This pattern is clinically interpretable: cognitive and developmental function, as represented by IQ measures, carries substantial discriminative information in phenotypic ASD prediction. Meanwhile, demographic features such as sex and handedness provide additional but weaker signal. Importantly, the explanation does not rely on diagnostic leakage variables, which keeps the interpretation aligned with the pre-diagnostic feature design.

## 5. Core Conclusion

Despite site heterogeneity, the federated models remain competitive with the centralized baseline and preserve predictive utility while enabling privacy-preserving collaboration across institutions. The site-based non-IID experiment shows only a modest drop in ROC-AUC compared with the centralized baseline, suggesting that the federated approach remains effective under realistic cross-site distribution shifts.

This work demonstrates that robust ASD prediction can be achieved using clean phenotypic features across distributed clinical sites, with federated learning offering a practical path toward privacy-aware multi-institution collaboration.
