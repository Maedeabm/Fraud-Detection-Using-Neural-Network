# Credit Card Fraud Detection using Neural Networks
Overview

In this project, we leverage the power of neural networks to detect potentially fraudulent credit card transactions. The dataset in use has been sourced from Kaggle's Credit Card Fraud Detection challenge.
📊 Dataset

    The dataset comprises transactions made via credit cards.
    From a total of 284,807 transactions, only 492 were flagged as fraud.
    This imbalance, with the fraudulent class accounting for a mere 0.172% of the dataset, posed a challenge.

🛠 Approach

    Data Preprocessing:
        Addressed the data imbalance by applying the Synthetic Minority Over-sampling Technique (SMOTE) to oversample the minority class.

    Neural Network Model:
        Developed using the powerful PyTorch library.

    Evaluation Metrics:
        Beyond mere accuracy, which can be misleading with imbalanced datasets, we also monitored Precision, Recall, F1 Score, and ROC AUC Score to holistically evaluate the model's performance.

📈 Results

    Accuracy: 98.16%
    Precision: 7.75%
    Recall: 88.78%
    F1 Score: 14.26%
    ROC AUC Score: 93.38%

🔍 Analysis

    Accuracy: Though high, accuracy alone doesn't tell the whole story especially with imbalanced datasets.
    Recall: The standout metric with a high score of 88.78%, showing the model's strength in identifying most of the fraudulent transactions.
    Precision: At 7.75%, it's an area where the model can improve, indicating a higher number of false positives.

🚀 Further Steps and Improvements

    Engage in hyperparameter tuning for the model.
    Experiment with feature engineering to enhance input data.
    Evaluate the potential of anomaly detection models for this use case.
    Consider ensemble methods to potentially boost performance.

📝 Conclusion

Fraud detection, given its complexity and the stakes involved, is a challenging domain. The results achieved here are promising and point towards a strong baseline model. Adjustments and refinements based on specific real-world use cases and costs associated with false positives/negatives would further enhance its utility.
