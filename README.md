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


# Fraud Detection using LSTM on the PaySim Dataset

Welcome to this guide on detecting fraudulent transactions using Long Short-Term Memory (LSTM) neural networks on the PaySim dataset. Strap in, we're about to dive deep!

## 📖 Table of Contents

- [Introduction](#introduction)
- [Setup & Prerequisites](#setup--prerequisites)
- [Data Preprocessing](#data-preprocessing)
- [Baseline Model](#baseline-model)
- [Addressing Class Imbalance](#addressing-class-imbalance)
- [Enhanced Model](#enhanced-model)
- [Results](#results)
- [Looking Ahead](#looking-ahead)

## 🚀 Introduction

Fraud detection is a hot topic in financial services, and machine learning, especially LSTMs, provides powerful tools to tackle this challenge. Here, we walk through the journey of building, refining, and improving a model's performance using the PaySim dataset.

## 🛠 Setup & Prerequisites

Before diving into the code, let's ensure we've got the right environment:

**Required Libraries**:
```python
pip install pandas keras imbalanced-learn
🔧 Data Preprocessing
1. Loading the Dataset:

    Using pandas, we can quickly load our dataset and start exploring.

2. Cleaning Our Data:

    Some columns may not be too helpful for our analysis. For instance, nameOrig, nameDest, and isFlaggedFraud can be removed.
    Using pd.get_dummies(), we transform our categorical variables.

3. Scaling Our Features:

    Neural networks like their inputs to be on a similar scale. We've utilized the MinMaxScaler to normalize our numerical columns, ensuring they're between 0 and 1.

4. Train-Test Split:

    To assess our model's performance, we set aside 20% of our data for testing.

🧠 Baseline Model

Starting simple, we outline a basic LSTM model to set a benchmark:
1. Defining LSTM Layer:

Our first layer has 100 units. LSTMs are great for sequence data, like time-series (or transactions in our case).
2. Avoiding Overfitting:

We introduce a Dropout layer, which randomly "drops out" a fraction of its inputs. This helps our model generalize better.
3. Making Predictions:

A Dense output layer with a sigmoid activation gives us a probability of whether a transaction is fraudulent.
⚖ Addressing Class Imbalance

A common challenge! Our data had way more genuine transactions than fraudulent ones, which can bias our model:
1. Leveling the Playing Field with SMOTE:

SMOTE stands for Synthetic Minority Over-sampling Technique. In simple terms, it creates "synthetic" samples of the minority class, making our classes balanced.
2. Reshaping Data for LSTM:

LSTMs are a bit picky! They need their input as a 3D array, so we reshape accordingly.
💡 Enhanced Model

Let's take our model to the next level:
1. Layering Up:

More layers and more units help our model capture complex patterns.
2. Thinking Both Ways with Bidirectional LSTM:

A bidirectional LSTM learns from the past and the future, giving us a broader perspective on our data.
3. More Dropout:

With more layers, comes the need for better regulation. Increased dropout layers help.
📊 Results

After all these improvements, our model became a sharp detective in catching fraudulent transactions! The precision and recall values showed that our model can now effectively pinpoint and recall most fraudulent transactions.
🌟 Looking Ahead

    Feature Engineering: There's always room for adding or optimizing features for better performance.
    Combining Models: Using an ensemble of LSTMs and other models might yield even better results.
    Trying Different Resampling Methods: Techniques like ADASYN or Borderline-SMOTE offer alternatives to SMOTE.
    Tuning it Right: With tools like Keras Tuner, we can experiment to find the perfect hyperparameters for our model.

Happy fraud hunting! 😉
