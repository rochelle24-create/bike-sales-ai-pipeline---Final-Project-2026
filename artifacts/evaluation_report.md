# 🚲 Bike Sales — Model Evaluation Report

**Author:** Rachel Barazani — AI Developer  
**Course:** AI Developer Program — Hebrew University 2026

---


## Prediction: Purchase Quantity

### Logistic Regression
- Accuracy: 39.86%
- F1 Score: 28.21%

```
              precision    recall  f1-score   support

           1       0.43      0.90      0.58      7473
           2       0.23      0.07      0.11      4039
           3       0.20      0.09      0.12      3065
           4       0.21      0.00      0.01      2234
           5       0.00      0.00      0.00      1579

    accuracy                           0.40     18390
   macro avg       0.21      0.21      0.16     18390
weighted avg       0.28      0.40      0.28     18390

```

### Random Forest
- Accuracy: 38.5%
- F1 Score: 34.37%

```
              precision    recall  f1-score   support

           1       0.50      0.72      0.59      7473
           2       0.24      0.24      0.24      4039
           3       0.21      0.15      0.18      3065
           4       0.20      0.08      0.12      2234
           5       0.15      0.05      0.07      1579

    accuracy                           0.39     18390
   macro avg       0.26      0.25      0.24     18390
weighted avg       0.33      0.39      0.34     18390

```

### ✅ Winner: Random Forest (F1: 34.37%)

---


## Prediction: Bike Model Recommendation

### Logistic Regression
- Accuracy: 51.82%
- F1 Score: 48.22%

```
              precision    recall  f1-score   support

           0       0.58      0.60      0.59      1834
           1       0.50      0.77      0.60      3043
           2       0.77      0.84      0.80      2520
           3       0.24      0.00      0.01      1966
           4       0.45      0.57      0.50      3848
           5       0.44      0.40      0.42      2495
           6       0.41      0.30      0.34      2684

    accuracy                           0.52     18390
   macro avg       0.48      0.50      0.47     18390
weighted avg       0.49      0.52      0.48     18390

```

### Random Forest
- Accuracy: 61.81%
- F1 Score: 60.51%

```
              precision    recall  f1-score   support

           0       0.67      0.77      0.72      1834
           1       0.55      0.79      0.65      3043
           2       0.91      0.88      0.89      2520
           3       0.40      0.19      0.26      1966
           4       0.60      0.61      0.61      3848
           5       0.55      0.51      0.53      2495
           6       0.58      0.49      0.53      2684

    accuracy                           0.62     18390
   macro avg       0.61      0.61      0.60     18390
weighted avg       0.61      0.62      0.61     18390

```

### ✅ Winner: Random Forest (F1: 60.51%)

---


## Prediction: Cash Payment Prediction

### Logistic Regression
- Accuracy: 76.61%
- F1 Score: 66.47%

```
              precision    recall  f1-score   support

           0       0.77      1.00      0.87     14089
           1       0.00      0.00      0.00      4301

    accuracy                           0.77     18390
   macro avg       0.38      0.50      0.43     18390
weighted avg       0.59      0.77      0.66     18390

```

### Random Forest
- Accuracy: 75.88%
- F1 Score: 69.74%

```
              precision    recall  f1-score   support

           0       0.78      0.96      0.86     14089
           1       0.43      0.10      0.17      4301

    accuracy                           0.76     18390
   macro avg       0.61      0.53      0.51     18390
weighted avg       0.70      0.76      0.70     18390

```

### ✅ Winner: Random Forest (F1: 69.74%)

---


## 🏆 Overall Best Model

- Prediction: Cash Payment Prediction
- Model: Random Forest
- F1 Score: 69.74%
- Saved as: artifacts/models/model.pkl
