# 🚲 Bike Sales — Model Evaluation Report

**Author:** Rachel Barazani — AI Developer  
**Course:** AI Developer Program — Hebrew University 2026

---


## Prediction: Purchase Quantity

### Logistic Regression
- Accuracy: 39.97%
- F1 Score: 28.03%

```
              precision    recall  f1-score   support

           1       0.43      0.91      0.58      7473
           2       0.21      0.08      0.12      4039
           3       0.22      0.05      0.08      3065
           4       0.19      0.02      0.04      2234
           5       0.00      0.00      0.00      1579

    accuracy                           0.40     18390
   macro avg       0.21      0.21      0.16     18390
weighted avg       0.28      0.40      0.28     18390

```

### Random Forest
- Accuracy: 40.12%
- F1 Score: 30.61%

```
              precision    recall  f1-score   support

           1       0.45      0.85      0.59      7473
           2       0.24      0.20      0.22      4039
           3       0.19      0.03      0.06      3065
           4       0.27      0.02      0.04      2234
           5       0.16      0.01      0.03      1579

    accuracy                           0.40     18390
   macro avg       0.26      0.23      0.19     18390
weighted avg       0.31      0.40      0.31     18390

```

### ✅ Winner: Random Forest (F1: 30.61%)

---


## Prediction: Bike Model Recommendation

### Logistic Regression
- Accuracy: 52.0%
- F1 Score: 48.47%

```
              precision    recall  f1-score   support

           0       0.58      0.61      0.60      1834
           1       0.51      0.76      0.61      3043
           2       0.78      0.84      0.80      2520
           3       0.25      0.01      0.02      1966
           4       0.45      0.57      0.50      3848
           5       0.44      0.42      0.43      2495
           6       0.41      0.28      0.34      2684

    accuracy                           0.52     18390
   macro avg       0.49      0.50      0.47     18390
weighted avg       0.49      0.52      0.48     18390

```

### Random Forest
- Accuracy: 62.9%
- F1 Score: 60.66%

```
              precision    recall  f1-score   support

           0       0.67      0.80      0.73      1834
           1       0.54      0.85      0.66      3043
           2       0.93      0.87      0.90      2520
           3       0.56      0.10      0.17      1966
           4       0.60      0.65      0.62      3848
           5       0.56      0.50      0.53      2495
           6       0.60      0.51      0.55      2684

    accuracy                           0.63     18390
   macro avg       0.64      0.61      0.60     18390
weighted avg       0.63      0.63      0.61     18390

```

### ✅ Winner: Random Forest (F1: 60.66%)

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
- Accuracy: 76.45%
- F1 Score: 67.74%

```
              precision    recall  f1-score   support

           0       0.77      0.99      0.87     14089
           1       0.45      0.03      0.06      4301

    accuracy                           0.76     18390
   macro avg       0.61      0.51      0.46     18390
weighted avg       0.70      0.76      0.68     18390

```

### ✅ Winner: Random Forest (F1: 67.74%)

---


## 🏆 Overall Best Model

- Prediction: Cash Payment Prediction
- Model: Random Forest
- F1 Score: 67.74%
- Saved as: artifacts/models/model.pkl
