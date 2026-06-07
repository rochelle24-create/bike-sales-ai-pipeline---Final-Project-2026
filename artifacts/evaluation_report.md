# 🚲 Bike Sales — Model Evaluation Report

**Author:** Rachel Barazani — AI Developer  
**Course:** AI Developer Program — Hebrew University 2026

---


## Prediction: Purchase Quantity

### Logistic Regression
- Accuracy: 35.68%
- F1 Score: 34.18%

```
              precision    recall  f1-score   support

           1       0.64      0.56      0.60      6992
           2       0.24      0.39      0.30      3675
           3       0.33      0.04      0.07      3506
           4       0.15      0.09      0.11      2461
           5       0.19      0.47      0.27      1756

    accuracy                           0.36     18390
   macro avg       0.31      0.31      0.27     18390
weighted avg       0.39      0.36      0.34     18390

```

### Random Forest
- Accuracy: 37.51%
- F1 Score: 39.02%

```
              precision    recall  f1-score   support

           1       0.71      0.58      0.64      6992
           2       0.24      0.27      0.26      3675
           3       0.26      0.22      0.24      3506
           4       0.20      0.25      0.22      2461
           5       0.17      0.25      0.20      1756

    accuracy                           0.38     18390
   macro avg       0.32      0.32      0.31     18390
weighted avg       0.41      0.38      0.39     18390

```

### ✅ Winner: Random Forest (F1: 39.02%)

---


## Prediction: Bike Model Recommendation

### Logistic Regression
- Accuracy: 52.46%
- F1 Score: 51.96%

```
              precision    recall  f1-score   support

           0       0.55      0.74      0.63      1821
           1       0.56      0.61      0.58      3028
           2       0.79      0.87      0.83      2559
           3       0.25      0.28      0.26      1953
           4       0.58      0.41      0.48      3817
           5       0.43      0.42      0.43      2495
           6       0.43      0.39      0.41      2717

    accuracy                           0.52     18390
   macro avg       0.51      0.53      0.52     18390
weighted avg       0.53      0.52      0.52     18390

```

### Random Forest
- Accuracy: 62.17%
- F1 Score: 61.51%

```
              precision    recall  f1-score   support

           0       0.62      0.84      0.71      1821
           1       0.56      0.77      0.65      3028
           2       0.92      0.86      0.89      2559
           3       0.37      0.28      0.32      1953
           4       0.67      0.53      0.59      3817
           5       0.55      0.55      0.55      2495
           6       0.60      0.53      0.56      2717

    accuracy                           0.62     18390
   macro avg       0.61      0.62      0.61     18390
weighted avg       0.62      0.62      0.62     18390

```

### ✅ Winner: Random Forest (F1: 61.51%)

---


## Prediction: Cash Payment Prediction

### Logistic Regression
- Accuracy: 58.7%
- F1 Score: 61.85%

```
              precision    recall  f1-score   support

           0       0.82      0.59      0.69     14091
           1       0.30      0.59      0.40      4299

    accuracy                           0.59     18390
   macro avg       0.56      0.59      0.54     18390
weighted avg       0.70      0.59      0.62     18390

```

### Random Forest
- Accuracy: 70.49%
- F1 Score: 71.39%

```
              precision    recall  f1-score   support

           0       0.83      0.78      0.80     14091
           1       0.39      0.47      0.43      4299

    accuracy                           0.70     18390
   macro avg       0.61      0.62      0.61     18390
weighted avg       0.73      0.70      0.71     18390

```

### ✅ Winner: Random Forest (F1: 71.39%)

---


## 🏆 Overall Best Model

- Prediction: Cash Payment Prediction
- Model: Random Forest
- F1 Score: 71.39%
- Saved as: artifacts/models/model.pkl
