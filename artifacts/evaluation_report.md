# 🚲 Bike Sales — Model Evaluation Report

**Author:** Rachel Barazani — AI Developer  
**Course:** AI Developer Program — Hebrew University 2026

---


## Prediction: Purchase Quantity

### Logistic Regression
- Accuracy: 40.08%
- F1 Score: 28.97%

```
              precision    recall  f1-score   support

           1       0.44      0.89      0.58      7537
           2       0.24      0.11      0.15      4026
           3       0.19      0.06      0.10      3062
           4       0.18      0.00      0.01      2170
           5       0.00      0.00      0.00      1590

    accuracy                           0.40     18385
   macro avg       0.21      0.21      0.17     18385
weighted avg       0.28      0.40      0.29     18385

```

### Random Forest
- Accuracy: 38.88%
- F1 Score: 34.54%

```
              precision    recall  f1-score   support

           1       0.51      0.73      0.60      7537
           2       0.25      0.23      0.24      4026
           3       0.21      0.16      0.18      3062
           4       0.16      0.07      0.10      2170
           5       0.16      0.04      0.07      1590

    accuracy                           0.39     18385
   macro avg       0.26      0.25      0.24     18385
weighted avg       0.33      0.39      0.35     18385

```

### ✅ Winner: Random Forest (F1: 34.54%)

---


## Prediction: Bike Model Recommendation

### Logistic Regression
- Accuracy: 50.71%
- F1 Score: 47.06%

```
              precision    recall  f1-score   support

           0       0.55      0.57      0.56      1876
           1       0.50      0.76      0.60      2994
           2       0.76      0.82      0.79      2546
           3       0.23      0.01      0.01      1967
           4       0.45      0.58      0.51      3822
           5       0.40      0.37      0.38      2468
           6       0.41      0.27      0.33      2712

    accuracy                           0.51     18385
   macro avg       0.47      0.48      0.45     18385
weighted avg       0.47      0.51      0.47     18385

```

### Random Forest
- Accuracy: 59.92%
- F1 Score: 58.58%

```
              precision    recall  f1-score   support

           0       0.65      0.74      0.70      1876
           1       0.53      0.78      0.63      2994
           2       0.90      0.86      0.88      2546
           3       0.39      0.17      0.24      1967
           4       0.58      0.61      0.59      3822
           5       0.51      0.48      0.49      2468
           6       0.57      0.46      0.51      2712

    accuracy                           0.60     18385
   macro avg       0.59      0.59      0.58     18385
weighted avg       0.59      0.60      0.59     18385

```

### ✅ Winner: Random Forest (F1: 58.58%)

---


## Prediction: Cash Payment Prediction

### Logistic Regression
- Accuracy: 76.81%
- F1 Score: 66.74%

```
              precision    recall  f1-score   support

           0       0.77      1.00      0.87     14122
           1       0.00      0.00      0.00      4263

    accuracy                           0.77     18385
   macro avg       0.38      0.50      0.43     18385
weighted avg       0.59      0.77      0.67     18385

```

### Random Forest
- Accuracy: 75.88%
- F1 Score: 69.75%

```
              precision    recall  f1-score   support

           0       0.78      0.96      0.86     14122
           1       0.42      0.10      0.16      4263

    accuracy                           0.76     18385
   macro avg       0.60      0.53      0.51     18385
weighted avg       0.70      0.76      0.70     18385

```

### ✅ Winner: Random Forest (F1: 69.75%)

---


## 🏆 Overall Best Model

- Prediction: Cash Payment Prediction
- Model: Random Forest
- F1 Score: 69.75%
- Saved as: artifacts/models/model.pkl
