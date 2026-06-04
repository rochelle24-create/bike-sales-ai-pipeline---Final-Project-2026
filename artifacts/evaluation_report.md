# 🚲 Bike Sales — Model Evaluation Report

**Author:** Rachel Barazani — AI Developer  
**Course:** AI Developer Program — Hebrew University 2026

---


## Prediction: Purchase Quantity

### Logistic Regression
- Accuracy: 20.15%
- F1 Score: 18.77%

```
              precision    recall  f1-score   support

           1       0.20      0.18      0.19      3689
           2       0.20      0.08      0.11      3568
           3       0.21      0.40      0.27      3686
           4       0.20      0.12      0.15      3719
           5       0.20      0.22      0.21      3722

    accuracy                           0.20     18384
   macro avg       0.20      0.20      0.19     18384
weighted avg       0.20      0.20      0.19     18384

```

### Random Forest
- Accuracy: 19.67%
- F1 Score: 19.64%

```
              precision    recall  f1-score   support

           1       0.20      0.21      0.21      3689
           2       0.19      0.20      0.20      3568
           3       0.20      0.21      0.20      3686
           4       0.20      0.18      0.19      3719
           5       0.20      0.18      0.19      3722

    accuracy                           0.20     18384
   macro avg       0.20      0.20      0.20     18384
weighted avg       0.20      0.20      0.20     18384

```

### ✅ Winner: Random Forest (F1: 19.64%)

---


## Prediction: Bike Model Recommendation

### Logistic Regression
- Accuracy: 14.3%
- F1 Score: 13.62%

```
              precision    recall  f1-score   support

           0       0.17      0.05      0.08      2705
           1       0.15      0.15      0.15      2593
           2       0.14      0.19      0.16      2590
           3       0.14      0.19      0.16      2582
           4       0.15      0.07      0.10      2662
           5       0.14      0.17      0.15      2625
           6       0.14      0.19      0.16      2627

    accuracy                           0.14     18384
   macro avg       0.15      0.14      0.14     18384
weighted avg       0.15      0.14      0.14     18384

```

### Random Forest
- Accuracy: 18.1%
- F1 Score: 18.11%

```
              precision    recall  f1-score   support

           0       0.17      0.19      0.18      2705
           1       0.18      0.18      0.18      2593
           2       0.18      0.18      0.18      2590
           3       0.19      0.19      0.19      2582
           4       0.18      0.17      0.18      2662
           5       0.18      0.17      0.18      2625
           6       0.19      0.18      0.19      2627

    accuracy                           0.18     18384
   macro avg       0.18      0.18      0.18     18384
weighted avg       0.18      0.18      0.18     18384

```

### ✅ Winner: Random Forest (F1: 18.11%)

---


## Prediction: Cash Payment Prediction

### Logistic Regression
- Accuracy: 83.37%
- F1 Score: 75.81%

```
              precision    recall  f1-score   support

           0       0.83      1.00      0.91     15327
           1       0.00      0.00      0.00      3057

    accuracy                           0.83     18384
   macro avg       0.42      0.50      0.45     18384
weighted avg       0.70      0.83      0.76     18384

```

### Random Forest
- Accuracy: 83.34%
- F1 Score: 75.8%

```
              precision    recall  f1-score   support

           0       0.83      1.00      0.91     15327
           1       0.00      0.00      0.00      3057

    accuracy                           0.83     18384
   macro avg       0.42      0.50      0.45     18384
weighted avg       0.70      0.83      0.76     18384

```

### ✅ Winner: Logistic Regression (F1: 75.81%)

---


## 🏆 Overall Best Model

- Prediction: Cash Payment Prediction
- Model: Logistic Regression
- F1 Score: 75.81%
- Saved as: artifacts/models/model.pkl
