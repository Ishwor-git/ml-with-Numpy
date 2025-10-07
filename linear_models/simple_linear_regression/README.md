# Linear Regression from Scratch using NumPy

## Overview
Linear Regression is one of the simplest and most widely used algorithms in machine learning. It describes the relationship between a dependent variable `y` and one or more independent variables `X` by fitting a linear equation to observedthe data.

The goal is to find the best-fitting line:
$
y = W X + b
$
where:
- $ W $ = weight (slope)
- $ b $ = bias (intercept)
- $ y $ = predicted output

---

## Mathematical Formulation

Given a dataset with `n` samples:
$
(X_1, y_1), (X_2, y_2), ..., (X_n, y_n)
$

Our model predicts:
$
\hat{y}_i = W X_i + b
$

The objective is to minimize the difference between predicted values ($ \hat{y} $) and actual values ($ y $).

---

## Cost Function (Mean Squared Error)

Cost function describes how off is the model from ideeal state.The most common cost function for linear regression is **Mean Squared Error (MSE)**:

$
J(W, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2
$

where:
- $ m $ = number of samples  
- $ \hat{y}_i $ = predicted output for sample i  
- $ y_i $ = actual target for sample i  

The factor \( \frac{1}{2m} \) is included for convenience when differentiating.

---

## Gradient Descent

We minimize the cost function using **Gradient Descent**, an iterative optimization algorithm that updates parameters in the direction of the negative gradient.

### Update Rules:
$
W := W - \alpha \frac{\partial J}{\partial W}
$  

$
b := b - \alpha \frac{\partial J}{\partial b}
$

where:
- $ \alpha $ = learning rate  
- $ \frac{\partial J}{\partial W} ,  \frac{\partial J}{\partial b} $ = gradients of the cost function

---

## Derivatives

From the MSE function, we derive the partial derivatives:

$
\frac{\partial J}{\partial W} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i) X_i
$  

$
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)
$

These are used in each iteration to adjust `W` and `b`.

---

## Algorithm Steps

1. Initialize parameters $ W = 0 $, $ b = 0 $
2. For a fixed number of iterations:
   - Compute predictions: $ \hat{y} = WX + b $
   - Compute gradients:
     - $ dw = \frac{1}{m} \sum (\hat{y} - y)X $

     - $ db = \frac{1}{m} \sum (\hat{y} - y) $
   - Update parameters:
     - $ W = W - \alpha dw $

     - $ b = b - \alpha db $
3. Repeat until convergence (i.e., minimal change in cost)

---

## Implementation Summary

- **Data Generation**: Synthetic data using NumPy with added noise
- **Model**: Linear model with parameters `W` and `b`
- **Training**: Batch gradient descent
- **Evaluation**: Mean Squared Error
- **Visualization**: Matplotlib to display data points and the fitted line

---

## Key Insights

- **Learning rate (α)** controls convergence speed.  
  Too high → overshoot the minimum  
  Too low → very slow learning
- **Number of iterations** should be enough for convergence but not excessive.
- **MSE** gives an intuitive sense of how far predictions are from actual values.

---

## 🚀 Extensions

Once you understand the basic single-variable regression, you can extend it to:
1. **Multiple Linear Regression** (multi-feature input)
2. **Stochastic Gradient Descent (SGD)**
3. **Polynomial Regression**
4. **Regularized Regression** (Ridge, Lasso)

---

## Summary Formula Sheet

| Concept | Formula |
|----------|----------|
| Hypothesis | $ \hat{y} = WX + b $ |
| Cost Function | $ J(W, b) = \frac{1}{2m} \sum (\hat{y}_i - y_i)^2 $ |
| Gradient w.r.t W | $ \frac{1}{m} \sum (\hat{y}_i - y_i) X_i $ |
| Gradient w.r.t b | $ \frac{1}{m} \sum (\hat{y}_i - y_i) $ |
| Update Rules | $ W = W - \alpha dw, \ b = b - \alpha db $ |

---

## References
- Andrew Ng — Machine Learning Course (Stanford)
- “Pattern Recognition and Machine Learning” — Christopher M. Bishop
- Scikit-learn Documentation on Linear Regression

---

## Summary
Linear Regression builds the foundation for understanding optimization, cost functions, and gradient-based learning. Mastering it prepares you to tackle advanced topics like logistic regression, neural networks, and deep learning.
