You can copy this directly into your **README.md** file on GitHub:

# Logistic Regression Using Diabetes Dataset

# Project Overview

This project implements **Logistic Regression from scratch using NumPy** and applies it to the **Pima Indians Diabetes Dataset** to predict whether a person is diabetic or not.

Instead of using Scikit-Learn's built-in Logistic Regression model, the algorithm is implemented manually using:

* Sigmoid Function
* Gradient Descent
* Weight and Bias Updates
* Binary Classification

# Objective

To build a Machine Learning model that predicts whether a patient has diabetes based on medical attributes.

# Technologies Used
* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-Learn

# Dataset Information

The dataset contains the following features:

| Feature                  | Description                |
| ------------------------ | -------------------------- |
| Pregnancies              | Number of pregnancies      |
| Glucose                  | Glucose level              |
| BloodPressure            | Blood pressure             |
| SkinThickness            | Skin thickness             |
| Insulin                  | Insulin level              |
| BMI                      | Body Mass Index            |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age                      | Age of the patient         |

### Target Variable

* 0 → Non-Diabetic
* 1 → Diabetic

## Logistic Regression Mathematics

### Linear Equation

[z = XW + b]
where:
 X = Input Features
 W = Weights
 b = Bias
 
### Sigmoid Function

[\sigma(z)=\frac{1}{1+e^{-z}}]

Used to convert the linear output into a probability between 0 and 1.

### Prediction
[\hat{Y} = \sigma(XW+b)]

### Cost Function (Binary Cross Entropy)

[
J(W,b)=-\frac{1}{m}\sum_{i=1}^{m}
[y_i\log(\hat{y_i})+(1-y_i)\log(1-\hat{y_i})]
]

---

### Gradient Equations

#### Weight Gradient

[dw=\frac{1}{m}X^T(\hat{Y}-Y)]

#### Bias Gradient

[db=\frac{1}{m}\sum(\hat{Y}-Y)]

### Gradient Descent Update

[W=W-\alpha dw]
[b=b-\alpha db]

where:
α = Learning Rate
m = Number of Training Samples

## Project Workflow
# Step 1: Import Libraries

* NumPy
* Pandas
* Scikit-Learn

# Step 2: Load Dataset
Read the diabetes dataset using Pandas.

# Step 3: Data Preprocessing
* Separate Features and Labels
* Standardize Features using StandardScaler

# Step 4: Split Dataset
Split the dataset into:
* Training Data (80%)
* Testing Data (20%)

# Step 5: Train Model
Train the custom Logistic Regression model using Gradient Descent.

# Step 6: Evaluate Model
Calculate accuracy on test data.

# Step 7: Make Predictions
Predict whether a patient is diabetic or not.

## Training the Model
```python
classifier = Logistic_Regression(
    learning_rate=0.01,
    no_of_iterations=1000
)

classifier.fit(X_train, Y_train)
```
#  Model Evaluation

```python
from sklearn.metrics import accuracy_score

predictions = classifier.predict(X_test)

accuracy = accuracy_score(Y_test, predictions)

print("Accuracy:", accuracy)
```
# Sample Prediction

```python
input_data = (5,166,72,19,175,25.8,0.587,51)

input_data_np = np.asarray(input_data)

input_data_reshaped = input_data_np.reshape(1,-1)

std_data = scaler.transform(input_data_reshaped)

prediction = classifier.predict(std_data)

print(prediction)
```
# Learning Outcomes
After completing this project, you will understand:
* Logistic Regression from scratch
* Sigmoid Function
* Gradient Descent
* Feature Scaling
* Binary Classification
* Model Evaluation
* Machine Learning Workflow


