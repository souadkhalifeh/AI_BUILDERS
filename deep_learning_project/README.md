# 🎯 Solo Portfolio Project — Deep Learning on Tabular Data

> **AI Builders Bootcamp · Deep Learning Track**
>
> Your first independent Deep Learning project.
>
> This project is your opportunity to take a real-world dataset, build a neural network, train it, evaluate it, improve it, and save a working AI model that you can showcase in your portfolio.

## Prerequisites

Before starting this project, you should have completed:

- Deep Learning Module 1 & 2
- Guided Deep Learning Build
- PyTorch Fundamentals

---

# 🚀 The Mission

During the guided project, you followed the complete Deep Learning workflow:

```
Dataset
   ↓
Data Cleaning
   ↓
Preprocessing
   ↓
Neural Network
   ↓
Training
   ↓
Evaluation
   ↓
Saving the Model
```

Now it is your turn to repeat this workflow independently using a new dataset.

The goal is not just to train a model.

The goal is to demonstrate that you can:

✅ Understand a real-world problem  
✅ Prepare data correctly  
✅ Build a neural network  
✅ Train and evaluate a model  
✅ Improve your architecture  
✅ Save a model ready for future deployment  

---

# Step 1 — Choose Your Dataset & Task

Choose **one dataset** from the options below.
Pick a dataset based on the Deep Learning task you want to practice.
---

# 🟢 Option A — Binary Classification

Recommended for your first independent Deep Learning project.
The goal is to predict one of two possible outcomes.

## Dataset Options

| Dataset | Predict | Source |
|---|---|---|
| ⭐ **Heart Disease** | Does the patient have heart disease? | UCI / Kaggle |
| **Pima Diabetes** | Is the patient diabetic? | Kaggle |
---

# 🟡 Option B — Multi-class Classification

The goal is to predict one class from multiple possible classes.

## Dataset Options

| Dataset | Predict | Source |
|---|---|---|
| ⭐ **Body Performance** | Fitness level A/B/C/D | Kaggle |
| **Dry Bean Dataset** | Bean category (7 classes) | UCI |

---

# 🔵 Option C — Regression

The goal is to predict a continuous numerical value.

## Dataset Options

| Dataset | Predict | Source |
|---|---|---|
| ⭐ **California Housing** | House price | Scikit-learn / Kaggle |
| **Bike Sharing Demand** | Number of rentals | UCI |
| **Student Performance** | Final grade | UCI |
---

# 🟣 Bring Your Own Dataset

You can choose another dataset if it follows these rules:

## Requirements

✅ Tabular data only  
(rows = samples, columns = features)

✅ Clear target column

✅ At least:

- 500 rows
- 5 features

❌ No images

❌ No raw text

> Computer Vision and NLP projects will be covered in later modules.

If you are unsure about your dataset, ask before starting.

---

# Step 2 — Choose the Correct Model Setup

Your output layer, loss function, and target format depend on your task.
---

## ⚠️ Important Reminder

Your model should output **raw logits**.
Do NOT manually add:
- Sigmoid
- Softmax
The loss function handles this internally.

---

# Step 3 — Project Requirements

# 1. Problem Understanding
Explain:
- What are you predicting?
- Why is this problem important?
- What type of Deep Learning task is it?

---

# 2. Data Exploration
Include:
- Dataset shape
- Dataset information
- Missing values check
- Target distribution
- At least two meaningful visualizations

---

# 3. Data Preparation
You must:
- Handle missing values
- Encode categorical features if needed
- Split the dataset into:

```
Train / Validation / Test
```

Example:

```
70% / 10% / 20%
```

---

# 4. Data Scaling


## Important:
Fit the scaler **only on training data**.

This prevents:
- Data leakage
- Incorrect evaluation
---

# 5. Dataset & DataLoader
Implement:
- Custom PyTorch Dataset
- DataLoader

Make sure your target data type matches your task.

---

# 6. Neural Network Model

Build your model using:
```python
nn.Module
```
The architecture must match your selected task.

---

# 7. Training

Implement the complete training workflow:

1. Forward pass
2. Calculate loss
3. Backpropagation
4. Update weights
5. Reset gradients

Your training should include:

- Validation loop
- `model.eval()`
- `torch.no_grad()`

---

# 8. Experiment & Model Improvement

Compare:
## Baseline Model
vs
## Improved Model

Possible improvements:
- More layers
- More neurons
- Dropout
- Different learning rate
- Different optimizer settings

Show your results in a comparison table.

---

# 9. Evaluation

Evaluate your final model on the **TEST dataset only**.
## Classification Tasks
Report:
- Accuracy
- F1-score
- Confusion Matrix

## Regression Tasks
Report:
- RMSE
- MAE
- R² Score

---

# 10. Save & Reload Your Model
Save:
- Model weights
- Data preprocessing objects (Scaler)

Then:
1. Reload your model
2. Reload your scaler
3. Confirm that predictions remain consistent

---

# 📦 Submission Requirements

Submit:
## 1. Jupyter Notebook
## 2. Dataset
Either:
- Upload the dataset
- Or provide the dataset link

## 3. Short Reflection
Answer:
- Why did you choose this dataset?
- What challenges did you face?
- What improvements did you try?
- What did you learn?

---

# 🎯 Final Goal

At the end of this project, you should be able to independently build a complete Deep Learning pipeline:

```
Data
 ↓
Preprocessing
 ↓
Neural Network
 ↓
Training
 ↓
Evaluation
 ↓
Saved AI Solution
```

This is the same workflow used by AI engineers when building real-world Deep Learning applications. 🚀
