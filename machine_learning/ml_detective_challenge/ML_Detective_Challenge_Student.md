# 🕵️ The Machine Learning Detective Challenge
### AI Builders Bootcamp — Capstone Assessment (Supervised + Unsupervised Learning)

---

## 1. Challenge Introduction

By now you have completed both the Supervised Learning and Unsupervised Learning modules. You know the algorithms. You can call .fit() and .predict(). But a real Machine Learning Engineer is not someone who memorizes algorithm names — it is someone who can walk into a messy, ambiguous business problem, understand the data on the table, and reason their way to the right approach.

This challenge does not hand you a clean dataset and ask you to run a model. Instead, it hands you real business scenarios from healthcare, finance, e-commerce, cybersecurity, and more — each with a description of the dataset you would actually be given. Your job is to act as the detective: read the clues, inspect the data, figure out what kind of problem you are facing, choose the right tools, and — most importantly — justify your reasoning.

There is often no single "perfect" answer. What matters is whether your reasoning is sound, whether you can defend your algorithm choice, and whether you think about the whole pipeline — data understanding, preprocessing, evaluation, and trade-offs — not just the model.

> **The mindset we are testing:** *"Think like a Machine Learning Engineer, not like a flashcard."*

### Learning Objectives
By completing this challenge you will be able to:

- Distinguish between Supervised and Unsupervised Learning from the data and the business goal.
- Identify the correct Machine Learning task (Regression, Classification, Clustering, Association, Recommendation, Dimensionality Reduction).
- Inspect a dataset and reason about labels, feature types, and data quality.
- Select appropriate algorithm(s) and justify the choice.
- Anticipate data challenges (imbalance, outliers, missing values, high dimensionality, text) and propose fixes.
- Communicate engineering decisions clearly and weigh trade-offs.

---

## 2. Student Instructions

For every scenario, work through these **ten steps** in your answer template:

| # | Step | What to answer |
|---|------|----------------|
| 1 | **Business Understanding** | What is the company trying to achieve? |
| 2 | **Dataset Understanding** | What data is available? Is there a target variable? What are the feature types? |
| 3 | **Learning Type** | Supervised or Unsupervised? |
| 4 | **ML Task** | Regression / Classification / Clustering / Association / Recommendation / Dimensionality Reduction? |
| 5 | **Algorithm Selection** | Which algorithm(s) would you choose? |
| 6 | **Reasoning** | Why is this algorithm appropriate for this problem and this data? |
| 7 | **Expected Output** | What does the model actually produce? |
| 8 | **Data Preparation** | What preprocessing steps are required? |
| 9 | **Challenges & Solutions** | What problems could occur in the data, and how would you solve them? |
| 10 | **Alternative Approach (optional)** | Is there another valid solution? |

### Rules of the Game

- Reasoning beats labels. A correct answer with no justification earns very few points.
- Start from the data. Always state what data you have and whether a target variable exists before choosing a task.
- Watch out for TRAP scenarios — some are designed to look like one task but actually are another. Read the dataset and the goal, not just the column names.
- If multiple algorithms are valid, say so and explain the trade-off.
- Use only the concepts taught in the bootcamp (see the Toolbox appendix).
- For the Bonus Section, you must compare algorithms and recommend one for production.

### Difficulty Legend

- 🟢 **Easy** — The learning type should be obvious.
- 🟡 **Medium** — Think before you decide.
- 🔴 **Advanced** — Multiple algorithms may be valid; trade-offs matter.
- ⚠️ **Trap** — Looks like one thing, may actually be another. Be careful.

---

## 3. Challenge Scenarios

There are **13** scenarios. Answer all ten steps for each one in your template.

### 🟢 Scenario 1 — Healthcare — Predicting Patient No-Shows

**Scenario.** A large hospital network wants to reduce wasted appointment slots. For every booked appointment they have rich historical data, including whether the patient ultimately showed up or did not show up.

**Business Goal.** Predict whether a patient will miss their upcoming appointment.

**Dataset Information**

- **Rows:** 110,000 appointment records
- **Target Variable:** Showed Up (Yes / No)

| Feature | Type |
|---------|------|
| Age | Numerical |
| Gender | Categorical |
| Day of Week | Categorical |
| Lead Time (days) | Numerical |
| SMS Reminder Sent | Categorical (Yes/No) |
| Distance to Clinic | Numerical |
| Previous No-Shows | Numerical |
| Showed Up | Categorical (Yes/No) — label |

**Additional Notes:**
- Mild class imbalance (more shows than no-shows)
- Some missing distance values
- Mixed numerical and categorical features

*Answer all 10 steps (Business Understanding → Alternative Approach) for this scenario.*

---

### 🟢 Scenario 2 — Real Estate — Estimating House Sale Price

**Scenario.** A property listing platform wants to suggest a fair listing price for newly added homes. They have a large archive of completed sales, each with the final price.

**Business Goal.** Estimate the expected sale price of a newly listed house.

**Dataset Information**

- **Rows:** 80,000 past home sales
- **Target Variable:** Sale Price (continuous $ amount)

| Feature | Type |
|---------|------|
| Square Footage | Numerical |
| Bedrooms | Numerical |
| Bathrooms | Numerical |
| Neighborhood | Categorical (high cardinality) |
| Year Built | Numerical |
| Lot Size | Numerical |
| Garage Capacity | Numerical |
| Sale Price | Numerical — label |

**Additional Notes:**
- Outliers from luxury homes
- Price distribution is right-skewed
- Neighborhood is high-cardinality categorical

*Answer all 10 steps (Business Understanding → Alternative Approach) for this scenario.*

---

### 🟢 Scenario 3 — Retail — Customer Segmentation for Marketing

**Scenario.** A supermarket chain with a large loyalty program wants to discover natural groups of similar customers so marketing can design targeted campaigns. There are no predefined groups and no labels.

**Business Goal.** Group customers into meaningful segments for targeted marketing.

**Dataset Information**

- **Rows:** 500,000 loyalty members
- **Target Variable:** None (no labels)

| Feature | Type |
|---------|------|
| Age | Numerical |
| Annual Income | Numerical |
| Total Spending | Numerical |
| Visit Frequency | Numerical |
| Average Basket Size | Numerical |

**Additional Notes:**
- Features on very different scales
- Number of groups is unknown
- No target variable

*Answer all 10 steps (Business Understanding → Alternative Approach) for this scenario.*

---

### 🟢 Scenario 4 — Finance — Loan Default Prediction (Imbalanced)

**Scenario.** A bank wants to flag risky applicants before approving loans. They have ten years of loan history, each labelled as paid off or defaulted. Defaults are the minority.

**Business Goal.** Predict whether a new applicant will default on their loan.

**Dataset Information**

- **Rows:** 120,000 historical loans
- **Target Variable:** Loan Status (Paid off / Defaulted)

| Feature | Type |
|---------|------|
| Applicant Income | Numerical |
| Employment Length | Numerical |
| Credit Score | Numerical |
| Loan Amount | Numerical |
| Debt-to-Income Ratio | Numerical |
| Home Ownership | Categorical |
| Loan Status | Categorical (Paid / Default) — label |

**Additional Notes:**
- Imbalanced dataset (defaults ~ 8% of records)
- Regulatory need for explainability
- Correlated financial features

*Answer all 10 steps (Business Understanding → Alternative Approach) for this scenario.*

---

### 🟢 Scenario 5 — E-commerce — 'Frequently Bought Together'

**Scenario.** An online store wants to power its classic 'Customers who bought this also bought…' feature. They have years of shopping carts, each being the set of products purchased in one transaction. There is no target variable.

**Business Goal.** Discover product co-purchase patterns to drive cross-selling.

**Dataset Information**

- **Rows:** 3,000,000 transactions (baskets)
- **Target Variable:** None (no labels)

| Feature | Type |
|---------|------|
| Transaction ID | Identifier |
| List of Products in Basket | Categorical (set of items) |

**Additional Notes:**
- Very sparse, high number of unique items
- No target variable
- Need to pick support / confidence thresholds

*Answer all 10 steps (Business Understanding → Alternative Approach) for this scenario.*

---

### 🟡 ⚠️ Scenario 6 — Telecommunications — Customer Churn

**Scenario.** A telecom company wants to know which customers are about to leave so it can act to retain them. The dataset is the familiar telecom table, and it contains BOTH a churn flag and a monthly-charges amount.

**Business Goal.** Identify customers likely to churn so retention offers can be targeted.

**Dataset Information**

- **Rows:** 7,000 customers
- **Target Variable:** Churn (Yes / No)  — note: Monthly Charges is also present

| Feature | Type |
|---------|------|
| Tenure (months) | Numerical |
| Contract Type | Categorical |
| Internet Service | Categorical |
| Tech Support | Categorical |
| Payment Method | Categorical |
| Monthly Charges | Numerical |
| Total Charges | Numerical (stored as text, has blanks) |
| Churn | Categorical (Yes/No) — label |

**Additional Notes:**
- TRAP: the same data could predict Monthly Charges (regression) — read the goal!
- Imbalanced (churners are the minority)
- Total Charges stored as text with blank values
- Many categorical features

*Answer all 10 steps (Business Understanding → Alternative Approach) for this scenario.*

---

### 🟡 ⚠️ Scenario 7 — Education — Exact Score vs. Pass/Fail

**Scenario.** A school has student records and the principal makes TWO different requests with the same dataset:  (A) 'Predict each student's exact final exam score (0–100).'  (B) 'Predict whether each student will pass or fail (pass = score ≥ 60).'  Answer the framework separately for (A) and (B) and explain what changes.

**Business Goal.** Help the principal predict student outcomes — as a number AND as a pass/fail decision.

**Dataset Information**

- **Rows:** 1,000 students
- **Target Variable:** (A) Final Exam Score (continuous)   /   (B) Pass or Fail (derived, score ≥ 60)

| Feature | Type |
|---------|------|
| Gender | Categorical |
| Parental Education | Categorical |
| Lunch Type | Categorical |
| Test-Prep Course | Categorical |
| Math Score | Numerical |
| Reading Score | Numerical |
| Writing Score | Numerical |
| Final Exam Score | Numerical — label |

**Additional Notes:**
- TRAP: same data, two different target framings
- Continuous target → regression; thresholded target → classification
- Small dataset
- Several categorical features need encoding

*Answer all 10 steps (Business Understanding → Alternative Approach) for this scenario.*

---

### 🟡 Scenario 8 — Cybersecurity — Detecting Unusual Network Activity

**Scenario.** A security operations team wants to flag network traffic that looks abnormal compared to normal behavior. Attacks are rare, constantly evolving, and mostly unlabeled — there is no reliable 'attack / not attack' column for most traffic.

**Business Goal.** Detect anomalous / suspicious network activity without relying on attack labels.

**Dataset Information**

- **Rows:** 2,000,000 connection records
- **Target Variable:** None / mostly unlabeled

| Feature | Type |
|---------|------|
| Packet Size | Numerical |
| Connection Duration | Numerical |
| Protocol | Categorical |
| Bytes Sent | Numerical |
| Bytes Received | Numerical |
| Failed Login Attempts | Numerical |

**Additional Notes:**
- Mostly unlabeled data
- Anomalies are rare and evolve over time (concept drift)
- Defining 'normal' is hard
- High volume of records

*Answer all 10 steps (Business Understanding → Alternative Approach) for this scenario.*

---

### 🟡 ⚠️ Scenario 9 — Entertainment — Movie Recommendation for a New User

**Scenario.** A streaming service wants to suggest movies to a brand-new user who has rated only 3 movies. They have a rich catalog describing each movie, and they want suggestions based on movie content.

**Business Goal.** Recommend movies to a new user based on the content of movies they liked.

**Dataset Information**

- **Rows:** 40,000 movies in catalog; 1 new user with 3 ratings
- **Target Variable:** None (no rating history to learn from)

| Feature | Type |
|---------|------|
| Genre | Categorical |
| Director | Categorical |
| Cast | Categorical (text) |
| Description | Text |
| Keywords | Text |

**Additional Notes:**
- TRAP: this is recommendation, not clustering or association
- Cold-start user (almost no history)
- Heavy text features
- Content-based fits because it relies on item content, not other users

*Answer all 10 steps (Business Understanding → Alternative Approach) for this scenario.*

---

### 🔴 ⚠️ Scenario 10 — Social Media — Auto-Organizing Posts by Topic

**Scenario.** A social platform wants to automatically organize millions of text posts into topic groups (sports, politics, food, tech…) to build trending sections. The categories are NOT predefined and there are NO topic labels.

**Business Goal.** Automatically group text posts by topic without predefined categories.

**Dataset Information**

- **Rows:** 5,000,000 text posts
- **Target Variable:** None (no labels, categories undefined)

| Feature | Type |
|---------|------|
| Post Text | Text |
| Timestamp | Numerical/Datetime |
| Likes | Numerical |
| Shares | Numerical |

**Additional Notes:**
- TRAP: feels like classification, but there are no labels → clustering
- Text must be converted to numbers (TF-IDF)
- High dimensionality / sparsity
- Number of topics unknown

*Answer all 10 steps (Business Understanding → Alternative Approach) for this scenario.*

---

### 🔴 ⚠️ Scenario 11 — Finance — Credit Card Fraud Detection (Extreme Imbalance)

**Scenario.** A payment processor needs to catch fraudulent transactions in near real-time without annoying legitimate customers. They DO have a labeled fraud history, but fraud is extremely rare (~0.1%). Two valid framings exist: supervised classification or unsupervised anomaly detection.

**Business Goal.** Detect fraudulent transactions in near real-time.

**Dataset Information**

- **Rows:** 50,000,000 transactions / month
- **Target Variable:** Fraud Label (Fraud / Legit) — extremely imbalanced

| Feature | Type |
|---------|------|
| Amount | Numerical |
| Time | Numerical |
| Merchant Category | Categorical |
| Location | Categorical |
| Device Info | Categorical |
| Fraud Label | Categorical (Fraud / Legit) — rare label |

**Additional Notes:**
- TRAP: classification vs. anomaly detection — both defensible
- Extreme class imbalance (~0.1% fraud)
- Real-time latency requirement
- High false-positive cost
- Fraud patterns evolve

*Answer all 10 steps (Business Understanding → Alternative Approach) for this scenario.*

---

### 🔴 ⚠️ Scenario 12 — Logistics — Compressing High-Dimensional Sensor Data

**Scenario.** A delivery fleet collects 200+ sensor readings per vehicle per second. Models train too slowly and the data is impossible to visualize. Before doing anything else, the team wants to reduce the number of features while keeping most of the information, then view the data in 2D.

**Business Goal.** Reduce the number of features while retaining most information; enable 2D visualization.

**Dataset Information**

- **Rows:** 10,000,000 sensor snapshots
- **Target Variable:** None (the goal is feature reduction, not prediction)

| Feature | Type |
|---------|------|
| 200+ sensor channels | Numerical (GPS, accelerometer, engine, fuel, braking…) |
| Many features are highly correlated | Numerical |

**Additional Notes:**
- TRAP: this is dimensionality reduction, NOT clustering (clustering may come later)
- Features highly correlated and on different scales
- Must choose how much variance to keep
- Very high dimensionality

*Answer all 10 steps (Business Understanding → Alternative Approach) for this scenario.*

---

### 🔴 Scenario 13 — Banking — Segmentation with Unknown Groups and Outliers

**Scenario.** A bank wants to segment millions of customers by transaction behavior. The data has clusters of different shapes and densities plus many unusual outlier accounts. The business does NOT know how many segments exist and wants outlier accounts isolated rather than forced into a group.

**Business Goal.** Segment customers AND isolate unusual outlier accounts.

**Dataset Information**

- **Rows:** 2,000,000 customers
- **Target Variable:** None (no labels)

| Feature | Type |
|---------|------|
| Avg Monthly Transactions | Numerical |
| Avg Transaction Value | Numerical |
| Account Age | Numerical |
| Number of Products Held | Numerical |
| Online Activity Score | Numerical |

**Additional Notes:**
- Unknown number of clusters
- Clusters of arbitrary shapes and densities
- Many outliers that must be isolated, not absorbed
- Large dataset

*Answer all 10 steps (Business Understanding → Alternative Approach) for this scenario.*

---

## 4. 🏆 Bonus Challenge — "Defend Your Architecture"

These scenarios have **no single right answer**. For each one you attempt:

- Propose at least two valid algorithms.
- Compare them across: accuracy, interpretability, scalability, training cost, and data requirements.
- Explain the trade-offs.
- Recommend ONE solution for production and justify it given real-world constraints.

### ⭐ Bonus 1 — Healthcare — Disease Diagnosis: Interpretability vs. Accuracy

**Scenario.** A hospital wants to predict whether a patient has a certain disease from 30 clinical features. Doctors must be able to explain every prediction to patients and regulators.

**Compare:** An interpretable model (e.g., Logistic Regression / shallow Decision Tree) vs. a higher-accuracy 'black box' (e.g., Random Forest / SVM).

---

### ⭐ Bonus 2 — E-commerce — Recommendations at Scale

**Scenario.** A fast-growing store currently uses content-based recommendations. They now have rich product metadata AND a lot of user behavior, and many brand-new (cold-start) users.

**Compare:** Content-based recommendation vs. a behavior-driven recommendation strategy (and a hybrid).

---

### ⭐ Bonus 3 — Finance — Fraud: Classification vs. Anomaly Detection

**Scenario.** Revisit fraud detection, but the trade-off is now the framing itself: use the labeled fraud history, or treat fraud as deviation from normal behavior?

**Compare:** Supervised classification (labeled fraud) vs. unsupervised anomaly/outlier detection (and whether to combine them).

---

### ⭐ Bonus 4 — Retail — Customer Grouping: K-Means vs. Hierarchical vs. DBSCAN

**Scenario.** A retailer wants customer segments from millions of records and is torn between three clustering algorithms. Consider known/unknown cluster count, scale, cluster shape, and outliers.

**Compare:** K-Means vs. Hierarchical Clustering vs. DBSCAN.

---

## 5. 📊 Evaluation Rubric (100 Points)

| Criterion | Points | What we look for |
|-----------|:------:|------------------|
| **Business & Dataset Understanding** | 15 | Correctly states the business goal, what data is available, whether a target exists, and feature types. |
| **Learning Type Identification** | 10 | Correctly labels Supervised vs. Unsupervised. |
| **ML Task Identification** | 15 | Picks the correct task (Regression / Classification / Clustering / Association / Recommendation / Dimensionality Reduction), incl. binary vs. multi-class. |
| **Algorithm Selection** | 15 | Chooses appropriate, taught algorithm(s) for the data and task. |
| **Justification & Reasoning** | 20 | The core of the grade. Clear, correct reasoning for the type, task, and algorithm. Recognizes traps. |
| **Data Preparation** | 10 | Identifies sensible preprocessing (encoding, scaling/normalization, missing values, text vectorization, split). |
| **Handling Challenges** | 10 | Spots realistic data problems (imbalance, outliers, leakage, high dimensionality) and proposes valid fixes. |
| **Communication Clarity** | 5 | Answers are organized, professional, and easy to follow. |
| **Total** | **100** | |

The **Bonus Section** is graded separately (up to **+15 points**) on comparison quality, trade-off depth, and a well-justified production recommendation.

### Grade Bands

- **90–100** — Thinks like an ML Engineer. Reasoning is sound even on traps.
- **75–89** — Strong; correct choices with good (if occasionally thin) justification.
- **60–74** — Passing; right tasks but reasoning is shallow or misses traps.
- **Below 60** — Needs review; confuses learning types or task categories.

---

## 6. Submission Instructions

- Complete the Excel template 'ML_Detective_Challenge_Answer_Template.xlsx'.
- Fill Sheet 1 ('Scenario Answers') — one row per scenario (13 rows).
- Fill Sheet 2 ('Bonus Challenge') — one row per bonus scenario you attempt.
- Fill Sheet 3 ('Self Evaluation') — reflect on your hardest decisions.
- Reasoning matters more than the final label — write full sentences, not one-word answers.
- Upload your completed .xlsx file to the learning platform by the deadline.

---

## Appendix — Allowed Concepts (Bootcamp Toolbox)

**Supervised:** Linear Regression · Logistic Regression · Decision Trees · Random Forest · KNN · SVM · Naive Bayes · Feature Scaling · Normalization · Standardization · Cost Function · Gradient Descent · Model Evaluation · Handling Imbalanced Datasets

**Unsupervised:** K-Means · Hierarchical Clustering · DBSCAN · Association Rules · Recommendation Systems · TF-IDF · Content-Based Recommendation · PCA · Outlier / Anomaly Detection

---
*© AI Builders Bootcamp — Machine Learning Detective Challenge (Student Version). A companion PDF and an Excel answer template are provided. Do not look at the Instructor Answer Key until you have completed your attempt.*