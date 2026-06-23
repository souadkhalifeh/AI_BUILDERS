# 🏢 Unsupervised Learning Capstone Challenge

# 🎯 Your Mission

You have just been hired as a **Data Consultant** by an online retail company.

The company gives you one year of raw transaction data and tells you:

> "We don't really understand our customers. Tell us who they are, what they buy together, what products we should recommend, and what actions we should take."

Your job is not to simply apply algorithms.
Your job is to investigate the data, discover hidden patterns, and transform those discoveries into business decisions.
You are expected to think like a real Machine Learning Engineer:
* Understand the business problem.
* Analyze the available data.
* Decide which unsupervised techniques are appropriate.
* Explain your decisions.
* Extract meaningful insights.
* Communicate recommendations to stakeholders.

---

# 🧠 The Golden Rule

❌ "I used K-Means."

is not an answer.

✅ "We identified four customer segments. These customers are high-value loyal customers, so the company should create retention campaigns and personalized offers."

is an ML engineering answer.

Your reasoning and business understanding matter more than the code itself.

---

# 🎓 Learning Objectives

By completing this project, you will practice:

* Understanding a real-world business problem.
* Exploring and cleaning raw data.
* Creating meaningful features.
* Applying unsupervised learning techniques.
* Choosing appropriate algorithms.
* Evaluating clustering results.
* Discovering purchasing patterns.
* Building recommendation logic.
* Translating ML outputs into business recommendations.

---

# 📁 Project Setup

Create a new Jupyter Notebook:

```
YourName_Unsupervised_Capstone.ipynb
```

Your notebook should be written as a professional report that another person can understand.

Use:

* Markdown cells → explain your thinking and decisions.
* Code cells → implement your analysis.

The notebook should not only show:

"What did you do?"

It should explain:

"Why did you do it?"

---

# 📊 Dataset

## UCI Online Retail Dataset

This dataset contains real transactions from a UK-based online gift store.

Period:

December 2010 → December 2011

Approximately:

541,000 transactions

---

## Available Features

| Column      | Description               |
| ----------- | ------------------------- |
| InvoiceNo   | Transaction ID            |
| StockCode   | Product ID                |
| Description | Product name              |
| Quantity    | Number of purchased items |
| InvoiceDate | Purchase date and time    |
| UnitPrice   | Product price             |
| CustomerID  | Customer identifier       |
| Country     | Customer country          |

---

# ⚠️ Important

The dataset is intentionally messy.
Before applying any algorithm, you must investigate:
* Missing values
* Duplicate records
* Returns and cancellations
* Incorrect values
* Outliers
* Data quality issues

Data preparation is part of the challenge.

---

# 🚀 Project Workflow

You will complete the project in the following stages.

---

# Part 1 — Business Understanding & Data Exploration

## Goal

Understand the business problem and the dataset.

Answer:

### Business Questions

* What does the company want to discover?
* Why is unsupervised learning suitable for this problem?
* What business decisions could this analysis support?

---

### Dataset Exploration

Analyze:

* Dataset shape
* Feature types
* Missing values
* Duplicate rows
* Unique customers
* Unique products
* Transaction statistics

---

## Required Visualizations:

Include at least:

* Distribution of purchase values
* Top-selling products
* Customer distribution by country

---

# Part 2 — Data Cleaning & Feature Engineering

## Goal

Transform raw transactions into meaningful customer information.

Clean:

* Missing Customer IDs
* Cancelled transactions
* Invalid quantities
* Invalid prices

Explain every decision.

---

# Part 3 — Customer Segmentation

## Goal

Discover meaningful customer groups.

The company wants to answer:

> "Who are our customers?"

---

## Feature Engineering

The dataset contains transactions, not customers.
You must create customer-level features.
Create an RFM table:

## Recency
How recently did the customer purchase?

## Frequency
How often does the customer purchase?

## Monetary
How much money did the customer spend?

---

# Modeling Decision

Before clustering, answer:
1. Why is clustering appropriate?
2. Why is supervised learning not suitable?
3. Which clustering algorithm will you use?

Possible choices:
* K-Means
* Hierarchical Clustering
* DBSCAN

Explain your choice.

---

# Model Evaluation

If using K-Means:
Analyze:
* Elbow Method
* Silhouette Score

If using another clustering method:
Explain your evaluation approach.

---

# Cluster Analysis

After creating clusters:
Do not call them:

❌ Cluster 0
❌ Cluster 1

Create business personas.

Example:

| Persona           | Description                               |
| ----------------- | ----------------------------------------- |
| Champions         | Recent, frequent, high-spending customers |
| Loyal Customers   | Regular buyers with consistent purchases  |
| At-Risk Customers | Previously active but declining           |
| New Customers     | Recent customers with limited history     |

Your segments may be different.

---

## Business Question

For each customer group:
How should the company market to them?

---

# Part 4 — PCA Visualization

## Goal

Visualize customer segments.
Apply PCA to reduce your customer features into 2 dimensions.

Explain:

* Why PCA was used.
* How much variance is retained.
* Whether the visualization supports your clustering results.

---

## Important Concept

PCA does NOT create clusters.
PCA only helps with:

* Visualization
* Dimensionality reduction

The clustering decision happens before PCA.

---

# Required Visualization:
A 2D PCA scatter plot showing customer groups.

---

# Part 5 — Market Basket Analysis

## Goal

Discover products frequently purchased together.
Business question:
> "What products should we bundle or recommend together?"

---

## Steps:
Create a basket dataset:
Rows:
Transactions

Columns:
Products

Values:
Purchased or not purchased

Apply:
Association Rules / Apriori

Analyze:
* Support
* Confidence
* Lift

---

# Deliverable

Select your strongest rules.
Translate them into business language.

Example:
Technical:
"Rule A → Rule B"

Business:
"Customers who purchase X often purchase Y. The company should create a bundle offer."

---

# Part 6 — Recommendation System

## Goal

Recommend products to customers.
Build a simple recommendation approach using customer purchase history.

---

## Deliverables

For one customer:
Recommend:
Top 5 products

Explain:
* Why were these products recommended?
* What customer behavior caused this recommendation?

---

# Recommendation Discussion

Explain:

## Cold Start Problem

What happens when:
* A new customer has no history?
How can we solve it?

Examples:
* Popular products
* Content-based recommendation
* Hybrid systems

---

## Evaluation

Explain:
How would the company know recommendations are successful?

Examples:
* Click rate
* Purchase rate
* Customer engagement

---

# Part 7 — Consultant Business Report

## Goal
The final output that management cares about.
Create a 1–2 page business report.
Do not include code.
Explain your findings.

---

Your report should answer:
## 1. Customer Segments
Who are the customers?
Describe each persona.

---

## 2. Purchasing Patterns
What products are commonly bought together?

---

## 3. Recommendations
What products should customers receive?

---

## 4. Business Actions
What should management do?

Examples:
* Marketing campaigns
* Customer retention strategies
* Product bundles
* Personalized offers

---

# 📦 Final Submission Requirements

Submit:

## 1. Jupyter Notebook
Containing:
✅ Data exploration
✅ Cleaning
✅ Feature engineering
✅ Modeling decisions
✅ Visualizations
✅ Business insights

---

## 2. Business Report
A short summary for stakeholders.

---

## 3. README File (GitHub)
Your README should include:
* Project title
* Business problem
* Dataset description
* Techniques used
* Main findings
* Business recommendations

---

# ✅ Final Submission Checklist
Before submitting, verify:

☐ Notebook runs without errors
☐ All preprocessing decisions are explained
☐ Missing values and data issues were handled
☐ Features were selected intentionally
☐ Algorithm choice was justified
☐ Cluster evaluation was performed
☐ Customer segments have meaningful names
☐ PCA interpretation is included
☐ Association rules were translated into business insights
☐ Recommendations are explained
☐ Cold-start problem is discussed
☐ Final business report is completed
☐ README file is prepared for GitHub

---

# Final Reminder
The goal of this project is not to prove that you know algorithms.
The goal is to prove that you can take a messy business problem, analyze data, discover patterns, and communicate valuable insights.

Think like a Data Consultant. 🚀
