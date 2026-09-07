# Employee-Burn-out-Rate-Prediction

# Employee Burnout Rate Prediction Using People Analytics and HR Metrics

## Project Overview

Employee burnout is a growing concern in modern workplaces, affecting employee well-being, productivity, engagement, and organizational performance. This project uses Machine Learning and People Analytics techniques to predict employee burnout rates using HR-related metrics and workplace factors.

The objective of this project is to develop a regression model that accurately predicts employee burnout and identifies the key factors contributing to burnout risk. The insights generated can help organizations proactively address employee well-being concerns, improve workforce productivity, and reduce employee turnover.

---

# Tools and Technologies Used

- **Python** – Programming Language
- **Pandas** – Data Manipulation and Analysis
- **NumPy** – Numerical Computing
- **Matplotlib** – Data Visualization
- **Seaborn** – Statistical Data Visualization
- **Scikit-Learn** – Machine Learning and Preprocessing
- **XGBoost** – Gradient Boosting Regression Model
- **Jupyter Notebook** – Development Environment

---

# Problem Statement

The goal of this project is to build a machine learning regression model capable of predicting employee burnout rates based on workforce and HR-related metrics.

Organizations can use these predictions to:

- Identify employees at risk of burnout.
- Improve workforce management strategies.
- Balance employee workloads effectively.
- Support employee mental health initiatives.
- Improve retention and productivity.
- Enable proactive HR decision-making.

---

# Dataset Description

The project uses two datasets:

## Training Dataset

The training dataset contains both predictor variables and the target variable.

### Features

| Feature | Description |
|----------|-------------|
| Employee ID | Unique employee identifier |
| Date of Joining | Date employee joined the organization |
| Gender | Male or Female |
| Company Type | Product or Service company |
| WFH Setup Available | Availability of Work From Home setup |
| Designation | Employee designation level |
| Resource Allocation | Workload allocated to employee |
| Mental Fatigue Score | Employee fatigue score |
| Burn Rate | Target variable |

---

## Test Dataset

The test dataset contains all predictor variables but does not contain the Burn Rate column.

The trained model is used to predict burnout rates for these employees.

---

# Project Workflow

The project followed the following workflow:

1. Data Loading
2. Data Understanding
3. Data Cleaning
4. Feature Engineering
5. Exploratory Data Analysis (EDA)
6. Data Preprocessing
7. Model Building
8. Model Evaluation
9. Feature Importance Analysis
10. Test Set Prediction
11. Business Recommendations

---

# Data Loading

The datasets were loaded using Pandas.

```python
import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
```

Dataset information was explored using:

```python
train.shape

train.info()

train.describe()

train.head()
```

---

# Data Understanding

The initial analysis focused on:

- Dataset dimensions
- Data types
- Missing values
- Numerical feature distributions
- Target variable distribution
- Overall dataset quality

Missing values were identified using:

```python
train.isnull().sum()
```

---

# Data Cleaning

## Handling Missing Values

### Burn Rate

Since Burn Rate is the target variable, rows with missing Burn Rate values were removed.

```python
train = train.dropna(subset=['Burn Rate'])
```

### Mental Fatigue Score

Missing values were replaced using the median value.

```python
train['Mental Fatigue Score'].fillna(
    train['Mental Fatigue Score'].median(),
    inplace=True
)

test['Mental Fatigue Score'].fillna(
    train['Mental Fatigue Score'].median(),
    inplace=True
)
```

### Resource Allocation

Missing values were replaced using the median value.

```python
train['Resource Allocation'].fillna(
    train['Resource Allocation'].median(),
    inplace=True
)

test['Resource Allocation'].fillna(
    train['Resource Allocation'].median(),
    inplace=True
)
```

Median imputation was chosen because it reduces the impact of outliers while preserving the central tendency of the dataset.

---

## Removing Employee ID

Employee ID was removed because it does not contribute any predictive information.

```python
train.drop('Employee ID', axis=1, inplace=True)

test.drop('Employee ID', axis=1, inplace=True)
```

---

# Feature Engineering

## Date Conversion

The Date of Joining feature was converted to datetime format.

```python
train['Date of Joining'] = pd.to_datetime(train['Date of Joining'])

test['Date of Joining'] = pd.to_datetime(test['Date of Joining'])
```

---

## Date-based Features

New features were extracted from Date of Joining.

### Joining Month

```python
Joining_Month
```

### Joining Day

```python
Joining_Day
```

### Joining Quarter

```python
Joining_Quarter
```

Implementation:

```python
for df in [train, test]:

    df['Joining_Month'] = df['Date of Joining'].dt.month
    df['Joining_Day'] = df['Date of Joining'].dt.day
    df['Joining_Quarter'] = df['Date of Joining'].dt.quarter

    df.drop('Date of Joining', axis=1, inplace=True)
```

---

# Exploratory Data Analysis (EDA)

EDA was performed to uncover patterns, relationships, and trends within the dataset.

---

## Burn Rate Distribution

Purpose:

- Analyze burnout distribution
- Detect skewness
- Detect outliers

```python
sns.histplot(train['Burn Rate'], kde=True)
plt.title('Burn Rate Distribution')
plt.show()
```

---

## Mental Fatigue Score vs Burn Rate

Purpose:

- Examine the relationship between fatigue and burnout.

```python
sns.scatterplot(
    data=train,
    x='Mental Fatigue Score',
    y='Burn Rate'
)
```

### Insight

Employees with higher mental fatigue scores generally showed significantly higher burnout rates.

---

## Resource Allocation vs Burn Rate

Purpose:

- Understand how workload affects burnout.

```python
sns.boxplot(
    x='Resource Allocation',
    y='Burn Rate',
    data=train
)
```

### Insight

Burnout levels increase as resource allocation increases.

---

## Designation vs Burn Rate

Purpose:

- Determine whether burnout varies across employee seniority levels.

```python
sns.boxplot(
    x='Designation',
    y='Burn Rate',
    data=train
)
```

### Insight

Employees in higher designation levels generally experience greater burnout.

---

## Gender vs Burn Rate

Purpose:

- Compare burnout rates across genders.

```python
sns.boxplot(
    x='Gender',
    y='Burn Rate',
    data=train
)
```

### Insight

Differences were observed but were less significant than workload and fatigue-related factors.

---

## Company Type vs Burn Rate

Purpose:

- Compare burnout patterns in Product and Service organizations.

```python
sns.boxplot(
    x='Company Type',
    y='Burn Rate',
    data=train
)
```

### Insight

Both company types exhibited similar burnout trends.

---

## WFH Setup Availability vs Burn Rate

Purpose:

- Assess the impact of remote work support on burnout.

```python
sns.boxplot(
    x='WFH Setup Available',
    y='Burn Rate',
    data=train
)
```

### Insight

Employees with WFH support generally showed slightly lower burnout levels.

---

## Correlation Analysis

A heatmap was created to identify relationships among numerical variables.

```python
numeric_cols = [
    'Designation',
    'Resource Allocation',
    'Mental Fatigue Score',
    'Joining_Month',
    'Joining_Day',
    'Joining_Quarter',
    'Burn Rate'
]

corr = train[numeric_cols].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm'
)
```

---

# Data Preprocessing

## Numerical Features

```python
Designation
Resource Allocation
Mental Fatigue Score
Joining_Month
Joining_Day
Joining_Quarter
```

Missing values were handled using:

```python
SimpleImputer(strategy='median')
```

---

## Categorical Features

```python
Gender
Company Type
WFH Setup Available
```

Categorical variables were encoded using:

```python
OneHotEncoder()
```

---

# Train-Test Split

The data was divided into training and validation sets.

```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

# Model Development

## Linear Regression

A Linear Regression model was developed as the baseline model.

### Benefits

- Easy to interpret
- Fast training time
- Useful benchmark model

---

## Random Forest Regressor

A Random Forest model was developed to capture complex and non-linear relationships.

### Benefits

- Handles feature interactions
- Resistant to overfitting
- Robust performance on structured datasets

---

## XGBoost Regressor

XGBoost was selected as the final model because of its superior predictive performance.

### Model Parameters

```python
XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

### Benefits

- High predictive accuracy
- Strong performance on tabular datasets
- Handles non-linear relationships effectively

---

# Model Evaluation

The following regression metrics were used.

## Mean Absolute Error (MAE)

Measures average prediction error.

## Root Mean Squared Error (RMSE)

Measures overall model accuracy.

## R-Squared (R²)

Measures explained variance.

```python
MAE

RMSE

R²
```

---

# Feature Importance

The final model's feature importance analysis identified the key drivers of employee burnout.

## Most Important Features

1. Mental Fatigue Score
2. Resource Allocation
3. Designation
4. WFH Setup Available
5. Company Type
6. Gender

---

# Key Findings and Insights

- Mental Fatigue Score was the strongest predictor of employee burnout.
- Burnout rates increased consistently as mental fatigue scores increased.
- Employees with heavier workloads experienced significantly higher burnout levels.
- Resource Allocation demonstrated a strong positive relationship with burnout.
- Employees in higher designation levels showed higher burnout rates due to increased responsibilities.
- Employees with Work From Home support generally reported slightly lower burnout levels.
- Burnout levels increased substantially once Mental Fatigue Scores exceeded approximately 7.
- Organizational and workload-related factors had a stronger influence on burnout than demographic characteristics.
- Product and Service organizations showed similar burnout trends.
- Combining fatigue and workload indicators provided strong predictive power for burnout estimation.

---

# Business Recommendations

Based on the analysis, organizations should:

- Regularly monitor employee mental fatigue levels.
- Implement employee wellness and mental health support programs.
- Balance workloads more effectively across teams.
- Develop burnout risk monitoring systems using predictive analytics.
- Introduce flexible work arrangements where possible.
- Monitor high-designation employees who may experience additional pressure.
- Use machine learning models as early-warning systems for identifying at-risk employees.
- Conduct periodic employee well-being assessments.

---

# Limitations

- Limited number of HR variables available.
- No information on employee age, salary, department, or work experience.
- External and personal factors influencing burnout were not captured.
- Missing values may introduce uncertainty into predictions.
- Human behavior is complex and cannot be fully explained using available variables.
- The model's performance depends on the quality of the available data.

---

# Future Improvements

- Hyperparameter tuning using GridSearchCV or Optuna.
- Model ensemble techniques.
- SHAP explainability analysis.
- Real-time burnout monitoring dashboards.
- Streamlit deployment for interactive predictions.
- Integration of additional HR, behavioral, and employee wellness metrics.

---

# Project Structure

```text
Employee-Burnout-Prediction/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   └── burnout_prediction.ipynb
│
├── outputs/
│   ├── submission.csv
│   └── visualizations/
│
├── README.md
│
└── requirements.txt
```

---

# Conclusion

This project successfully developed a machine learning solution for predicting employee burnout rates using People Analytics and HR metrics. Through data cleaning, feature engineering, exploratory data analysis, and predictive modeling, the study identified Mental Fatigue Score and Resource Allocation as the strongest drivers of burnout.

Among the models evaluated, XGBoost provided the best performance due to its ability to capture complex relationships within the data. The resulting model can help organizations proactively identify employees at risk of burnout, improve workforce planning, support employee well-being initiatives, and enable data-driven HR decision-making.

By leveraging predictive analytics, organizations can create healthier work environments, improve employee satisfaction, and enhance overall organizational productivity.


