

# Patient Health and Lifestyle Dataset Analysis

This project involves the analysis of a comprehensive health dataset that includes patient demographic details, lifestyle factors, and medical history, with a focus on predicting health outcomes using machine learning models.

## Table of Contents

- [Dataset Overview](#dataset-overview)
- [Data Dictionary](#data-dictionary)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
  - [2. Preprocessing](#2-preprocessing)
  - [3. Feature Engineering](#3-feature-engineering)
  - [4. Model Training](#4-model-training)
  - [5. Model Evaluation](#5-model-evaluation)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)
- [Citation](#citation)

## Dataset Overview

This dataset captures various health-related features, including demographic information, health conditions, and lifestyle choices, making it valuable for predicting outcomes such as heart disease risk. It includes information on:

- **Demographic details**: Age, sex, and state of residence.
- **Health conditions**: History of heart disease, stroke, diabetes, asthma, COPD, arthritis, and more.
- **Lifestyle factors**: Smoking, alcohol consumption, vaccination status, and physical difficulties.

The dataset is structured to facilitate the exploration of relationships between health conditions and lifestyle choices.

## Data Dictionary

| Feature                | Description |
|------------------------|-------------|
| `PatientID`            | Unique identifier for each patient. |
| `State`                | Geographic state of residence. |
| `Sex`                  | Gender of the patient. |
| `GeneralHealth`        | Self-reported health status. |
| `AgeCategory`          | Categorized age group of the patient. |
| `HeightInMeters`       | Height of the patient (in meters). |
| `WeightInKilograms`    | Weight of the patient (in kilograms). |
| `BMI`                  | Body Mass Index, calculated from height and weight. |
| `HadHeartAttack`       | Indicator of whether the patient had a heart attack. |
| `HadAngina`            | Indicator of whether the patient experienced angina. |
| `HadStroke`            | Indicator of whether the patient had a stroke. |
| `HadAsthma`            | Indicator of whether the patient has asthma. |
| `HadSkinCancer`        | Indicator of whether the patient had skin cancer. |
| `HadCOPD`              | Indicator of whether the patient had chronic obstructive pulmonary disease (COPD). |
| `HadDepressiveDisorder`| Indicator of whether the patient was diagnosed with a depressive disorder. |
| `HadKidneyDisease`     | Indicator of whether the patient had kidney disease. |
| `HadArthritis`         | Indicator of whether the patient had arthritis. |
| `HadDiabetes`          | Indicator of whether the patient had diabetes. |
| `DeafOrHardOfHearing`  | Indicator of hearing impairment. |
| `BlindOrVisionDifficulty` | Indicator of vision impairment. |
| `DifficultyConcentrating` | Indicator of concentration difficulties. |
| `DifficultyWalking`    | Indicator of walking difficulties. |
| `DifficultyDressingBathing` | Indicator of difficulties in dressing or bathing. |
| `DifficultyErrands`    | Indicator of difficulties in running errands. |
| `SmokerStatus`         | Status of whether the patient is a smoker. |
| `ECigaretteUsage`      | Indicator of e-cigarette usage. |
| `ChestScan`            | Indicator of whether the patient had a chest scan. |
| `RaceEthnicityCategory`| Race or ethnicity of the patient. |
| `AlcoholDrinkers`      | Status of whether the patient consumes alcohol. |
| `HIVTesting`           | Status of whether the patient was tested for HIV. |
| `FluVaxLast12`         | Status of whether the patient received a flu vaccine in the last 12 months. |
| `PneumoVaxEver`        | Status of whether the patient ever received a pneumococcal vaccine. |
| `TetanusLast10Tdap`    | Status of whether the patient received a tetanus vaccine in the last 10 years. |
| `HighRiskLastYear`     | Indicator of whether the patient was at high risk in the last year. |
| `CovidPos`             | Status of whether the patient tested positive for COVID-19. |

## Project Structure

```
.
├── README.md               # Project documentation
├── data                    # Folder containing the dataset
├── notebooks               # Jupyter Notebooks for analysis
│   └── analysis.ipynb
├── src                     # Source code for data processing, modeling, and evaluation
│   ├── preprocessing.py    # Data preprocessing functions
│   ├── feature_engineering.py # Feature engineering functions
│   └── model_training.py   # Model training scripts
└── requirements.txt        # Required libraries
```

## Installation

To install the required packages, use the following command:

```bash
pip install -r requirements.txt
```

## Usage

The project includes the following steps for data analysis and model training:

### 1. Exploratory Data Analysis (EDA)

Perform EDA to understand the dataset’s structure and examine distributions and correlations. Visualizations are created using libraries like Matplotlib and Seaborn.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Example: Plotting BMI distribution
sns.histplot(data=df, x="BMI", hue="HadHeartAttack", kde=True)
plt.title("BMI Distribution by Heart Attack Status")
plt.show()
PBP(data)
PDP(data)
PlotPie(data)
```

![](https://github.com/Arif-miad/Hearte-Disease-Prediction-/blob/main/h1.png)
![](https://github.com/Arif-miad/Hearte-Disease-Prediction-/blob/main/h2.png)
![](https://github.com/Arif-miad/Hearte-Disease-Prediction-/blob/main/h3.png)

```python
plt.figure(figsize=(10, 8))

correlation_matrix = data[['HeightInMeters', 'WeightInKilograms', 'BMI','HadHeartAttack',

       'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD',

       'DifficultyConcentrating']].astype(float).corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)

plt.title('Correlation Heatmap')

plt.show()
```



### 2. Preprocessing

This step includes handling missing values, encoding categorical variables, and scaling numeric features to prepare the dataset for model training.

### 3. Feature Engineering

Create new features or transform existing ones to enhance the predictive power of the models.

```python
# Example: Creating a new feature for age in numeric format
df['AgeNumeric'] = df['AgeCategory'].map(age_mapping)
```

### 4. Model Training

Apply machine learning models such as **Logistic Regression**, **Random Forest**, and **XGBoost** to predict health outcomes based on patient data.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Logistic Regression example
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
```

### 5. Model Evaluation

Evaluate each model using accuracy, precision, recall, and F1-score to determine the most effective model.

```python
from sklearn.metrics import classification_report

# Model evaluation example
y_pred = log_reg.predict(X_test)
print(classification_report(y_test, y_pred))
```

## Results

The results of the project, including model performance metrics and insights gained from EDA, are documented here. Comparisons between models provide insights into the best predictors of health outcomes.

## Future Work

- **Explore Additional Models**: Test other machine learning algorithms like SVM or neural networks.
- **Feature Importance Analysis**: Identify the most influential factors in predicting health outcomes.
- **Deploy Model**: Create a web-based interface for real-time health risk prediction.

## License

This project is licensed under the MIT License.
