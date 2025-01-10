# Exotic Fruit Classification Project
This repository is the fourth project of the master's degree in AI Engineering with Profession AI.

## Project Overview
The goal of this project is to develop a machine learning model capable of predicting the type of fruit based on numerical characteristics.

### Problem Statement
The current process of classifying exotic fruits is manual, error-prone, and inefficient. An automated and accurate system is crucial to optimizing business operations and maintaining high-quality standards.

By implementing an automated classification model, TropicTaste Inc. will:

- **Enhance Operational Efficiency:** Automating classification will reduce time and resource requirements, increasing productivity.
- **Minimize Human Errors:** A machine learning model will ensure higher accuracy in classification.
- **Optimize Inventory Management:** Accurate classification will enable better inventory handling, ensuring optimal storage conditions for each fruit type.
- **Increase Customer Satisfaction:** Accurate identification and classification will help maintain high-quality standards, improving customer satisfaction.

### Dataset
The project uses a dataset contains the following variables:

1. **Fruit:** The type of fruit. This is the target variable to predict.
2. **Weight (g):** The weight of the fruit in grams (continuous variable).
3. **Average Diameter (mm):** The average diameter of the fruit in millimeters (continuous variable).
4. **Average Length (mm):** The average length of the fruit in millimeters (continuous variable).
5. **Skin Hardness (1-10):** The skin hardness of the fruit on a scale from 1 to 10 (continuous variable).
6. **Sweetness (1-10):** The sweetness of the fruit on a scale from 1 to 10 (continuous variable).
7. **Acidity (1-10):** The acidity of the fruit on a scale from 1 to 10 (continuous variable).

### Algorithm
The K-Nearest Neighbors (KNN) algorithm is implemented for classification.

### Expected Output
The model must accurately predict the type of fruit based on the provided data.

### 1. Dataset Preparation
- Load and preprocess the exotic fruit data.
- Handle missing values, normalize, and scale the data.

### 2. KNN Model Implementation
- Develop and train the KNN model.
- Optimize hyperparameters to improve predictive accuracy.

### 3. Performance Evaluation
- Use cross-validation techniques to evaluate the model's generalization capability.
- Calculate performance metrics such as accuracy and classification error.

### 4. Results Visualization
- Create charts to visualize and compare the model's performance.
- Analyze and interpret the results to identify areas for improvement.

## Project Structure
- **`data`**: Contains the dataset csv files.
- **`notebooks`**: Includes Jupyter notebooks for exploratory data analysis and visualization.
- **`src/`**:
  - `data_processing.py`: Functions for loading, cleaning, and preprocessing data.
  - `knn.py`: Implements the KNN classifier and its evaluation pipeline.
  - `constants.py`: RANDOM_STATE.
  - `results`: Stores generated evaluation charts, metrics, and reports.
- **`README.md`**: Project documentation.
