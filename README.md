# **Gym Customer Churn Analysis and Clustering Project**

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Dataset Information](#dataset-information)
3. [Project Workflow](#project-workflow)
    - [Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
    - [Churn Prediction Modeling](#2-churn-prediction-modeling)
    - [Customer Segmentation Using Clustering](#3-customer-segmentation-using-clustering)
4. [Results and Insights](#results-and-insights)
    - [Model Performance Summary](#model-performance-summary)
    - [Cluster Analysis Summary](#cluster-analysis-summary)
5. [Conclusion and Recommendations](#conclusion-and-recommendations)
6. [Technologies Used](#technologies-used)
7. [How to Run the Project](#how-to-run-the-project)
8. [Author](#author)

---

## **Project Overview**
This project focuses on analyzing gym customer data to understand customer churn behavior and segment customers using clustering techniques. The goal is to identify high-risk churners, predict churn, and develop actionable insights to help improve customer retention strategies.

In this project, we:
- Analyzed the features associated with customer churn.
- Built and evaluated various machine learning models for churn prediction.
- Performed customer segmentation using K-means clustering.
- Provided business recommendations to reduce churn based on the analysis.

---

## **Dataset Information**
The dataset contains gym customer information, including features such as:
- **Customer Demographics** (age, gender)
- **Engagement** (class attendance, lifetime membership)
- **Contract Details** (contract period, months until contract ends)
- **Spending** (additional charges)
- **Churn** (whether the customer has churned or not)

---

## **Project Workflow**

### **1. Exploratory Data Analysis (EDA)**
- Explored the distributions of key features, such as customer age, contract length, lifetime, and class attendance.
- Analyzed the correlations between features to understand which factors influence customer churn.
- Visualized the relationships between customer lifetime, class frequency, and churn behavior.

### **2. Churn Prediction Modeling**
- Implemented **Logistic Regression**, **Random Forest**, **Gradient Boosting**, and **Decision Tree** models to predict customer churn.
- Evaluated models using metrics such as **Accuracy**, **Precision**, **Recall**, **F1 Score**, and **ROC AUC**.
- Compared the model performances to identify the best model for predicting churn.

### **3. Customer Segmentation Using Clustering**
- Applied **K-means clustering** to segment customers into 5 clusters.
- Visualized clusters using **PCA** to project high-dimensional data into 2D space.
- Analyzed the **churn rate** for each cluster to identify high-risk groups and loyal customers.
- Provided targeted recommendations based on cluster characteristics.

---

## **Results and Insights**

### **Model Performance Summary**
| Model               | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---------------------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 91.75%   | 88.20%    | 77.72% | 82.63%   | 96.71%  |
| **Random Forest**       | 91.25%   | 85.48%    | 78.71% | 81.96%   | 95.99%  |
| **Gradient Boosting**   | 92.50%   | 87.77%    | 81.68% | 84.61%   | 97.25%  |
| **Decision Tree**       | 88.88%   | 78.11%    | 77.72% | 77.91%   | 85.18%  |

**Key Findings:**
- **Gradient Boosting** provided the best performance, with the highest accuracy (92.5%) and ROC AUC (97.25%).
- **Logistic Regression** performed well, particularly in precision, making it a viable alternative for churn prediction.

### **Cluster Analysis Summary**
| Cluster | Churn Rate | Key Characteristics |
|---------|------------|---------------------|
| **0**   | 93.22%     | High churn risk, disengaged customers |
| **1**   | 43.33%     | Moderate churn risk, partially engaged |
| **2**   | 1.37%      | Loyal customers, high engagement |
| **3**   | 26.68%     | Low to moderate churn risk |
| **4**   | 0.32%      | Extremely loyal customers |

**Key Insights:**
- Cluster 0 represents the **highest churn risk**, with 93.22% churn. These customers are likely disengaged and require immediate attention to prevent further churn.
- Clusters 2 and 4 represent **highly loyal customer segments**, with churn rates of 1.37% and 0.32%, respectively. These groups should be rewarded to maintain loyalty.

---

## **Conclusion and Recommendations**
1. **Target Cluster 0 for Retention**: Implement retention strategies such as personalized offers, customer surveys, and targeted incentives to reduce churn in this high-risk group.
2. **Engage Cluster 1**: Focus on engagement initiatives, such as loyalty programs or fitness challenges, to retain customers in Cluster 1.
3. **Reward Loyal Customers (Clusters 2 and 4)**: Offer loyalty rewards, premium services, or VIP programs to retain the gym’s most loyal customers.
4. **Leverage Predictive Models**: Use the **Gradient Boosting** model to monitor and predict churn on a regular basis, enabling proactive retention efforts.

---

## **Technologies Used**
- **Python**: Programming language used for data analysis and modeling.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn**: For machine learning models and clustering.
- **SciPy**: For hierarchical clustering and dendrogram visualization.
- **Jupyter Notebook**: Environment for writing and running the project code.

---

## **How to Run the Project**

1. Clone this repository:
    ```bash
    git clone https://github.com/bhutto17/Fitness-Center-Retention-Analysis.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Jupyter Notebook:
    ```bash
    jupyter notebook Fitness_Center_Retention_Analysis.ipynb
    ```

---

## **Author**
**Faizan Bhutto**  
[LinkedIn](https://www.linkedin.com/in/faizanbhutto) | [GitHub](https://github.com/bhutto17)

Feel free to reach out if you have any questions or feedback regarding this project.
