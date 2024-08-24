# Home Credit Default Risk Prediction

## Project Overview

This project is part of the **Home Credit Default Risk** competition hosted on Kaggle. The primary objective is to predict the probability of a customer defaulting on a loan using various features from the provided datasets. By building an accurate predictive model, Home Credit can enhance its credit risk management process, minimizing potential financial losses.

## Project Details

- **Project Name:** Home Credit Default Risk Prediction
- **Author:** [Aarohi Verma]
- **Kaggle Competition Link:** [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
- **Dataset:** Multiple CSV files provided by Home Credit, including application records, bureau records, previous loan details, and more, offering a comprehensive view of customers' credit histories and personal information.

## Objective

To develop a machine learning model that predicts the likelihood of a customer defaulting on a loan. This prediction will help Home Credit make more informed lending decisions by identifying high-risk customers early.

## Project Components

### 1. Data Preprocessing
   - Handled missing values using imputation techniques (mean, median, mode, or specific value for categorical features).
   - Encoded categorical features using one-hot encoding and label encoding.
   - Performed data normalization and scaling to standardize features.
   - Merged multiple datasets to create a unified dataset for analysis.

### 2. Exploratory Data Analysis (EDA)
   - Analyzed feature distributions using histograms and density plots.
   - Visualized correlations between features and the target variable using heatmaps and scatter plots.
   - Identified trends and patterns that could impact loan default risk.

### 3. Feature Engineering
   - Created new features, such as income-to-loan ratio, credit-to-income ratio, and days employed relative to age.
   - Generated statistical features by aggregating data from previous loans and credit bureau records.
   - Selected relevant features using techniques like feature importance scores and correlation analysis.

### 4. Model Development
   - Experimented with various machine learning algorithms:
     - Logistic Regression
     - Random Forest
     - Gradient Boosting Machines (GBM)
     - XGBoost
     - LightGBM
   - Used cross-validation to ensure robust model evaluation.
   - Applied hyperparameter tuning using grid search and randomized search to optimize model performance.

### 5. Model Evaluation
   - Evaluated models using metrics such as:
     - ROC-AUC score
     - Precision, Recall, F1-score
     - Confusion Matrix for detailed analysis
   - Compared the performance of different models to select the best one.
   - Analyzed feature importance to understand key predictors of default.

### 6. Model Deployment (Optional)
   - Saved the trained model using joblib for future predictions.
   - Developed a pipeline for real-time predictions based on incoming customer data.

## Results

- The final model achieved a **ROC-AUC score of [Your Score]** on the test set.
- The model successfully identified high-risk customers, helping Home Credit reduce the risk of loan defaults.

## Key Insights

- Customers with higher income-to-loan ratios and longer credit histories tend to have lower default rates.
- Previous loan defaults and irregular payment histories are strong indicators of future defaults.
- Employment status, age, and loan type significantly influence default risk.

## Challenges

- Handling the large number of features and optimizing model performance.
- Managing class imbalance in the target variable (more non-defaults than defaults).
- Ensuring model generalization to unseen data to prevent overfitting.

## Future Work

- **Incorporate additional data sources:** Using external financial data or social media data to improve prediction accuracy.
- **Advanced Modeling:** Experimenting with deep learning models such as neural networks.
- **Feature Selection:** Implementing advanced techniques like Recursive Feature Elimination (RFE) for optimal feature selection.
- **Real-time Prediction Pipeline:** Developing an API for real-time scoring of new loan applications.

## Known Issues

- High multicollinearity among certain features.
- The model requires further tuning to handle edge cases more effectively.

## Conclusion

The Home Credit Default Risk prediction project demonstrates the application of data science and machine learning techniques in financial risk management. By accurately predicting default risk, financial institutions can improve their decision-making processes, enhance customer experience, and minimize financial losses.

## Acknowledgements

- **Kaggle** for hosting the competition and providing the dataset.
- **Home Credit Group** for the opportunity to work on a real-world credit risk problem.
- **The Data Science Community** for valuable insights, shared resources, and discussions that guided the project development.

---

