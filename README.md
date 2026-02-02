I built this project after realizing that many students, including me, experience days where productivity drops without clearly understanding why. Instead of using common Kaggle datasets, I decided to design my own behavioral dataset that captures daily habits like sleep, study time, phone usage, mood, exercise, and distractions. The goal was to analyze how these lifestyle factors affect productivity and lead to burnout. Using this data, I engineered a custom Burnout Score, performed exploratory data analysis to find meaningful patterns, built a regression model to predict productivity, and a classification model to predict burnout level. Finally, I created a Streamlit app where a user can enter daily inputs and get predictions, turning the analysis into an interactive data product.

Steps followed

Designed behavioral dataset schema and generated synthetic data

Performed data cleaning and feature creation such as weekend flag and burnout score

Conducted EDA and derived insights from correlations and patterns

Built Linear Regression model to predict productivity score

Built Random Forest model to classify burnout level

Saved trained models using pickle for deployment

Developed a Streamlit app to take user input and show predictions
