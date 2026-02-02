I built this project after realizing that many students, including me, experience days where productivity drops without clearly understanding why. Instead of using common Kaggle datasets, I decided to design my own behavioral dataset that captures daily habits like sleep, study time, phone usage, mood, exercise, and distractions. The goal was to analyze how these lifestyle factors affect productivity and lead to burnout. Using this data, I engineered a custom Burnout Score, performed exploratory data analysis to find meaningful patterns, built a regression model to predict productivity, and a classification model to predict burnout level. Finally, I created a Streamlit app where a user can enter daily inputs and get predictions, turning the analysis into an interactive data product.

--Steps followed

1. Designed behavioral dataset schema and generated synthetic data
2. Performed data cleaning and feature creation such as weekend flag and burnout score
3. Conducted EDA and derived insights from correlations and patterns
4. Built Linear Regression model to predict productivity score
5. Built Random Forest model to classify burnout level
6. Saved trained models using pickle for deployment
7. Developed a Streamlit app to take user input and show predictions

-- Here is the streamlit app visual representation pdf
[Streamlit.pdf](https://github.com/user-attachments/files/25020887/Streamlit.pdf)

