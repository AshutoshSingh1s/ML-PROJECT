🌍 GDP Prediction and Analysis
📌 Project Overview

This project focuses on GDP Prediction and Analysis using historical GDP data of countries.
It performs:

GDP Growth Calculation (year-over-year % change)

Exploratory Data Analysis (EDA) of GDP trends across countries

Visualization of GDP Growth using Matplotlib/Plotly

Machine Learning-based GDP Forecasting

The project can be used by researchers, students, and policymakers to analyze how countries' GDP changes over time and predict future economic growth.

⚡ Features

✅ Cleaned and preprocessed GDP dataset

✅ Computed yearly GDP growth % for each country

✅ Visualized GDP trends (line charts, comparative analysis)

✅ Built GDP Prediction Model using Linear Regression / ML techniques

✅ GUI (Tkinter-based) to interactively predict GDP values

🛠️ Tech Stack

Python (Pandas, NumPy, Matplotlib, Scikit-learn)

Plotly (for interactive visualizations)

Tkinter (for GUI interface)

Jupyter Notebook (for EDA and modeling)

📂 Project Structure
GDP-Prediction-Project/
│── data/
│   └── gdp.csv                # Dataset
│── src/
│   ├── filedata.py            # Handles GDP calculations & ML logic
│   ├── final.py               # Tkinter GUI for user interaction
│   └── utils.py               # Helper functions
│── notebooks/
│   └── gdp_analysis.ipynb     # Jupyter Notebook (EDA + Prediction)
│── README.md                  # Project Documentation

📊 Example Workflow
Step 1️⃣ Load Dataset
import pandas as pd
df = pd.read_csv("data/gdp.csv")

Step 2️⃣ Calculate GDP Growth %
df['GDP Growth %'] = df.groupby("Country Name")['Value'].pct_change() * 100

Step 3️⃣ Plot GDP Trend
import matplotlib.pyplot as plt
india = df[df['Country Name'] == 'India']
plt.plot(india['Year'], india['Value'])
plt.title("GDP Trend - India")
plt.show()

Step 4️⃣ Predict Future GDP
from sklearn.linear_model import LinearRegression
model = LinearRegression()
X = india[['Year']]
y = india['Value']
model.fit(X, y)
future_gdp = model.predict([[2030]])
print("Predicted GDP in 2030:", future_gdp[0])

📈 Sample Output

GDP growth trends of countries over time

Comparative analysis between countries

Future GDP prediction for selected years

🚀 How to Run

Install dependencies:
pip install -r requirements.txt
Run analysis notebook:
jupyter notebook notebooks/gdp_analysis.ipynb
Run GUI for prediction:
python src/final.py
