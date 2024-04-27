import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the Excel file
base_dir = Path(__file__).resolve().parent
file_path = os.path.join(base_dir, "input/private-investment-in-artificial-intelligence.xls")
df = pd.read_excel(file_path)

first_few_rows = df.head()
basic_stats = df.describe()
missing_values = df.isnull().sum()
unique_values_count = df['Organization Type'].value_counts() if 'Organization Type' in df.columns else pd.DataFrame({"Message": ["'Organization Type' column not found."]})
mean_by_entity = df.groupby('Entity').mean()
output_file_path = os.path.join(base_dir, "output/Unleashing AI/analysis_output.csv")
first_few_rows.to_csv(output_file_path.replace('.csv', '_first_few_rows.csv'))
basic_stats.to_csv(output_file_path.replace('.csv', '_basic_stats.csv'))
missing_values.to_csv(output_file_path.replace('.csv', '_missing_values.csv'))
if not unique_values_count.empty:
    unique_values_count.to_csv(output_file_path.replace('.csv', '_unique_values_count.csv'))
else:
    with open(output_file_path.replace('.csv', '_unique_values_count.csv'), 'w') as f:
        f.write("'Organization Type' column not found.")

mean_by_entity.to_csv(output_file_path.replace('.csv', '_mean_by_entity.csv'))

print(f"Analysis results saved to separate CSV files.")

# Drop columns that only contain missing values
df = df.drop(columns=['Code'], errors='ignore')
# Remove rows where the 'Year' column is NaN
df = df.dropna(subset=['Year'])

# Let 'X' contain features and 'y' contains labels
X = df.drop(columns=['Entity'])
y = df['Entity']

# Pipeline for imputing, scaling, and logistic regression
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit and predict
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print("Model Evaluation Metrics:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")


# Perform cross-validation
cv_scores = cross_val_score(pipeline, X, y, cv=5)  # 5-fold cross-validation
print("Cross-Validation Scores:", cv_scores)

# Load dataset (time-series)
df = pd.read_excel(file_path, index_col='Entity') # Adjust file path and index column
df = df.drop('Total', axis=0)

norm = plt.Normalize(df['World'].min(), df['World'].max())
colors = plt.cm.viridis(norm(df['World']))  # Or any other colormap

plt.figure(figsize=(14, 7))  # Increase figure size for better readability
plt.bar(df.index, df['World'], color=colors)  # Replace 'Year' with the column name containing numeric data

plt.title('Entity')
plt.xlabel('Entity')
plt.ylabel('World')  # Adjust the label to represent what 'Year' stands for, e.g., 'Investment Amount'
plt.xticks(rotation=90)  # Rotate labels to prevent overlap
plt.subplots_adjust(bottom=0.3)  # Increase the bottom margin to ensure entity labels are not cut off
plt.tight_layout()  # Adjust the layout
plt.show()

# Load dataset (time-series)
df = pd.read_excel(file_path, index_col='Entity') # Adjust file path and index 
df['Year'] = pd.to_datetime(df['Year'], format='%Y') 

# Sort DataFrame by 'Year' to ensure it's in chronological order
df.sort_values('Year', inplace=True)
df = df.drop_duplicates(subset='Year', keep='first')  # keeps first occurrence

# Set 'Year' as the DataFrame's index and sort it
df.set_index('Year', inplace=True)
df.sort_index(inplace=True)
# Let's ensure this is a numeric column representing the time series data
if 'World' in df.columns:
    result = seasonal_decompose(df['World'], model='additive', period=1)
    result.plot()
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=90)  # Rotate labels to prevent overlap
    plt.subplots_adjust(bottom=0.5)  # Increase the bottom margin to ensure entity labels are not cut off
    plt.tight_layout()  # Adjust the layout
    plt.show()
else:
    print("The column 'World' does not exist in the DataFrame.")
          
        
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:]
order = (1, 1, 1) # ARIMA parameters
seasonal_order = (1, 1, 1, 12) # Seasonal parameters
model = SARIMAX(train['World'], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
model_fit = model.fit()

# Forecast
forecast = pd.Series(model_fit.forecast(steps=len(test)))

# Evaluate the model
mse = mean_squared_error(test['World'], forecast)
print('Mean Squared Error:', mse)

# Plot the forecast
plt.figure(figsize=(10, 6))
plt.plot(train.index, train['World'], label='Train')
plt.plot(test.index, test['World'], label='Test')
plt.plot(test.index, forecast, label='Forecast')
plt.title('SARIMA Forecast')
plt.xlabel('Date')
plt.ylabel('Values (Exponent of 10)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=90)  # Rotate labels to prevent overlap
plt.subplots_adjust(bottom=0.3)  # Increase the bottom margin to ensure entity labels are not cut off
plt.tight_layout()  # Adjust the layout
plt.show()