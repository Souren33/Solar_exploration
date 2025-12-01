#%%
import pandas as pd 
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

file_path = 'SolarPrediction.csv'
data = pd.read_csv(file_path)
#print(data.head())



def clean_data(df):
    df = df.dropna()
    df = df[df['Radiation'] >= 0]
    return df


def finding_peak_radiation_per_day(df):
    # Convert UNIXTime to datetime and extract just the date
    df['DateTime'] = pd.to_datetime(df['UNIXTime'], unit='s')  # adjust unit if needed
    df['Date'] = df['DateTime'].dt.date
    
    # Find peak radiation for each day
    peak_per_day = df.groupby('Date')['Radiation'].max()
    
    return peak_per_day
    

def calculate_average_radiation(df):
    average_radiation = df['Radiation'].mean()
    return average_radiation

def plotting_radiation_vs_time(df):
    plt.figure(figsize=(10,6))
    sns.lineplot(x='UNIXTime', y='Radiation', data=df)
    plt.title('Solar Radiation Over Time')
    plt.xlabel('Time (UNIX)')
    plt.ylabel('Radiation')
    plt.show()




def training_knn_model(df, k_max=30):
    features = df[['Temperature', 'Humidity', 'Pressure']]
    target = df['Radiation']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    k_values = range(1, k_max + 1)
    train_scores = []
    test_scores = []
    
    for k in k_values:
        print(f'Training KNN with k={k}')
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f'k={k}, Mean Squared Error: {mse}')
        
        train_scores.append(knn.score(X_train, y_train))
        test_scores.append(knn.score(X_test, y_test))

    # Plot the elbow curve (AFTER the loop)
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, train_scores, marker='o', label='Training Score', linewidth=2)
    plt.plot(k_values, test_scores, marker='s', label='Testing Score', linewidth=2)
    plt.xlabel('K Value')
    plt.ylabel('Score (R²)')
    plt.title('Elbow Curve for KNN Regressor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Train final model with optimal k (pick from elbow curve)
    optimal_k = 10  # You decide based on the plot
    knn = KNeighborsRegressor(n_neighbors=optimal_k)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Final Model - k={optimal_k}, Mean Squared Error: {mse}')

    return knn, X_test, y_test


if __name__ == "__main__":
    cleaned_data = clean_data(data)
    peak_radiation = finding_peak_radiation_per_day(cleaned_data)
    average_radiation = calculate_average_radiation(cleaned_data)
    
    plt.figure(figsize=(12, 6))
    plt.plot(peak_radiation.index, peak_radiation.values, marker='o', linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('Peak Daily Solar Irradiance')
    plt.title('Peak Daily Solar Irradiance Over Time')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    knn_model, X_test, y_test = training_knn_model(cleaned_data, k_max=30)
    predictions = knn_model.predict(X_test)

#%%
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_test, alpha=0.5, color='red', label='Actual')
plt.scatter(y_test, predictions, alpha=0.5, color='blue', label='Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Radiation')
plt.ylabel('Radiation')
plt.title('KNN Model: Actual vs Predicted Radiation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%%
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Your predictions and actual values
y_true = y_test
y_pred = predictions

# 1. Mean Squared Error (MSE)
mse = mean_squared_error(y_true, y_pred)
print(f'Mean Squared Error: {mse}')

# 2. Root Mean Squared Error (RMSE) - easier to interpret
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')

# 3. Mean Absolute Error (MAE) - average prediction error
mae = mean_absolute_error(y_true, y_pred)
print(f'Mean Absolute Error: {mae}')

# 4. R² Score (0 to 1, closer to 1 is better)
r2 = r2_score(y_true, y_pred)
print(f'R² Score: {r2}')

# 5. Mean Absolute Percentage Error (MAPE) - percentage error
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print(f'Mean Absolute Percentage Error: {mape:.2f}%')
# %%




#Would it be intresting to see if there is a cluster 