import pandas as pd 
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns

# Download latest version
generation_data_path = r"Solar_pred_data\Plant_1_Generation_Data.csv"
Weather_data_path = r"Solar_pred_data\Plant_1_Weather_Sensor_Data.csv"


#making data 
generation_data = pd.read_csv(generation_data_path)
Weather_data = pd.read_csv(Weather_data_path)



#what is the structure of the data
print (generation_data.info())
