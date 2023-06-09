import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the folder path where the CSV files are located
folder_path = 'our_results/table3/cleaned_results'

# create new folder to save boxplots
boxplot_folder = 'our_results/boxplots'
os.makedirs(boxplot_folder, exist_ok=True)

# Get a list of all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# create set of all datasets in folder
datasets = set()
for csv_file in csv_files:
    # Extract the dataset name from the file name
    dataset_name = os.path.basename(csv_file).split('_')[0]
    datasets.add(dataset_name)

# Create an empty dictionary to store dataset-wise CSV data
dataset_csv_data = {}

# concatenate all four models of a dataset
for csv_file in csv_files:
    # Extract the dataset name from the file name
    dataset_name = os.path.basename(csv_file).split('_')[0]
    
    # Read the CSV file into a DataFrame
    csv_data = pd.read_csv(csv_file)
    
    # Check if dataset name exists in the dictionary
    if dataset_name in dataset_csv_data:
        # Append the CSV data to the existing dataset key
        dataset_csv_data[dataset_name] = pd.concat([dataset_csv_data[dataset_name], csv_data])
    else:
        # Create a new key-value pair for the dataset
        dataset_csv_data[dataset_name] = csv_data

# Iterate over the dataset-wise CSV data and save to new CSV files
for dataset_name, data in dataset_csv_data.items():
    ax = sns.boxplot(x='Type', y='mCPF', data=data, order=['N', 'C', 'P', 'CP'])

    # Calculate the average per 'Type' for the 'All' column
    averages = data.groupby('Type')['All'].mean().round(5)
    
    # Add average annotations on top of the boxes
    y_min, y_max = plt.ylim()
    for i in range(4):
        average = averages[i]
        ax.text(i, y_max + 0.02, average, ha='center', va='center', weight='bold')

    plt.savefig(os.path.join(boxplot_folder, f'{dataset_name}.png'))
    plt.close()
    