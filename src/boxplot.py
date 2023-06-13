import os
import matplotlib.pyplot as plt
import seaborn as sns

def create_boxplots(path, dataset, df):
    # create new folder to save boxplots
    os.makedirs(path, exist_ok=True)

    ax = sns.boxplot(x='Type', y='mCPF', data=df, order=['N', 'C', 'P', 'CP'])

    # Calculate the average per 'Type' for the 'All' column
    averages = df.groupby('Type')['All'].mean().round(5)

    plt.xlabel('Type', fontsize=15, weight='bold')
    plt.ylabel('mCPF', fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Add average annotations on top of the boxes
    y_min, y_max = plt.ylim()
    for i in range(4):
        average = averages[i]
        ax.text(i, y_max + 0.03, average, ha='center', va='center', weight='bold', size=15)

    plt.savefig(os.path.join(path, f'{dataset}.png'), bbox_inches='tight')
    plt.close()