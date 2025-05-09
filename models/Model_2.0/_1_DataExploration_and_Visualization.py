import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Load your metadata
df = pd.read_csv("floor_plan_metadata_extended.csv")

# Basic statistics of room counts and areas
print(df.filter(like='Count_').describe())
print(df.filter(like='_SqFt').describe())

# Visualize room count distributions
plt.figure(figsize=(15, 10))
count_cols = [col for col in df.columns if col.startswith('Count_')]
n = len(count_cols)
plots_per_fig = 12

for batch_start in range(0, n, plots_per_fig):
    batch_cols = count_cols[batch_start:batch_start+plots_per_fig]
    num_plots = len(batch_cols)
    rows = math.ceil(num_plots / 4)
    plt.figure(figsize=(15, 5 * rows))
    for i, col in enumerate(batch_cols):
        plt.subplot(rows, 4, i + 1)
        sns.countplot(x=col, data=df)
        plt.title(col.replace('Count_', ''))
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'room_count_distributions_{batch_start // plots_per_fig + 1}.png')
    plt.close()

# Visualize room area distributions by plot size
for plot_size in df['PlotSize'].unique():
    plt.figure(figsize=(15, 10))
    plot_data = df[df['PlotSize'] == plot_size]
    area_cols = [col for col in df.columns if '_SqFt' in col and 'Total' not in col][
                :12]  # Limit to first 12 room types

    for i, col in enumerate(area_cols):
        if i >= 12:  # Limit to 12 subplots per figure
            break
        plt.subplot(3, 4, i + 1)
        sns.histplot(plot_data[col].dropna(), kde=True)
        plt.title(f"{col} - {plot_size}")
        plt.xlabel('Square Feet')

    plt.tight_layout()
    plt.savefig(f'{plot_size}_room_areas.png')

# Room size relationships (e.g., Bedroom vs Bathroom sizes)
plt.figure(figsize=(15, 10))
if 'Bedroom1_SqFt' in df.columns and 'Bathroom1_SqFt' in df.columns:
    sns.scatterplot(x='Bedroom1_SqFt', y='Bathroom1_SqFt',
                    hue='PlotSize', data=df)
    plt.title('Bedroom vs. Bathroom Size Relationship')
    plt.savefig('bedroom_bathroom_relationship.png')