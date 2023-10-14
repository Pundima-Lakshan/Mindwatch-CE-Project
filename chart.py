import pandas as pd
import matplotlib.pyplot as plt

# Load the direction data from the CSV file
df = pd.read_csv('direction_data.csv')

# Create a binary column for "Looking Forward"
df['Looking Forward'] = (df['Direction'] == 'Forward').astype(int)

# Create a line chart
plt.figure(figsize=(10, 6))
plt.plot(df['Duration (s)'], df['Looking Forward'], color='blue', marker='o', linestyle='-')
plt.xlabel('Duration (s)')
plt.ylabel('Looking Forward (1) / Not Looking Forward (0)')
plt.title('Looking Forward vs. Not Looking Forward')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
