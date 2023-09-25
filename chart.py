import pandas as pd
import matplotlib.pyplot as plt

# Load the direction data from the CSV file
df = pd.read_csv('direction_data.csv')

# Filter the data to only include "Forward" direction
forward_data = df[df['Direction'] == 'Forward']

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(forward_data['Duration (s)'], height=1, width=1, color='blue')
plt.xlabel('Duration (s)')
plt.ylabel('Count')
plt.title('Duration of Forward Direction')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
