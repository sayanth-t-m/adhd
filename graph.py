import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Simulate data for ADHD, Dyslexia, and Control groups
n_samples = 100  # Number of samples per group

# Saccade Frequency (higher for ADHD)
saccade_freq_adhd = np.random.normal(10, 2, n_samples)  # Mean=10, SD=2
saccade_freq_dyslexia = np.random.normal(7, 1.5, n_samples)  # Mean=7, SD=1.5
saccade_freq_control = np.random.normal(6, 1, n_samples)  # Mean=6, SD=1

# Fixation Duration (longer for Dyslexia)
fixation_duration_adhd = np.random.normal(200, 30, n_samples)  # Mean=200ms, SD=30
fixation_duration_dyslexia = np.random.normal(350, 50, n_samples)  # Mean=350ms, SD=50
fixation_duration_control = np.random.normal(250, 20, n_samples)  # Mean=250ms, SD=20

# Create DataFrame
data = pd.DataFrame({
    'Group': ['ADHD'] * n_samples + ['Dyslexia'] * n_samples + ['Control'] * n_samples,
    'SaccadeFrequency': np.concatenate([saccade_freq_adhd, saccade_freq_dyslexia, saccade_freq_control]),
    'FixationDuration': np.concatenate([fixation_duration_adhd, fixation_duration_dyslexia, fixation_duration_control])
})

# Set seaborn style
sns.set(style="whitegrid")

# 1. Box Plots for Saccade Frequency and Fixation Duration
plt.figure(figsize=(12, 5))

# Box Plot for Saccade Frequency
plt.subplot(1, 2, 1)
sns.boxplot(x='Group', y='SaccadeFrequency', data=data, palette="muted")
plt.title('Saccade Frequency by Group')
plt.xlabel('Group')
plt.ylabel('Saccade Frequency (saccades/second)')

# Box Plot for Fixation Duration
plt.subplot(1, 2, 2)
sns.boxplot(x='Group', y='FixationDuration', data=data, palette="muted")
plt.title('Fixation Duration by Group')
plt.xlabel('Group')
plt.ylabel('Fixation Duration (ms)')

plt.tight_layout()
plt.show()

# 2. Scatter Plot: Saccade Frequency vs Fixation Duration
plt.figure(figsize=(8, 6))
sns.scatterplot(x='SaccadeFrequency', y='FixationDuration', hue='Group', data=data, palette="deep")
plt.title('Saccade Frequency vs Fixation Duration')
plt.xlabel('Saccade Frequency (saccades/second)')
plt.ylabel('Fixation Duration (ms)')
plt.legend(title='Group')
plt.show()