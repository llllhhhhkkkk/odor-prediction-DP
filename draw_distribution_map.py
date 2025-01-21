import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('./curated_GS_LF_merged_4983.csv')

odor_types = data.columns[2:]
odor_counts_specific = data[odor_types].sum().sort_values(ascending=False)
odor_counts_specific_1 = odor_counts_specific[:50]
odor_counts_specific_2 = odor_counts_specific[50:100]
odor_counts_specific_3 = odor_counts_specific[100:]

y_max = 2000


plt.figure(figsize=(15, 8))
odor_counts_specific_1.plot(kind='bar')
plt.ylim(0, y_max)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('./odor_distribution1.png')

plt.figure(figsize=(15, 8))
odor_counts_specific_2.plot(kind='bar')
plt.ylim(0, y_max)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('./odor_distribution2.png')

plt.figure(figsize=(15, 8))
odor_counts_specific_3.plot(kind='bar')
plt.ylim(0, y_max)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('./odor_distribution3.png')

plt.figure(figsize=(15, 22))
odor_counts_specific.plot(kind='barh')
plt.title('Frequency Distribution of Odors Type')
plt.xlabel('Frequency')
plt.ylabel('Odor Type')
plt.xticks()
plt.tight_layout()
plt.savefig('./odor_distribution.png')
