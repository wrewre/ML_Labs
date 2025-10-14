import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\prana\OneDrive\Desktop\College\3rd year\5th sem\machine_learning\lab6\DCT_mal.csv")

# Drop label for correlation matrix
corr = df.drop(columns=['label']).corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap="coolwarm", cbar=True)
plt.title("Feature Correlation Heatmap")
plt.show()
