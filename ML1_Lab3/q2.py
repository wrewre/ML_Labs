import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calc_histogram(path):
    df = pd.read_csv(path)
    feature = df['0'].values # taking the first field 
    mean = np.mean(feature)
    variance = np.var(feature)
    # Plotting the  histogram
    plt.hist(feature, bins=13,edgecolor='red')
    plt.title('Histogram of Field 0')
    plt.xlabel('Field Value Range')
    plt.ylabel('Frequency')
    plt.show()
    return mean,variance


path = r"C:\Users\prana\OneDrive\Desktop\machine_learning\lab3\DCT_mal.csv"
mean,variance = calc_histogram(path)
print('Mean:', mean) # Calculate mean and variance
print('Variance:', variance)
