import pandas as pd 
import numpy as np 

# loading the dataset
df = pd.read_csv(r"C:\Users\prana\OneDrive\Desktop\machine_learning\lab3\DCT_mal.csv")
def std_mean_dist(path):
    df = pd.read_csv(path)#Calculating the standard deviation of each class
    sd = df.groupby("LABEL").std()
    # Calculating the mean of each class
    mean = df.groupby("LABEL").mean()
    labels=mean.index.tolist() # a list of all the labels in the dataset
    cent1=mean.loc[labels[0]].values
    cent2=mean.loc[labels[1]].values
    distance=np.linalg.norm(cent1-cent2)
    return sd,mean,distance


path = r"C:\Users\prana\OneDrive\Desktop\machine_learning\lab3\DCT_mal.csv"
sd,mean,distance = std_mean_dist(path)
print("\nThe standard deviation of each Class is:\n",sd)
print("The mean of each Class is:\n",mean)
print("The distance between the classes are:",distance)


r"C:\Users\prana\OneDrive\Desktop\machine_learning\lab3\DCT_mal.csv"