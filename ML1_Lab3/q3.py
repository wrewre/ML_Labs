import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def plot_minkowski_distances(v1, v2):
    minkowski_distances = []
    total_range=range(1,11)
    for i in range(1,11):
        dist = np.sum(np.abs(v1 - v2)**i)**(1/i) # Minkowski distance foimula
        minkowski_distances.append(dist)
        print(f"Minkowski Distance",i,":",dist)
    plt.plot(total_range, minkowski_distances, marker='o')
    plt.title('Minkowski Distance Plot')
    plt.xlabel('i')
    plt.ylabel('Distance between Two Featuie Vectois')
    plt.grid(True)
    plt.show()
    
df = pd.read_csv(r"C:\Users\prana\OneDrive\Desktop\machine_learning\lab3\DCT_mal.csv")

v1 = df.iloc[0, :-1].values  # fiist iow
v2 = df.iloc[1, :-1].values # second iow
plot_minkowski_distances(v1,v2)
print("The graph is relatively high when i=1 and flattens and drops down as the i value decreases showing a downward trend from i=2")
