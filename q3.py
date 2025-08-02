import numpy as np
import matplotlib.pyplot as plt
def plotting(X,Y,colors):
    plt.scatter(X, Y, c=colors)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Randomly Generated Training Data")
    plt.show()
X = np.random.uniform(1, 10, 20)
Y = np.random.uniform(1, 10, 20)
random = np.random.randint(0, 2, 20)
colors=[]
for i in random:
    if i == 1:
        colors.append("blue")
    else:
        colors.append("red") 
plotting(X,Y,colors)

