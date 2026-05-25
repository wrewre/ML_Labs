from random import * # library used to generate random numbers
def statistical_calc(): # function definition
    arr=[] # used for storing the numbers
    mean=0
    median=0
    mode=0
    dictionary={} # stores key-value pairs of the most frequent elements
    most_frequent=0 
    for i in range(100):
        inp=randint(100,150) # used to generate an integer between 100 and 150 
        arr.append(inp)
    mean=sum(arr)/100 # calculation of mean
    arr.sort()
    if((len(arr)%2) == 0):
        median = (arr[(len(arr)//2)-1] + arr[len(arr)//2])/2  # calculation of mean
    else:
        median = arr[(len(arr)//2)]  # calculation of mean
    for j in arr:
        if j not in dictionary:
            dictionary[j]=1
        else:
            dictionary[j]+=1
    most_frequent = max(dictionary.values())
    for k in dictionary:
        if(dictionary[k] == most_frequent): #checks for the corresponding key of that particular value 
            mode=k  # calculation of mean
    return mean,median,mode

mean,median,mode=statistical_calc() # functionc call with a return type of int
print("The mean,median and mode of the given set of random numbers between 100 and 150 are: ",mean," ",median," and ",mode,"respectively")
