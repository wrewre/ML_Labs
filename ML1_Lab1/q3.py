input1=eval(input("Enter the first list:"))
input2=eval(input("Enter the second list:"))
def common_occurence(l1,l2):
    matrix1=[]
    matrix2=[]
    common=[]
    for i in range(len(l1)):
        if l1[i] not in matrix1: # condition to remove duplicates 
            matrix1.append(l1[i])
    for j in range(len(l2)): 
        if l2[j] not in matrix2: # condition to remove duplicates
            matrix2.append(l2[j])
    print(matrix1)
    print(matrix2)
    for k in range(len(matrix1)): 
        for p in range(len(matrix2)):
            if matrix1[k] == matrix2[p]: # condition to check the common elements
                common.append(matrix1[k])
    
    return common 
final=common_occurence(input1,input2) #function call with a return type of list
print("The common elements are:",end=" ")
for i in range(len(final)):
    print(final[i],end=' ') 
