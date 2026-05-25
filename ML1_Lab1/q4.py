def matrix_transpose(matrix): # function
    for i in range(0,len(matrix)):
        for j in range(i+1,len(matrix)):
            matrix[i][j],matrix[j][i] = matrix[j][i],matrix[i][j] #swapping elements in i and j positions 
    return matrix

matrix=[[1,2,3],[4,5,6],[7,8,9]]
final=matrix_transpose(matrix) # function call with a return type of list
print("The transpose of the matrix is:")
for i in range(len(final)):
    for j in range(len(final[0])):
        print(final[i][j],end=' ') #displaying the output
    print()

