def matrix_multiplication(a,b):
    col1=len(a[0]) # len(column(matrix1))
    row2=len(b) # len(row(matrix2))
    if col1 != row2: # dimensions check
        print("Invalid matrices")
    for i in range(len(a)):
        for j in range(len(b[0])):
            summation=0
            for k in range(len(b)):
                summation+=a[i][k]*b[k][j] # multiplication of the corresponding row to its columns
            print(summation,end=' ')    
        print()


a=[[1,2,3],[4,5,6],[7,8,9]]
b=[[1,2,5],[2,2,1],[3,3,1]]
matrix_multiplication(a,b) # function call

