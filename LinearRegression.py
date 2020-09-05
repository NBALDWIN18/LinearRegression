'''
Description: Shows undering of linear regression as a concept without the use of any libraries
IMPORTANT - Many of these LinAlg functions are not general and only apply to special matrix sizes (purely proof of concept)
Author: Nolan Baldwin
'''


'''
Dot product of 2 matrices
:param: m1 - matrix 1 [X:Y] matrix
:param: m2 - matrix 2 [Y:Z] matrix
:return: [X:Z] matrix
'''
def MatMult(m1, m2):
    res = []
    for i in range(len(m1)):
        res.append(list())
        for j in range(len(m2[0])):
            res[i].append(0)
            for k in range(len(m2)):
                res[i][j] += (m1[i][k] * m2[k][j])
    return res

'''
Subtracts m1 by m2
:param: m1 - [X:1] matrix 
:param: m2 - [X:1] matrix
:return: [X:1] matrix of m1 - m2
'''
def MatSub(m1,m2):
    res = []
    for i in range(len(m1)):
        res.append([m1[i][0] - m2[i][0]])
    return res

'''
Returns column j of a matrix as a row ([1:X] matrix)
:param: m - the input matrix
:param: j - the column number
return [1:X] matrix
'''
def ColumnToRow(m,j=0):
    r = list()
    for i in range(len(m)):
        r.append(m[i][j])
    return r

'''
Does element multiplication of m1*m2
:param: m1 - matrix 1
:param: m2 - matrix 2
:param: i - the column of m2 being used
:return: the result of the multiplication
'''
def EleMult(m1,m2,i):
    res = []
    for k in range(len(m1)):
        res.append(m1[k][0]*m2[k][i])
    return res

'''
Calculates the error
:param: X - X data
:param: Y - Y data
:param: theta - theta matrix
:return: Error between regression and actual (float)
'''
def Error(X, Y, theta):
    hypo = MatMult(X,theta)
    total = 0
    m = len(hypo)
    for i in range(m):
        total += ((hypo[i][0] - Y[i][0])**2)/(2*m)
    return total

'''
Uses gradient descent to adjust thetas for linear regression
:param: X - X data
:param: Y - Y data
:param: theta - theta matrix
:param: alpha - gradient descent step size
:param: num_iters - the number of iterations
:return: a list of error for each iteration (should descrease consistently if working) 
'''
def GradientDescent(X, Y, theta, alpha, num_iters):
    m = len(Y)
    error = []
    for i in range(num_iters):
        temps = []
        for j in range(len(theta)):
            temps.append((alpha/m) * sum(EleMult(MatSub(MatMult(X,theta), Y), X, j)))
        for t in range(len(temps)):
            theta[t][0] -= temps[t]
        error.append(Error(X,Y,theta))
    return error

theta = [[0],[0]]
X = []
Y = []
with open('./data.csv') as data:
    for line in data.readlines()[1:]:
        vals = line.split(',')
        X.append([1,float(vals[0])])
        Y.append([float(vals[1])])

print(GradientDescent(X,Y,theta,0.05,100))