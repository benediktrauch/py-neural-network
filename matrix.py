# Sigmoid derivative helper function
def sigmoid_derivative(x):
    return x*(1-x)


# Sigmoid helper function
def sigmoid(x):
    temp = 1/(1+exp(-x))
    return temp


# Exponential function e^x
def exp(t):
    e = 2.71828182846
    return e**t


# Sigmoid for all elements in matrix
def sig(matrix):
    result = []

    for i in range(len(matrix)):
        result.append([])
        for j in range(len(matrix[0])):
            result[i].append(sigmoid(matrix[i][j]))
    return result


# Transpose any matrix
def transpose(matrix1):
    result = []

    if type(matrix1[0]) == int:
        for t in range(len(matrix1)):
            result.append([])

        for i in range(len(matrix1)):
            result[i].append(matrix1[i])
    else:
        for t in range(len(matrix1[0])):
            result.append([])

        for i in range(len(matrix1)):
            for j in range(len(matrix1[i])):
                result[j].append(matrix1[i][j])
    return result


# Sigmoid derivative
def derivative(matrix):
    result = []

    for i in range(len(matrix)):
        result.append([])
        for j in range(len(matrix[0])):
            result[i].append(sigmoid_derivative(matrix[i][j]))
    return result


# Subtract two matrices optimized for sin
def subtract_sin(matrix1, matrix2):
    result = []

    print len(matrix1), len(matrix1[0])
    print matrix1
    print len(matrix2), len(matrix2[0])
    print matrix2

    if len(matrix2) != len(matrix1):
        return "error"

    else:
        for i in range(len(matrix1)):
            for j in range(len(matrix1[0])):
                result.append(matrix1[i][j]-matrix2[i][j])
        return result


# Subtract two matrices (edited for XOR)
def subtract(matrix1, matrix2):
    result = []
    temp = []

    if len(matrix1) == 1:
        for _ in range(len(matrix2) - 1):
            matrix1.append(matrix1[0])

    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            result.append(matrix1[i][j] - matrix2[i][j])
    return result


# Add two matrices
def add(matrix1, matrix2):
    result = []

    if len(matrix2) != len(matrix1):
        return "error"

    else:
        for i in range(len(matrix1)):
            result.append([])
            for j in range(len(matrix1[0])):
                result[i].append(matrix1[i][j]+matrix2[i][j])

        return result


# Matrix Multiplication
def multiply(matrix1, matrix2):
    elements = []
    element = 0
    result = []

    # print len(matrix1), len(matrix1[0])
    # print matrix1
    # print len(matrix2), len(matrix2[0])
    # print matrix2

    if len(matrix2) != len(matrix1[0]):
        return "error"

    else:
        for i in range(len(matrix1)):
            result.append([])
            for j in range(len(matrix2[0])):
                for k in range(len(matrix2)):
                    elements.append(matrix1[i][k] * matrix2[k][j])

                for item in elements:
                    element = element + item

                result[i].append(element)

                element = 0
                elements = []

        return result
