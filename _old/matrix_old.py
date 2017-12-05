import random
import time


def sigmoidDeriv(x):
    return x*(1-x)


def sigmoid(x):
    sig = 1/(1+exp(-x))
    return sig


def exp(t):
    e = 2.71828182846
    return e**t


def matrixExp(matrix1):
    result = matrix1
    e = 2.71828182846

    for i in range(len(result)):
        for j in range(len(result[i])):
            result[i][j] = e**(result[i][j])
    return result


def transposeMatrix(matrix1):
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


def sigMatrix(matrix):
    result = []

    for i in range(len(matrix)):
        result.append([])
        for j in range(len(matrix[0])):
            result[i].append(sigmoid(matrix[i][j]))
    return result


def derivMatrix(matrix):
    result = []

    for i in range(len(matrix)):
        result.append([])
        for j in range(len(matrix[0])):
            result[i].append(sigmoidDeriv(matrix[i][j]))
    return result


def sigMultiplyMatrix(matrix1, matrix2):
    layer = []
    sig = []
    l_elements = []
    l_element = 0
    s_elements = []
    s_element = 0

    if len(matrix2) != len(matrix1[0]):
        return ("error")

    else:
        for i in range(len(matrix1)):
            layer.append([])
            sig.append([])
            for j in range(len(matrix2[0])):
                for k in range(len(matrix2)):
                    temp = sigmoid(matrix1[i][k] * matrix2[k][j])
                    s_elements.append(temp)
                    l_elements.append(matrix1[i][k] * matrix2[k][j])

                for item in s_elements:
                    s_element = s_element + item
                for item in l_elements:
                    l_element = l_element + item

                sig[i].append(s_element)
                layer[i].append(l_element)

                s_elements = []
                s_element = 0

                l_elements = []
                l_element = 0

        # print result
        return [sig, layer]


def derivMultiplyMatrix(matrix1, matrix2):
    elements = []
    element = 0
    result = []

    # print len(matrix1), len(matrix1[0])
    # print matrix1
    # print len(matrix2), len(matrix2[0])
    # print matrix2

    if len(matrix2) != len(matrix1[0]):
        return ("error")

    else:
        for i in range(len(matrix1)):
            result.append([])
            for j in range(len(matrix2[0])):
                for k in range(len(matrix2)):
                    elements.append(sigmoid(matrix1[i][k] * matrix2[k][j]))

                for item in elements:
                    element = element + item

                result[i].append(element)

                element = 0
                elements = []

        # print result
        return (result)


def matrixVecSubtract(matrix1, matrix2):
    result = []
    elements = []
    element = 0

    if len(matrix2) != len(matrix1):
        return ("error")

    else:
        for i in range(len(matrix1)):
            for j in range(len(matrix1[0])):
                result.append(matrix1[i][j]-matrix2[i][j])
        return result

def matrixAdd(matrix1, matrix2):
    result = []
    elements = []
    element = 0

    if len(matrix2) != len(matrix1):
        return ("error")

    else:
        for i in range(len(matrix1)):
            result.append([])
            for j in range(len(matrix1[0])):
                result[i].append(matrix1[i][j]+matrix2[i][j])

        return result


def deltaHidden(matrix1, delta2, weights):
    result = []

    delta = delta2[:]

    for i in range(len(matrix1)-1):
        for j in range(len(matrix1[0])):
            temp = multiplyMatrix(delta, transposeMatrix(weights))
            print temp
            result.append(multiplyMatrix(matrix1, temp))

    print result
    return result


def multiplyMatrix(matrix1, matrix2):
    elements = []
    element = 0
    result = []

    if len(matrix2) != len(matrix1[0]):
        return ("error")

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

        # print result
        return result


def createMatrix(rows, collumns, dataList):
    matrix = []
    for i in range(rows):
        rowList = []
        for j in range(collumns):
            rowList.append(dataList[collumns * i + j])
        matrix.append(rowList)
    return matrix


def main():
    x = 4
    y = 5
    z = 20

    dataList1 = []
    dataList2 = []

    for i in range(x*y):
        dataList1.append(random.uniform(-1, 1))  # uniform
    for i in range(y*z):
        dataList2.append(random.uniform(-1, 1))  # randint

    matrix1 = createMatrix(x, y, dataList1)
    matrix2 = createMatrix(y, z, dataList2)

    result = multiplyMatrix(matrix1, matrix2)


# start_time = time.time()
# main()
# print("--- %s seconds ---" % (time.time() - start_time))
