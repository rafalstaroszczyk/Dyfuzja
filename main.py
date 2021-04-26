import numpy as np
import matplotlib.pyplot as plt


def student_index(index):
    return index % 100, int(index/1000)


def teoretyczne(t, T1, T2, D):
    nmax=1000
    u = np.zeros(11)
    x = np.linspace(0, 1, 11)
    for n in range(1, nmax + 1):
        if n % 2 == 0:
            u = u + 1/n * np.exp(- n**2 * np.pi**2 * D * t) * np.sin(n * np.pi * x)
        else:
            u = u - 1/n * np.exp(- n**2 * np.pi**2 * D * t) * np.sin(n * np.pi * x)

    u = u * 2 * (T2 - T1) / np.pi
    u += (T2 - T1) * x + T1
    return u


def main():
    tolerance = 10**-4
    index = 180893
    D = 0.00001
    h = 0.1  # skok x
    k = h ** 2 / (2 * D)  # skok t
    index1 = student_index(index)[0]
    index2 = student_index(index)[1]
    #data = [[student_index(index)[0], 1, 1, 1, 1, 1, 1, 1, 1, 1, student_index(index)[1]]]
    data = [[index1 for i in range(10)] + [index2]]
    j = 0
    error = 1
    while(error > tolerance):
        temp = [index1]
        for i in range(1, 10):
            temp.append((data[j][i + 1] + data[j][i - 1])/2)

        temp.append(index2)
        data.append(temp)
        j += 1
        error = 0
        for i in range(11):
            error += abs((data[j][i]-data[j-1][i]) / data[j][i])
        error /= 11


    ax = plt.axes(projection='3d')
    X, Y = np.meshgrid(list(x for x in range(11)), list(range(j+1)))
    X = X * h
    Y = Y * k
    np.savetxt('dane', data, fmt = '%10.5f')
    ax.scatter(X, Y, data)
    teor = np.zeros((11, j+1))
    print(np.shape(teor[:, 1]))
    for i in range(j+1):
        teor[:, i] = teoretyczne(i * k, index1, index2, D)
    teor = np.transpose(teor)
    ax.scatter(X, Y, teor)
    plt.show()


if __name__ == '__main__':
    main()
