import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def student_index(index):
    return index % 100, int(index/1000)


def main():
    index = 180893
    D = 0.00001
    h = 0.1  # skok x
    k = h ** 2 / (2 * D)  # skok t
    data = [[student_index(index)[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, student_index(index)[1]]]
    j = 0
    while(j<100):
        temp = [student_index(index)[0]]
        for i in range(1, 10):
            temp.append((data[j][i + 1] + data[j][i - 1])/2)

        temp.append(student_index(index)[1])
        data.append(temp)
        j += 1


    ax = plt.axes(projection='3d')
    X, Y = np.meshgrid(list(x for x in range(11)), list(range(j+1)))
    data = np.array(data)
    np.savetxt('dane', data, fmt = '%10.5f')
    print(np.shape(data))
    print(np.shape(X))
    print(np.shape(Y))
    ax.scatter(X, Y, data)
    #print(np.array(data, ndmin = 2))
    #plt.plot(range(11), data[100])
    plt.show()


if __name__ == '__main__':
    main()
