import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1,11,1000)
y = []

def function(x):
    y = (x ** 2) * (2.71828 ** (-x))
    return y

def derivative(x):
    y_d = (2.71828 ** (-x)) * (2 * x - x ** 2)
    return y_d

for pt_x in x:
    y.append(function(pt_x))



def gradientDescent(func,der, x0, n, epochs):
    x_newList = [x0]
    y_newList = [func(x0)]
    for epoch in range(0,epochs):
        y_newList.append(func(x_newList[epoch]))

        x_new = x_newList[epoch] - n * der(x_newList[epoch])
        x_newList.append(x_new)

    return [x_newList,y_newList]

algorithmResults = gradientDescent(function,derivative, 1.1, 0.01, 10000)

x_new = algorithmResults[0]
y_new = algorithmResults[1]

# print(len(x_new),len(y_new))


fig, ax = plt.subplots()
plt.xlim([-2, 10])
# plt.ylim(-1,8)

ax.plot(x,y,c='red')
ax.plot(x_new,y_new,c='green')
plt.show()