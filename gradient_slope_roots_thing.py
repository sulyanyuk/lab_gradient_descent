import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

x = np.linspace(-1, 11, 1000)
y = []

def function(x):
    y = (x ** 2) * (2.71828 ** (-x))
    return y

def derivative(x):
    y_d = (2.71828 ** (-x)) * (2 * x - x ** 2)
    return y_d

for pt_x in x:
    y.append(function(pt_x))

def gradientDescent(func, der, x0, n, epochs):
    x_newList = [x0]
    y_newList = [func(x0)]
    for epoch in range(0, epochs):
        y_newList.append(func(x_newList[epoch]))
        x_new = x_newList[epoch] - n * der(x_newList[epoch])
        x_newList.append(x_new)
    return [x_newList, y_newList]

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.35)
plt.xlim([-2, 10])

ax.plot(x, y, c='red')
line, = ax.plot([], [], c='green')

slider_x0_ax = plt.axes([0.25, 0.25, 0.65, 0.03])
slider_n_ax = plt.axes([0.25, 0.15, 0.65, 0.03])
slider_epochs_ax = plt.axes([0.25, 0.05, 0.65, 0.03])

slider_x0 = Slider(slider_x0_ax, 'x0', -2.0, 10.0, valinit=1.1,valstep=0.01)
slider_n = Slider(slider_n_ax, 'n', 0.001, 0.1, valinit=0.01,valstep=0.001)
slider_epochs = Slider(slider_epochs_ax, 'epochs', 1, 2000, valinit=200,valstep=1)

def update(val):
    x0 = slider_x0.val
    n = slider_n.val
    epochs = slider_epochs.val
    results = gradientDescent(function, derivative, x0, n, epochs)
    line.set_data(results[0], results[1])
    fig.canvas.draw_idle()

slider_x0.on_changed(update)
slider_n.on_changed(update)
slider_epochs.on_changed(update)

update(None)

plt.show()
