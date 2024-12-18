import math

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """
    Вычисляет значение сигмоидной функции для заданного x
    Формула: 1 / (1 + e^(-x))
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Вычисляет производную сигмоидной функции
    Формула: sigmoid(x) * (1 - sigmoid(x))
    """
    sig = sigmoid(x)
    return sig * (1 - sig)

def sinh(x):
    """
    Гиперболический синус: sinh(x) = (e^x - e^(-x)) / 2
    Работает с одиночными числами или массивами.
    """
    if isinstance(x, np.ndarray):
        return np.array([(math.exp(xi) - math.exp(-xi)) / 2 for xi in x])
    else:
        return (math.exp(x) - math.exp(-x)) / 2

def cosh(x):
    """
    Гиперболический косинус: cosh(x) = (e^x + e^(-x)) / 2
    Работает с одиночными числами или массивами.
    """
    if isinstance(x, np.ndarray):
        return np.array([(math.exp(xi) + math.exp(-xi)) / 2 for xi in x])
    else:
        return (math.exp(x) + math.exp(-x)) / 2

def tanh(x):
    """
    Гиперболический тангенс: tanh(x) = sinh(x) / cosh(x)
    Работает с одиночными числами или массивами.
    """
    return sinh(x) / cosh(x)

def ctg(x):
    """
    Гиперболический котангенс: ctg(x) = cosh(x) / sinh(x)
    Работает с одиночными числами или массивами.
    """
    # Обработка вырожденных случаев
    if isinstance(x, np.ndarray):
        with np.errstate(divide='ignore', invalid='ignore'):
            return cosh(x) / sinh(x)
    else:
        try:
            return cosh(x) / sinh(x)
        except ZeroDivisionError:
            return float('inf')

def tanh_derivative(x):
    """
    Производная гиперболического тангенса: 1 - tanh^2(x)
    """
    t = tanh(x)
    return 1 - t ** 2

# 1. Вычисление значений сигмоиды в заданных точках
x_points = np.array([0, 3, -3, 8, -8, 15, -15])
sigmoid_values = sigmoid(x_points)

print("Значения сигмоиды в заданных точках:")
for x, y in zip(x_points, sigmoid_values):
    print(f"x = {x:3d}, sigmoid(x) = {y:.15f}")


def tanh_scaled(x):
    """
    Гиперболический тангенс в диапазоне [0, 1]:
    tanh_scaled(x) = (tanh(x) + 1) / 2
    """
    return (tanh(x) + 1) / 2
# Построение графиков
plt.figure(figsize=(20, 15))

# График сигмоиды
plt.subplot(2, 3, 1)
x = np.linspace(-15, 15, 1000)
plt.plot(x, sigmoid(x), 'b-', label='Сигмоида')
plt.scatter(x_points, sigmoid(x_points), color='red', label='Заданные точки')
plt.grid(True)
plt.legend()
plt.title('График сигмоидной функции')
plt.xlabel('x')
plt.ylabel('y')

# График гиперболического тангенса
plt.figure(figsize=(10, 5))
plt.plot(x, tanh(x), 'g-', label='Оригинальный tanh(x)')
plt.plot(x, tanh_scaled(x), 'b-', label='Масштабированный tanh_scaled(x)')
plt.grid(True)
plt.legend()
plt.title('График гиперболического тангенса и его масштабированной версии')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# График производной сигмоиды
plt.subplot(2, 3, 3)
plt.plot(x, sigmoid_derivative(x), 'r-', label='Производная сигмоиды')
plt.grid(True)
plt.legend()
plt.title('График производной сигмоиды')
plt.xlabel('x')
plt.ylabel('y')

# График производной tanh
plt.subplot(2, 3, 4)
plt.plot(x, tanh_derivative(x), 'm-', label='Производная tanh')
plt.grid(True)
plt.legend()
plt.title('График производной tanh')
plt.xlabel('x')
plt.ylabel('y')

# График гиперболического синуса
plt.subplot(2, 3, 5)
plt.plot(x, sinh(x), 'c-', label='sinh(x)')
plt.grid(True)
plt.legend()
plt.title('График гиперболического синуса')
plt.xlabel('x')
plt.ylabel('y')

# График гиперболического косинуса
plt.subplot(2, 3, 6)
plt.plot(x, cosh(x), 'orange', label='cosh(x)')
plt.grid(True)
plt.legend()
plt.title('График гиперболического косинуса')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()

# Вычисление производных в некоторых точках для демонстрации
test_points = np.array([0, 1, -1, 2, -2])
print("\nЗначения производной сигмоиды в тестовых точках:")
for x in test_points:
    print(f"x = {x:2d}, sigmoid'(x) = {sigmoid_derivative(x):.15f}")

print("\nЗначения производной tanh в тестовых точках:")
for x in test_points:
    print(f"x = {x:2d}, tanh'(x) = {tanh_derivative(x):.15f}")