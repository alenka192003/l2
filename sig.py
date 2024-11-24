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

def tanh(x):
    """
    Вычисляет значение гиперболического тангенса
    Используем встроенную функцию numpy для точности
    """
    return np.tanh(x)

def tanh_derivative(x):
    """
    Вычисляет производную гиперболического тангенса
    Формула: 1 - tanh^2(x)
    """
    return 1 - np.tanh(x)**2

# 1. Вычисление значений сигмоиды в заданных точках
x_points = np.array([0, 3, -3, 8, -8, 15, -15])
sigmoid_values = sigmoid(x_points)

print("Значения сигмоиды в заданных точках:")
for x, y in zip(x_points, sigmoid_values):
    print(f"x = {x:3d}, sigmoid(x) = {y:.15f}")

# Построение графиков
plt.figure(figsize=(15, 10))

# График сигмоиды
plt.subplot(2, 2, 1)
x = np.linspace(-15, 15, 1000)
plt.plot(x, sigmoid(x), 'b-', label='Сигмоида')
plt.scatter(x_points, sigmoid(x_points), color='red', label='Заданные точки')
plt.grid(True)
plt.legend()
plt.title('График сигмоидной функции')
plt.xlabel('x')
plt.ylabel('y')

# График гиперболического тангенса
plt.subplot(2, 2, 2)
plt.plot(x, tanh(x), 'g-', label='tanh(x)')
plt.grid(True)
plt.legend()
plt.title('График гиперболического тангенса')
plt.xlabel('x')
plt.ylabel('y')

# График производной сигмоиды
plt.subplot(2, 2, 3)
plt.plot(x, sigmoid_derivative(x), 'r-', label='Производная сигмоиды')
plt.grid(True)
plt.legend()
plt.title('График производной сигмоиды')
plt.xlabel('x')
plt.ylabel('y')

# График производной tanh
plt.subplot(2, 2, 4)
plt.plot(x, tanh_derivative(x), 'm-', label='Производная tanh')
plt.grid(True)
plt.legend()
plt.title('График производной tanh')
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