import numpy as np
import matplotlib.pyplot as plt


# Функция сигмоиды
def sigmoid(z):
    z = np.clip(z, -500, 500)  # Предотвращение переполнения
    return 1 / (1 + np.exp(-z))


# Функция вычисления стоимости
def compute_cost(X, y, w):
    m = len(y)
    z = np.dot(X, w)
    predictions = sigmoid(z)
    epsilon = 1e-15  # Для избежания логарифма от 0
    cost = -np.mean(y * np.log(predictions + epsilon) + (1 - y) * np.log(1 - predictions + epsilon))
    return cost


# Функция вычисления градиента
def compute_gradient(X, y, w):
    m = len(y)
    z = np.dot(X, w)
    predictions = sigmoid(z)
    gradient = np.dot(X.T, (predictions - y)) / m
    return gradient


# Градиентный спуск
def gradient_descent(X, y, alpha, epochs):
    w = np.zeros(X.shape[1])  # Инициализация весов нулями
    for epoch in range(epochs):
        gradient = compute_gradient(X, y, w)
        w -= alpha * gradient
        if epoch % 1000 == 0 or epoch == epochs - 1:  # Лог каждые 1000 эпох
            cost = compute_cost(X, y, w)
            print(f"Epoch {epoch}, Cost: {cost:.4f}")
    return w


# Нормализация данных
def normalize(X):
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    return (X - min_val) / (max_val - min_val), min_val, max_val


# Нормализация новых данных
def normalize_new_data(new_data, min_val, max_val):
    return (new_data - min_val) / (max_val - min_val)


# Чтение данных из файла
data = np.loadtxt("ex2data1.txt", delimiter=",")
X = data[:, :2]
y = data[:, 2]

# Нормализация данных
X_normalized, min_val, max_val = normalize(X)
x1 = X_normalized[:, 0]
x2 = X_normalized[:, 1]

# Добавление нелинейных признаков
X_poly = np.column_stack([
    np.ones(X_normalized.shape[0]),
    x1,
    x2,
    x1 * x2,
    x1 ** 2,
    x2 ** 2
])

# Параметры обучения
alpha = 0.1
epochs = 30000

# Обучение модели
w = gradient_descent(X_poly, y, alpha, epochs)
print("Обученные веса:", w)

# Новая точка
new_data = np.array([[39.53833914367223, 76.03681085115882]])
new_data_normalized = normalize_new_data(new_data, min_val, max_val)
x1_new = new_data_normalized[:, 0]
x2_new = new_data_normalized[:, 1]

# Создание полиномиальных признаков для новой точки
new_data_poly = np.column_stack([
    np.ones(new_data_normalized.shape[0]),
    x1_new,
    x2_new,
    x1_new * x2_new,
    x1_new ** 2,
    x2_new ** 2
])

# Вычисление вероятности неисправности
probability = sigmoid(np.dot(new_data_poly, w))
print("Вероятность неисправности:", probability)

# Визуализация результатов
x1_vals = np.linspace(X_normalized[:, 0].min(), X_normalized[:, 0].max(), 100)
x2_vals = np.linspace(X_normalized[:, 1].min(), X_normalized[:, 1].max(), 100)
x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)

# Полиномиальные признаки для визуализации решающей границы
grid_poly = np.column_stack([
    np.ones(x1_grid.ravel().shape),
    x1_grid.ravel(),
    x2_grid.ravel(),
    x1_grid.ravel() * x2_grid.ravel(),
    x1_grid.ravel() ** 2,
    x2_grid.ravel() ** 2
])

z = sigmoid(np.dot(grid_poly, w)).reshape(x1_grid.shape)

# Построение контурного графика вероятности
plt.contourf(x1_grid, x2_grid, z, levels=50, cmap="coolwarm", alpha=0.8)
plt.colorbar(label="P(y=1)")
plt.scatter(X_normalized[y == 0][:, 0], X_normalized[y == 0][:, 1], color="blue", label="Class 0")
plt.scatter(X_normalized[y == 1][:, 0], X_normalized[y == 1][:, 1], color="red", label="Class 1")
plt.xlabel("Вибрация (нормализованная)")
plt.ylabel("Неравномерность вращения (нормализованная)")
plt.title("Вероятность неисправности двигателя")
plt.legend()
plt.show()

# Восстановление исходных значений для сетки
x1_vals_original = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
x2_vals_original = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
x1_grid_original, x2_grid_original = np.meshgrid(x1_vals_original, x2_vals_original)

# Нормализация сетки для расчетов
x1_grid_norm = (x1_grid_original - min_val[0]) / (max_val[0] - min_val[0])
x2_grid_norm = (x2_grid_original - min_val[1]) / (max_val[1] - min_val[1])

# Полиномиальные признаки для нормализованных точек сетки
grid_poly = np.column_stack([
    np.ones(x1_grid_norm.ravel().shape),
    x1_grid_norm.ravel(),
    x2_grid_norm.ravel(),
    x1_grid_norm.ravel() * x2_grid_norm.ravel(),
    x1_grid_norm.ravel() ** 2,
    x2_grid_norm.ravel() ** 2
])

# Вычисление z для разделяющей границы
z = sigmoid(np.dot(grid_poly, w)).reshape(x1_grid_original.shape)

# Построение графика
plt.figure(figsize=(8, 6))

# Рисуем разделяющую границу
plt.contour(x1_grid_original, x2_grid_original, z, levels=[0.5], colors='black', linewidths=2, linestyles='dashed')

# Наносим точки из исходного файла (ненормализованные данные)
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color="blue", label="Class 0")
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color="red", label="Class 1")

# Настройка осей и заголовков
plt.xlabel("Вибрация")
plt.ylabel("Неравномерность вращения")
plt.title("Разделяющая граница на исходных данных")
plt.legend()
plt.show()

# Построение разделяющей прямой
plt.figure(figsize=(8, 6))

# Создаем полиномиальные признаки только с линейными компонентами
X_linear = np.column_stack([
    np.ones(X_normalized.shape[0]),
    X_normalized[:, 0],  # x1
    X_normalized[:, 1]   # x2
])

# Градиентный спуск для линейной модели
w_linear = gradient_descent(X_linear, y, alpha, epochs)
print("Обученные веса для линейной модели:", w_linear)

# Построение разделяющей границы для линейной модели
x1_vals_original = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
x2_vals_original = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
x1_grid_original, x2_grid_original = np.meshgrid(x1_vals_original, x2_vals_original)

# Нормализация сетки для расчетов
x1_grid_norm = (x1_grid_original - min_val[0]) / (max_val[0] - min_val[0])
x2_grid_norm = (x2_grid_original - min_val[1]) / (max_val[1] - min_val[1])

# Полиномиальные признаки только для линейных данных
grid_linear = np.column_stack([
    np.ones(x1_grid_norm.ravel().shape),
    x1_grid_norm.ravel(),
    x2_grid_norm.ravel()
])

# Вычисляем z для линейной разделяющей границы
z_linear = sigmoid(np.dot(grid_linear, w_linear)).reshape(x1_grid_original.shape)

# Построение графика
plt.figure(figsize=(8, 6))

# Рисуем разделяющую границу
plt.contour(x1_grid_original, x2_grid_original, z_linear, levels=[0.5], colors='black', linewidths=2, linestyles='dashed')

# Наносим точки из исходного файла (ненормализованные данные)
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color="blue", label="Class 0")
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color="red", label="Class 1")

# Настройка осей и заголовков
plt.xlabel("Вибрация")
plt.ylabel("Неравномерность вращения")
plt.title("Линейная разделяющая граница на исходных данных")
plt.legend()
plt.show()

# Контур для вероятности P(y=1) = 0.5 (разделяющая линия)
plt.contour(x1_grid, x2_grid, z, levels=[0.5], colors='black', linewidths=2, linestyles='dashed')

# Нанесем исходные точки
plt.scatter(X_normalized[y == 0][:, 0], X_normalized[y == 0][:, 1], color="blue", label="Class 0")
plt.scatter(X_normalized[y == 1][:, 0], X_normalized[y == 1][:, 1], color="red", label="Class 1")

# Оформление графика
plt.xlabel("Вибрация (нормализованная)")
plt.ylabel("Неравномерность вращения (нормализованная)")
plt.title("Разделяющая граница для неисправности двигателя")
plt.legend()
plt.show()

# Функция воркера для ввода данных
def worker():
    print("Введите значения вибрации и неравномерности вращения:")
    vibration = float(input("Вибрация: "))
    unevenness = float(input("Неравномерность вращения: "))

    # Нормализация новых данных
    new_data = np.array([[vibration, unevenness]])
    new_data_normalized = normalize_new_data(new_data, min_val, max_val)
    x1_new = new_data_normalized[:, 0]
    x2_new = new_data_normalized[:, 1]

    # Создание полиномиальных признаков
    new_data_poly = np.column_stack([
        np.ones(new_data_normalized.shape[0]),
        x1_new,
        x2_new,
        x1_new * x2_new,
        x1_new ** 2,
        x2_new ** 2
    ])

    # Вычисление вероятности неисправности
    probability = sigmoid(np.dot(new_data_poly, w))
    print(f"Вероятность неисправности: {probability[0]:.4f}")


# Запуск воркера
worker()