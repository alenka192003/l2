import numpy as np
import matplotlib.pyplot as plt


# --- Шаг 1. Загрузка данных ---
def load_data(file_path, delimiter=','):
    """
    Загружаем данные из файла. Предполагаем, что данные в формате CSV.
    Последний столбец - целевая переменная (например, цена), остальные столбцы - признаки.
    """
    data = np.loadtxt(file_path, delimiter=delimiter)
    X = data[:, :-1]  # Признаки (количество передач, скорость оборота двигателя)
    y = data[:, -1]  # Целевая переменная (цена)
    return X, y


# --- Шаг 2. Нормализация данных ---
def normalize_features(X, y):
    """
    Нормализуем признаки и целевую переменную, чтобы улучшить сходимость градиентного спуска.
    Нормализация уменьшает масштабы чисел, что помогает алгоритму быстрее сходиться.
    """
    # Вычисляем средние значения и стандартные отклонения для нормализации
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)  # Среднее и стандартное отклонение для признаков
    y_mean, y_std = y.mean(), y.std()  # Среднее и стандартное отклонение для целевой переменной

    X_norm = (X - X_mean) / X_std
    y_norm = (y - y_mean) / y_std  # Аналогично для y
    return X_norm, y_norm, X_mean, X_std, y_mean, y_std


def add_bias_term(X):
    """
    Добавляем в матрицу признаков столбец единиц для учета свободного члена в модели линейной регрессии.
    """
    m = X.shape[0]  # Количество примеров
    return np.hstack([np.ones((m, 1)), X])  # Добавляем столбец единиц



# --- Шаг 3. Метод градиентного спуска ---
def gradient_descent(X, y, learning_rate, iterations):
    """
    Реализация градиентного спуска для минимизации функции стоимости.
    Обновление коэффициентов происходит на основе производных функции стоимости.

    Параметры:
    - X: Матрица признаков (с добавленным столбцом единиц)
    - y: Целевая переменная
    - learning_rate: Скорость обучения (насколько сильно мы изменяем коэффициенты)
    - iterations: Количество итераций (шагов) градиентного спуска

    Возвращаем:
    - theta: Вектор коэффициентов, найденных методом градиентного спуска
    - history: Список значений функции стоимости на каждой итерации
    """
    m, n = X.shape  # Количество примеров и количество признаков
    theta = np.zeros(n)  # Инициализация коэффициентов (включая свободный член)
    history = []  # Для отслеживания изменения функции стоимости

    for _ in range(iterations):
        # Прогнозируем значения y
        predictions = X @ theta
        errors = predictions - y  # Ошибка предсказания

        # Функция стоимости (среднеквадратическая ошибка)
        cost = (1 / (2 * m)) * np.sum(errors ** 2)
        history.append(cost)  # Добавляем стоимость в историю

        # Градиенты для каждого коэффициента
        gradients = (1 / m) * (X.T @ errors)

        # Обновляем коэффициенты с учетом скорости обучения
        theta -= learning_rate * gradients

    return theta, history


# --- Шаг 4. Аналитическое решение ---
def analytical_solution(X, y):
    """
    Аналитическое решение задачи линейной регрессии с использованием формулы нормальных уравнений.
    Это решение находит такие коэффициенты, которые минимизируют функцию стоимости.


    Параметры:
    - X: Матрица признаков
    - y: Целевая переменная

    Возвращаем:
    - theta: Вектор коэффициентов
    """
    return np.linalg.inv(X.T @ X) @ (X.T @ y)


# --- Шаг 5. Приведение коэффициентов к исходному масштабу ---
def unscale_theta(theta, X_mean, X_std, y_mean, y_std):
    """
    Преобразует коэффициенты, полученные на нормализованных данных, обратно к исходному масштабу.
    Это необходимо, чтобы интерпретировать их в контексте оригинальных данных (до нормализации).

    Параметры:
    - theta: Вектор коэффициентов, найденных методом градиентного спуска или аналитически
    - X_mean, X_std: Средние значения и стандартные отклонения признаков
    - y_mean, y_std: Среднее и стандартное отклонение целевой переменной

    Возвращаем:
    - theta_unscaled: Вектор коэффициентов в исходных единицах
    """
    theta_unscaled = np.zeros_like(theta)
    theta_unscaled[1:] = theta[1:] * y_std / X_std  # Преобразуем коэффициенты для признаков
    # Для свободного члена также учитываем масштабирование целевой переменной
    theta_unscaled[0] = theta[0] * y_std + y_mean - np.sum((theta[1:] * X_mean / X_std) * y_std)
    return theta_unscaled


# Функция для предсказания стоимости трактора
def predict_tractor_price(X_raw, theta_gd_unscaled):
    # Добавляем столбец единиц для предсказания
    X_with_bias = np.hstack([np.ones((X_raw.shape[0], 1)), X_raw])

    # Предсказываем стоимость
    predicted_price = X_with_bias @ theta_gd_unscaled

    return predicted_price


def analyze_tractors(X_raw, y_raw, theta_gd_unscaled):
    # Добавляем столбец единиц для предсказания
    X_with_bias = np.hstack([np.ones((X_raw.shape[0], 1)), X_raw])

    # Предсказываем стоимость
    predicted_prices = X_with_bias @ theta_gd_unscaled

    print("\nДетальный анализ тракторов:")
    for i in range(len(X_raw)):
        print(f"\nТрактор {i + 1}:")
        print(f"Количество передач: {X_raw[i][0]}")
        print(f"Скорость оборота двигателя: {X_raw[i][1]}")
        print(f"Реальная стоимость: {y_raw[i]:.2f} рублей")
        print(f"Предсказанная стоимость: {predicted_prices[i]:.2f} рублей")

        # Расчет погрешности
        error = abs(y_raw[i] - predicted_prices[i])
        error_percentage = (error / y_raw[i]) * 100
        print(f"Абсолютная погрешность: {error:.2f} рублей")
        print(f"Относительная погрешность: {error_percentage:.2f}%")




# --- Шаг 6. Визуализация сходимости функции стоимости ---
def plot_cost_history(learning_rates, X, y, iterations):
    """
    Визуализация сходимости функции стоимости для различных значений скорости обучения.
    Это поможет увидеть, как скорость обучения влияет на процесс оптимизации.
    """
    plt.figure(figsize=(10, 6))

    # Используем разные стили линий для лучшего различия
    line_styles = ['-', '--', '-.', ':']

    for i, lr in enumerate(learning_rates):
        _, cost_history = gradient_descent(X, y, lr, iterations)
        plt.plot(cost_history, label=f"Learning Rate: {lr}", linestyle=line_styles[i], linewidth=2)

    # Настройка отображения графика
    plt.xlabel("Итерации", fontsize=12)  # Подпись оси X
    plt.ylabel("Функция стоимости (ошибка)", fontsize=12)  # Подпись оси Y
    plt.title("Сходимость градиентного спуска", fontsize=14)  # Заголовок
    plt.legend(loc='upper right', fontsize=10)  # Легенда для графиков
    plt.grid(True)  # Включаем сетку для лучшей читаемости

    # Масштабируем ось X, чтобы отобразить меньшее количество итераций (например, только первые 200)
    plt.xlim(0, min(iterations, 200))  # Ограничиваем ось X, показывая только первые 200 итераций

    # Визуализация
    plt.show()


# --- Главная программа ---
if __name__ == "__main__":
    # Загрузка и нормализация данных
    X_raw, y_raw = load_data("ex1data2.txt")  # Загрузка данных из файла
    X_norm, y_norm, X_mean, X_std, y_mean, y_std = normalize_features(X_raw, y_raw)  # Нормализация
    X = add_bias_term(X_norm)  # Добавляем столбец единиц для свободного члена

    # Градиентный спуск
    learning_rates = [0.01, 0.03, 0.1, 0.3]  # Разные скорости обучения для эксперимента
    best_theta, best_cost, best_lr = None, float('inf'), None  # Инициализация для поиска лучших параметров
    iterations = 1000  # Количество итераций для градиентного спуска

    for lr in learning_rates:
        # Применяем градиентный спуск для каждой скорости обучения
        theta_gd, cost_history = gradient_descent(X, y_norm, lr, iterations)
        final_cost = cost_history[-1]
        if final_cost < best_cost:  # Выбираем наилучший результат
            best_theta, best_cost, best_lr = theta_gd, final_cost, lr

    # Аналитическое решение
    theta_analytical = analytical_solution(X, y_norm)  # Получаем коэффициенты аналитически

    # Приведение коэффициентов к исходному масштабу
    theta_gd_unscaled = unscale_theta(best_theta, X_mean, X_std, y_mean, y_std)
    theta_analytical_unscaled = unscale_theta(theta_analytical, X_mean, X_std, y_mean, y_std)

    # Предсказание стоимости
    predicted_prices = predict_tractor_price(X_raw, theta_gd_unscaled)


    # Вывод результатов
    print("Метод градиентного спуска:")
    print(f"Лучший коэффициент скорости обучения: {best_lr}")
    print(f"Коэффициенты: {theta_gd_unscaled}")
    print(f"Функция стоимости на последней итерации: {best_cost:.4f}")

    print("\nАналитическое решение:")
    print(f"Коэффициенты: {theta_analytical_unscaled}")

    analyze_tractors(X_raw, y_raw, theta_gd_unscaled)
    # Визуализация сходимости функции стоимости
    plot_cost_history(learning_rates, X, y_norm, iterations)


