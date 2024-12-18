import numpy as np
import matplotlib.pyplot as plt


# --- Шаг 1. Загрузка данных ---
def load_data(file_path, delimiter=','):
    """
    Загружаем данные из файла.

    Размерности:
    - data: (m, n+1) - матрица данных, где m - число примеров, n - число признаков
    - X: (m, n) - матрица признаков
    - y: (m,) - вектор целевой переменной
    """
    data = np.loadtxt(file_path, delimiter=delimiter)
    X = data[:, :-1]  # Признаки (m x n)
    y = data[:, -1]  # Целевая переменная (m,)
    return X, y


# --- Шаг 2. Нормализация данных ---
def normalize_features(X, y):
    """
    Нормализуем признаки и целевую переменную.

    Входные размерности:
    - X: (m, n) - матрица признаков
    - y: (m,) - вектор целевой переменной

    Выходные размерности:
    - X_norm: (m, n) - нормализованная матрица признаков
    - y_norm: (m,) - нормализованный вектор целевой переменной
    - X_mean: (n,) - вектор средних значений признаков
    - X_std: (n,) - вектор стандартных отклонений признаков
    - y_mean: скаляр - среднее значение целевой переменной
    - y_std: скаляр - стандартное отклонение целевой переменной
    """
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)  # (n,), (n,)
    y_mean, y_std = y.mean(), y.std()  # скаляры

    X_norm = (X - X_mean) / X_std  # (m, n)
    y_norm = (y - y_mean) / y_std  # (m,)
    return X_norm, y_norm, X_mean, X_std, y_mean, y_std


def add_bias_term(X):
    """
    Добавляем столбец единиц для учета свободного члена.

    Входная размерность:
    - X: (m, n) - матрица признаков

    Выходная размерность:
    - X_with_bias: (m, n+1) - матрица признаков с добавленным столбцом единиц
    """
    m = X.shape[0]  # Количество примеров
    return np.hstack([np.ones((m, 1)), X])  # (m, n+1)


# --- Шаг 3. Метод градиентного спуска ---
def gradient_descent(X, y, learning_rate, iterations):
    """
    Реализация градиентного спуска.

    Входные размерности:
    - X: (m, n+1) - матрица признаков с добавленным столбцом единиц
    - y: (m,) - вектор целевой переменной

    Выходные размерности:
    - theta: (n+1,) - вектор коэффициентов
    - history: (iterations,) - история значений функции стоимости
    """
    m, n = X.shape  # Количество примеров и признаков
    theta = np.zeros(n)  # (n+1,) - вектор коэффициентов
    history = []  # История функции стоимости

    for _ in range(iterations):
        predictions = X @ theta  # (m,) - предсказанные значения
        errors = predictions - y  # (m,) - ошибки предсказания

        # Функция стоимости
        cost = (1 / (2 * m)) * np.sum(errors ** 2)  # скаляр
        history.append(cost)

        # Градиенты: (n+1,)
        gradients = (1 / m) * (X.T @ errors)

        # Обновляем коэффициенты
        theta -= learning_rate * gradients

    return theta, history


# --- Шаг 4. Аналитическое решение ---
def analytical_solution(X, y):
    """
    Аналитическое решение задачи линейной регрессии.

    Входные размерности:
    - X: (m, n+1) - матрица признаков с добавленным столбцом единиц
    - y: (m,) - вектор целевой переменной

    Выходная размерность:
    - theta: (n+1,) - вектор коэффициентов
    """
    #np.linalg.inv. Получает обратную матрицу данной. 3
    return np.linalg.inv(X.T @ X) @ (X.T @ y)  # (n+1,)


# --- Шаг 5. Приведение коэффициентов к исходному масштабу ---
def unscale_theta(theta, X_mean, X_std, y_mean, y_std):
    """
    Преобразует коэффициенты к исходному масштабу.

    Входные размерности:
    - theta: (n+1,) - вектор коэффициентов
    - X_mean: (n,) - вектор средних значений признаков
    - X_std: (n,) - вектор стандартных отклонений признаков
    - y_mean: скаляр
    - y_std: скаляр

    Выходная размерность:
    - theta_unscaled: (n+1,) - вектор коэффициентов в исходном масштабе
    """
    theta_unscaled = np.zeros_like(theta)
    theta_unscaled[1:] = theta[1:] * y_std / X_std  # (n,)
    # Свободный член
    theta_unscaled[0] = theta[0] * y_std + y_mean - np.sum((theta[1:] * X_mean / X_std) * y_std)
    return theta_unscaled


def predict_tractor_price(X_raw, theta_gd_unscaled):
    """
    Предсказание стоимости трактора.

    Входные размерности:
    - X_raw: (m, n) - исходная матрица признаков
    - theta_gd_unscaled: (n+1,) - вектор коэффициентов

    Выходная размерность:
    - predicted_price: (m,) - вектор предсказанных цен
    """
    # Добавляем столбец единиц для предсказания
    X_with_bias = np.hstack([np.ones((X_raw.shape[0], 1)), X_raw])  # (m, n+1)

    # Предсказываем стоимость
    predicted_price = X_with_bias @ theta_gd_unscaled  # (m,)

    return predicted_price


def analyze_tractors(X_raw, y_raw, theta_gd_unscaled):
    # Добавляем столбец единиц для предсказания
    X_with_bias = np.hstack([np.ones((X_raw.shape[0], 1)), X_raw])

    # Предсказываем стоимость
    predicted_prices = X_with_bias @ theta_gd_unscaled

    print("\nДетальный анализ тракторов:")
    for i in range(len(X_raw)):
        print(f"\nТрактор {i + 1}:")
        print(f"Количество передач: {X_raw[i][1]}")
        print(f"Скорость оборота двигателя: {X_raw[i][0]}")
        print(f"Реальная стоимость: {y_raw[i]:.2f} рублей")
        print(f"Предсказанная стоимость: {predicted_prices[i]:.2f} рублей")

        # Расчет погрешности
        error = abs(y_raw[i] - predicted_prices[i])
        error_percentage = (error / y_raw[i]) * 100
        print(f"Абсолютная погрешность: {error:.2f} рублей")
        print(f"Относительная погрешность: {error_percentage:.2f}%")

def normalize_features_2(X, y):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    # Если стандартное отклонение равно нулю, установим его значение на 1 (или другое значение, подходящее для вашей задачи)
    X_std = np.where(X_std == 0, 1, X_std)
    X_norm = (X - X_mean) / X_std

    y_mean = np.mean(y)
    y_std = np.std(y)
    # То же самое для y
    y_std = 1 if y_std == 0 else y_std
    y_norm = (y - y_mean) / y_std

    return X_norm, X_mean, X_std, y_norm, y_mean, y_std  # Возвращаем все значения

def compare_methods(X_raw, y_raw, X_norm, y_norm, X_mean, X_std, y_mean, y_std, best_theta, theta_analytical):
    """
    Сравнение результатов градиентного спуска и аналитического решения
    """
    # Приводим коэффициенты к исходному масштабу
    theta_gd_unscaled = unscale_theta(best_theta, X_mean, X_std, y_mean, y_std)
    theta_analytical_unscaled = unscale_theta(theta_analytical, X_mean, X_std, y_mean, y_std)

    # Добавляем столбец единиц для предсказания
    X_with_bias = np.hstack([np.ones((X_raw.shape[0], 1)), X_raw])

    # Предсказываем значения обоими методами
    predictions_gd = X_with_bias @ theta_gd_unscaled
    predictions_analytical = X_with_bias @ theta_analytical_unscaled

    # Рассчитываем метрики качества
    mse_gd = np.mean((y_raw - predictions_gd) ** 2)
    mse_analytical = np.mean((y_raw - predictions_analytical) ** 2)

    mae_gd = np.mean(np.abs(y_raw - predictions_gd))
    mae_analytical = np.mean(np.abs(y_raw - predictions_analytical))

    print("\nСравнительный анализ методов:")
    print("\n1. Коэффициенты моделей:")
    print(f"Градиентный спуск:     {theta_gd_unscaled}")
    print(f"Аналитическое решение: {theta_analytical_unscaled}")
    print(f"\nРазница в коэффициентах: {np.abs(theta_gd_unscaled - theta_analytical_unscaled)}")

    print("\n2. Метрики качества:")
    print(f"MSE градиентного спуска: {mse_gd:.2f}")
    print(f"MSE аналитического решения: {mse_analytical:.2f}")
    print(f"MAE градиентного спуска: {mae_gd:.2f}")
    print(f"MAE аналитического решения: {mae_analytical:.2f}")

    print("\n3. Сравнение предсказаний:")
    print("\nПример предсказаний для первых 5 наблюдений:")
    for i in range(min(5, len(y_raw))):
        print(f"\nНаблюдение {i + 1}:")
        print(f"Реальное значение: {y_raw[i]:.2f}")
        print(f"Предсказание (градиентный спуск): {predictions_gd[i]:.2f}")
        print(f"Предсказание (аналитическое): {predictions_analytical[i]:.2f}")
        print(f"Разница между методами: {np.abs(theta_gd_unscaled - theta_analytical_unscaled)}")


def evaluate_learning_rate(learning_rate, X, y, iterations):
    """
    Оценка качества коэффициента скорости обучения
    """
    theta, cost_history = gradient_descent(X, y, learning_rate, iterations)

    # Критерии оценки:
    final_cost = cost_history[-1]  # Конечное значение функции стоимости

    # Оценка скорости сходимости
    convergence_threshold = 1e-6
    convergence_iteration = 0
    for i in range(1, len(cost_history)):
        if abs(cost_history[i] - cost_history[i - 1]) < convergence_threshold:
            convergence_iteration = i
            break

    # Оценка стабильности
    oscillations = 0
    for i in range(2, len(cost_history)):
        if (cost_history[i - 1] - cost_history[i - 2]) * (cost_history[i] - cost_history[i - 1]) < 0:
            oscillations += 1

    return {
        'final_cost': final_cost,
        'convergence_speed': convergence_iteration,
        'stability': oscillations,
        'theta': theta,
        'cost_history': cost_history  # Добавляем историю стоимости в возвращаемый словарь
    }


def find_best_learning_rate(learning_rates, X, y, iterations):
    """
    Находит наилучший коэффициент скорости обучения
    """
    results = {}
    for lr in learning_rates:
        results[lr] = evaluate_learning_rate(lr, X, y, iterations)

    # Нормализация метрик
    min_cost = min(r['final_cost'] for r in results.values())
    max_cost = max(r['final_cost'] for r in results.values())
    min_conv = min(r['convergence_speed'] for r in results.values())
    max_conv = max(r['convergence_speed'] for r in results.values())


    # Вычисление общего показателя качества
    best_lr = None
    best_score = float('-inf')
    best_theta = None

    print("\nАнализ коэффициентов скорости обучения:")
    for lr, metrics in results.items():
        # Нормализованные показатели
        cost_score = 1 - (metrics['final_cost'] - min_cost) / (max_cost - min_cost + 1e-10)
        conv_score = 1 - (metrics['convergence_speed'] - min_conv) / (max_conv - min_conv + 1e-10)

        # Общий показатель (с весами)
        total_score = 0.5 * cost_score + 0.5 * conv_score

        print(f"\nLearning rate: {lr}")
        print(f"Изменение ошибки по итерациям:")
        cost_history = metrics['cost_history']
        for i, cost in enumerate(cost_history):
            if i % 100 == 0 or i == len(cost_history) - 1:  # Выводим каждые 100 итераций и последнюю
                print(f"Итерация {i}: cost = {cost:.6f}")
        print(f"Конечная ошибка: {metrics['final_cost']:.6f}")
        print(f"Скорость сходимости (итераций): {metrics['convergence_speed']}")
        print(f"Общий показатель качества: {total_score:.4f}")

        if total_score > best_score:
            best_score = total_score
            best_lr = lr
            best_theta = metrics['theta']

    print(f"\nНаилучший коэффициент скорости обучения: {best_lr}")
    return best_lr, best_theta



# --- Шаг 6. Визуализация сходимости функции стоимости ---
def plot_cost_history(learning_rates, X, y, iterations):
    """
    Визуализация сходимости функции стоимости для различных значений скорости обучения.
    Показывает, как скорость обучения влияет на процесс оптимизации.
    """
    plt.figure(figsize=(15, 8))  # Увеличиваем размер графика

    # Используем разные стили и цвета линий для лучшего различия
    line_styles = ['-', '--', '-.', ':']
    colors = ['blue', 'green', 'red', 'purple']

    for i, lr in enumerate(learning_rates):
        _, cost_history = gradient_descent(X, y, lr, iterations)
        plt.plot(cost_history,
                 label=f"Learning Rate: {lr}",
                 linestyle=line_styles[i % len(line_styles)],  # Циклический доступ
                 color=colors[i % len(colors)],  # Циклический доступ
                 linewidth=2)

    # Настройка отображения графика
    plt.xlabel("Итерации", fontsize=12)
    plt.ylabel("Функция стоимости (ошибка)", fontsize=12)
    plt.title("Сходимость градиентного спуска для разных коэффициентов обучения", fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True)

    # Ограничиваем ось X, показывая только первые 200-300 итераций
    plt.xlim(0, min(iterations, 150))
    plt.yscale('log')  # Логарифмическая шкала для лучшей визуализации

    plt.tight_layout()
    plt.show()


#с альфой доделать в отчете уюбрать скорость сходимости итераций находим мин и все

def plot_individual_cost_histories(learning_rates, X, y, iterations):
    """
    Создаем отдельные графики сходимости для каждого коэффициента обучения
    """
    plt.figure(figsize=(15, 12))  # Большой размер для субграфиков

    # Создаем сетку субграфиков 2x2
    for i, lr in enumerate(learning_rates, 1):
        plt.subplot(2, 2, i)

        # Выполняем градиентный спуск
        theta, cost_history = gradient_descent(X, y, lr, iterations)

        # Строим график для текущего коэффициента обучения
        plt.plot(cost_history, label=f"LR: {lr}", color='blue')
        plt.title(f"Learning Rate: {lr}")
        plt.xlabel("Итерации")
        plt.ylabel("Функция стоимости")
        plt.yscale('log')  # Логарифмическая шкала
        plt.xlim(0, min(iterations, 300))
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()

def get_input_and_predict(X_raw, y_raw, X_mean, X_std, y_mean, y_std, best_theta):
    """
    Функция для получения данных от пользователя (скорость оборотов и количество передач),
    предсказания стоимости трактора и вывода результатов.
    """
    # Получаем данные от пользователя
    speed = float(input("Введите скорость оборота двигателя (например, 1500): "))
    gears = int(input("Введите количество передач (например, 6): "))

    # Создаем массив признаков для введенных данных
    new_data = np.array([[speed, gears]])

    # Нормализуем эти данные
    new_data_norm, _, _, _, _, _ = normalize_features_2(new_data, np.array([0]))  # Если функция возвращает 6 значений

    # Добавляем столбец единиц для свободного члена
    new_data_norm_with_bias = add_bias_term(new_data_norm)

    # Приводим коэффициенты градиентного спуска к исходному масштабу
    theta_gd_unscaled = unscale_theta(best_theta, X_mean, X_std, y_mean, y_std)

    # Прогнозируем стоимость
    predicted_price = predict_tractor_price(new_data, theta_gd_unscaled)

    print(f"\nПредсказанная стоимость трактора: {predicted_price[0]:.2f} рублей")

# --- Главная программа ---
if __name__ == "__main__":
    X_raw, y_raw = load_data("ex1data2.txt")  # Загрузка данных из файла
    X_norm, y_norm, X_mean, X_std, y_mean, y_std = normalize_features(X_raw, y_raw)  # Нормализация
    X = add_bias_term(X_norm)  # Добавляем столбец единиц для свободного члена

    # Градиентный спуск
    learning_rates = [0.3, 0.5, 1, 1.1]  # Разные скорости обучения для эксперимента
    #learning_rates = [0.5, 1,1.5,5]  # Разные скорости обучения для эксперимента

    iterations = 2000  # Количество итераций для градиентного спуска

    #iterations = 5000  # Количество итераций для градиентного спуска

    # Находим лучший learning rate и соответствующие theta
    best_lr, best_theta = find_best_learning_rate(learning_rates, X, y_norm, iterations)

    #best_lr, best_theta = find_best_learning_rate(learning_rates2, X, y_norm, iterations)

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

    print("\nАналитическое решение:")
    print(f"Коэффициенты: {theta_analytical_unscaled}")


    analyze_tractors(X_raw, y_raw, theta_gd_unscaled)
    # Визуализация сходимости функции стоимости
    plot_cost_history(learning_rates, X, y_norm, iterations)
    plot_individual_cost_histories(learning_rates,X,y_norm,iterations)
    compare_methods(X_raw, y_raw, X_norm, y_norm, X_mean, X_std, y_mean, y_std, best_theta, theta_analytical)

# Вызываем функцию для получения данных от пользователя и предсказания стоимости
    get_input_and_predict(X_raw, y_raw, X_mean, X_std, y_mean, y_std, best_theta)
