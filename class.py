import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, \
    recall_score
from sklearn.pipeline import make_pipeline


# Загрузка данных
def load_data(filename):
    """
    Загрузка данных из текстового файла

    Args:
        filename (str): Путь к файлу с данными

    Returns:
        tuple: Массивы признаков X и меток y
    """
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, :-1]  # Признаки: вибрация и неравномерность вращения
    y = data[:, -1]  # Метки класса (0 - исправен, 1 - неисправен)
    return X, y


# Визуализация данных
def plot_data(X, y, title='Распределение данных'):
    """
    Визуализация данных с разделением на классы

    Args:
        X (np.array): Признаки
        y (np.array): Метки классов
        title (str): Заголовок графика
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Исправен')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Неисправен')
    plt.xlabel('Вибрация')
    plt.ylabel('Неравномерность вращения')
    plt.title(title)
    plt.legend()
    plt.show()


# Линейная логистическая регрессия
def linear_logistic_regression(X, y):
    """
    Обучение и оценка линейной логистической регрессии

    Args:
        X (np.array): Признаки
        y (np.array): Метки классов

    Returns:
        dict: Результаты обучения модели
    """
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Линейная логистическая регрессия
    model = LogisticRegression(random_state=42, max_iter=5000)
    model.fit(X_train, y_train)

    # Предсказание и оценка
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Вывод подробного отчета о классификации
    print("\nПодробный отчет о классификации (Линейная логистическая регрессия):")
    print(classification_report(y_test, y_pred, target_names=['Исправен', 'Неисправен']))

    print("\nМатрица ошибок (Confusion Matrix):")
    print(conf_matrix)
    print("\nТолкование матрицы ошибок:")
    print(f"Истинно положительные (TP): {conf_matrix[1, 1]} - правильно определенные неисправные двигатели")
    print(f"Истинно отрицательные (TN): {conf_matrix[0, 0]} - правильно определенные исправные двигатели")
    print(f"Ложно положительные (FP): {conf_matrix[0, 1]} - исправные двигатели, определенные как неисправные")
    print(f"Ложно отрицательные (FN): {conf_matrix[1, 0]} - неисправные двигатели, определенные как исправные")

    return {
        'model': model,
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': conf_matrix,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


# Нелинейная логистическая регрессия с полиномиальными признаками
def polynomial_logistic_regression(X, y, degree=2):
    """
    Обучение и оценка нелинейной логистической регрессии с полиномиальными признаками

    Args:
        X (np.array): Признаки
        y (np.array): Метки классов
        degree (int): Степень полиномиальных признаков

    Returns:
        dict: Результаты обучения модели
    """
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Нелинейная логистическая регрессия с полиномиальными признаками
    poly_model = make_pipeline(
        PolynomialFeatures(degree=degree),
        LogisticRegression(random_state=42)
    )
    poly_model.fit(X_train, y_train)

    # Предсказание и оценка
    y_pred = poly_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Вывод подробного отчета о классификации
    print("\nПодробный отчет о классификации (Нелинейная логистическая регрессия):")
    print(classification_report(y_test, y_pred, target_names=['Исправен', 'Неисправен']))

    print("\nМатрица ошибок (Confusion Matrix):")
    print(conf_matrix)
    print("\nТолкование матрицы ошибок:")
    print(f"Истинно положительные (TP): {conf_matrix[1, 1]} - правильно определенные неисправные двигатели")
    print(f"Истинно отрицательные (TN): {conf_matrix[0, 0]} - правильно определенные исправные двигатели")
    print(f"Ложно положительные (FP): {conf_matrix[0, 1]} - исправные двигатели, определенные как неисправные")
    print(f"Ложно отрицательные (FN): {conf_matrix[1, 0]} - неисправные двигатели, определенные как исправные")

    return {
        'model': poly_model,
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': conf_matrix,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


# Визуализация границ решения
def plot_decision_boundary(X, y, model, title='Граница решения'):
    """
    Визуализация границы решения модели

    Args:
        X (np.array): Признаки
        y (np.array): Метки классов
        model (object): Обученная модель
        title (str): Заголовок графика
    """
    plt.figure(figsize=(10, 6))

    # Создание сетки точек для визуализации
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # Прогнозирование для каждой точки сетки
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Визуализация границы решения с помощью plt.plot
    for x_val in range(Z.shape[0] - 1):
        for y_val in range(Z.shape[1] - 1):
            if Z[x_val, y_val] != Z[x_val, y_val + 1] or Z[x_val, y_val] != Z[x_val + 1, y_val]:
                # Рисуем линию на границе между классами
                color = 'green' if Z[x_val, y_val] == 0 else 'orange'
                plt.plot([xx[x_val, y_val], xx[x_val, y_val + 1]],
                         [yy[x_val, y_val], yy[x_val, y_val + 1]],
                         color=color, linewidth=2)

    # Отрисовка точек данных
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Исправен')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Неисправен')

    plt.xlabel('Вибрация')
    plt.ylabel('Неравномерность вращения')
    plt.title(title)
    plt.legend()
    plt.show()


# Основная функция
def main():
    # Загрузка данных
    X, y = load_data('ex2data1.txt')

    # Визуализация исходных данных
    plot_data(X, y, 'Распределение данных двигателей')

    # Линейная логистическая регрессия
    linear_result = linear_logistic_regression(X, y)
    print("\nЛинейная логистическая регрессия:")
    print(f"Точность: {linear_result['accuracy'] * 100:.2f}%")
    print(f"F1-score: {linear_result['f1_score']:.2f}")
    print(f"Точность (Precision): {linear_result['precision']:.2f}")
    print(f"Полнота (Recall): {linear_result['recall']:.2f}")
    plot_decision_boundary(X, y, linear_result['model'], 'Линейная граница решения')

    # Нелинейная логистическая регрессия
    poly_result = polynomial_logistic_regression(X, y, degree=2)
    print("\nНелинейная логистическая регрессия (полиномиальные признаки):")
    print(f"Точность: {poly_result['accuracy'] * 100:.2f}%")
    print(f"F1-score: {poly_result['f1_score']:.2f}")
    print(f"Точность (Precision): {poly_result['precision']:.2f}")
    print(f"Полнота (Recall): {poly_result['recall']:.2f}")
    plot_decision_boundary(X, y, poly_result['model'], 'Нелинейная граница решения')


if __name__ == '__main__':
    main()