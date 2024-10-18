import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def selection(df, key):
    count = df[key].value_counts()
    return count.index, count.values

def describe_rus(df, key):
    stats = df[key].describe()
    translated_stats = {
        "count": "Количество",
        "mean": "Среднее значение",
        "std": "Стандартное отклонение",
        "min": "Минимум",
        "25%": "25%",
        "50%": "Медиана",
        "75%": "75%",
        "max": "Максимум",
        "unique": "unique",
        "top": "наиболее часто встречающееся значение ",
        "freq": "Частота"
    }

    stats.index = stats.index.map(translated_stats.get)

    print(stats, '\n')

def draw(plt, df, key: str, labels: tuple, position):
    plt.subplot(position[0], position[1], position[2])
    i, v = selection(df, key)

    plt.bar(i, v, color='skyblue', edgecolor='black')
    plt.title(key, fontsize=10, fontweight='bold')
    plt.xlabel(labels[0], fontsize=8)
    plt.ylabel(labels[1], fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.xticks(fontsize=8, rotation=45, ha='right')
    plt.yticks(fontsize=8)

    describe_rus(df, key)


def approximate(plt, df, x_features: list, y_feature: str, frac):
    print(f"независимые признаки:  {x_features}")
    print(f"зависимый признак:  {y_feature}")

    length = int(len(df[x_features[0]]) * (1 - frac))
    a_raw = []

    for i in range(length):
        a_raw.append([1] + [
            0 if df[x][i] == 'No' else 1 if df[x][i] == 'Yes' else df[x][i]
            for x in x_features
        ])

    a = np.array(a_raw)
    y = np.array(df[y_feature][:length])
    at_a_inv_at = np.linalg.inv(np.matmul(a.T, a)).dot(a.T)
    ratios = np.matmul(at_a_inv_at, y)

    print('коэффициенты: ', ratios)

    # Если два признака, то рисуем график
    if len(ratios) == 2:
        plt.figure()
        for i in range(100):
            plt.scatter(df[x_features[0]][i], y[i], color='dodgerblue', alpha=0.7)
        x = np.array([df[x_features[0]].min(), df[x_features[0]].max()])
        y1 = ratios[0] + ratios[1] * x
        plt.plot(x, y1, color='red', linewidth=2)

        plt.title(f'Регрессионная линия: {x_features[0]} vs {y_feature}', fontsize=10, fontweight='bold')
        plt.xlabel(x_features[0], fontsize=8)
        plt.ylabel(y_feature, fontsize=8)
        plt.grid(True)
        plt.savefig('regression.png')
        # plt.show()

    return ratios


def determinant(df, ratios: list, x_features: list, y_feature: str, frac):
    length = len(df)
    pos = int(length * frac)
    sum1, sum2 = 0, 0
    y = df[y_feature]

    for i in range(pos):
        y_predicted = ratios[0] + sum(
            ratios[j + 1] * (0 if df[x_features[j]][length - pos + j] == 'No' else
                             1 if df[x_features[j]][length - pos + j] == 'Yes' else
                             df[x_features[j]][length - pos + j])
            for j in range(len(x_features))
        )
        sum1 += (y[i] - y_predicted) ** 2
        sum2 += (y[i] - y.mean()) ** 2

    return 1 - sum1 / sum2


if __name__ == '__main__':
    df = pd.read_csv("Student_Performance.csv", sep=',')
    # # Удаляет все строки, где есть хотя бы одно NaN
    # data = df.dropna()
    plt.figure(1)
    plt.subplots_adjust(hspace=0.5, wspace=0.3)

    # Построение гистограмм по ключевым признакам
    draw(plt, df, 'Hours Studied', ('hours', 'кол-во'), (2, 3, 1))
    draw(plt, df, 'Previous Scores', ('scores', 'кол-во'), (2, 3, 2))
    draw(plt, df, 'Extracurricular Activities', ('activity', 'кол-во'), (2, 3, 3))
    draw(plt, df, 'Sleep Hours', ('hours', 'кол-во'), (2, 3, 4))
    draw(plt, df, 'Sample Question Papers Practiced', ('papers', 'кол-во'), (2, 3, 5))
    draw(plt, df, 'Performance Index', ('index', 'кол-во'), (2, 3, 6))

    plt.savefig('my_plot.png')
    # plt.show()

# Аппроксимация по разным наборам признаков
model1 = approximate(plt, df, ['Extracurricular Activities', 'Sleep Hours'], 'Performance Index', 0.8)
model2 = approximate(plt, df, ['Hours Studied', 'Previous Scores', 'Extracurricular Activities',
                                'Sleep Hours', 'Sample Question Papers Practiced'], 'Performance Index', 0.8)
model3 = approximate(plt, df, ['Hours Studied'], 'Performance Index', 0.8)

# Расчёт коэффициента детерминации
r1 = determinant(df, model1, ['Extracurricular Activities', 'Sleep Hours'], 'Performance Index', 0.2)

r2 = determinant(df, model2, ['Hours Studied', 'Previous Scores', 'Extracurricular Activities','Sleep Hours', 'Sample Question Papers Practiced'], 'Performance Index',
                 0.2)
r3 = determinant(df, model3, ['Hours Studied'], 'Performance Index', 0.2)

# Создание синтетической переменной (Эта переменная, Effort_Index, представляет собой сумму количества часов, которые студент потратил на учебу, и количества практических заданий (вопросов), которые он прорешал. Она отражает общий уровень усилий, приложенных к обучению, и может быть полезной для объяснения производительности (Performance Index).)
df['Effort_Index'] = df['Hours Studied'] + df['Sample Question Papers Practiced']

r4 = determinant(df, model1, ['Extracurricular Activities', 'Effort_Index'], 'Performance Index',
                 0.2)

print('R1:', r1)
print('R2:', r2)
print('R3:', r3)
print('R4:', r4)
