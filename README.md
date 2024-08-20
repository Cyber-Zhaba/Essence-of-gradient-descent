# Поиск коэффициентов функции по известным значениям


```python
import numpy as np
from scipy.optimize import minimize
from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
```

## Сущность градиентного спуска

Градиентный спуск - это метод оптимизации, который позволяет найти минимум функции. Он основан на идее, что если взять производную функции в точке, то она будет показывать направление наискорейшего роста функции. Следовательно, если взять производную и двигаться в противоположном направлении, то можно найти минимум функции. 

## Полином

Возьмём полином шестой степени вида $$f(x) = x \times (x - 1) \times (x - 1.5) \times (x - 3.5) \times (x - 4) \times (x - 4.5)$$

Предскажим эти коэффициенты с помощью градиентного спуска


```python
def f(x):
    return x * (x - 1) * (x - 1.5) * (x - 3.5) * (x - 4) * (x - 4.5)
```

Сгенерируем 200 точек и добавим к ним небольшой шум


```python
X = (np.random.rand(200) * 5. - 0.1)
X.sort()
Y = np.vectorize(f)(X) + (0.5 - np.random.rand(200)) * 7
# Render in svg
go.Figure(data=px.scatter(x=X, y=Y, title='Исходные данные'))
```



Объявим функцию потерь. Это сумма квадратов ошибок. Таким образом, найдя минимум этой функции, мы найдём коэффициенты изначального полинома


```python
def loss(coefs, *args):
    y_pred = np.polyval(coefs, X)
    return np.sum((Y - y_pred) ** 2)
```

Градиентный спуск


```python
pred = minimize(fun=loss, x0=np.zeros(7)).x; pred
```




    array([   1.0044308 ,  -14.47990737,   78.59571809, -197.03060961,
            222.70288328,  -90.11245856,   -0.49789156])



Оценка полученных коэффициентов


```python
scatter = go.Scatter(x=X, y=Y, mode='markers', name='Исходные данные')
line = go.Scatter(x=X, y=np.polyval(pred, X), mode="lines", name='Предсказание')

go.Figure(data=[scatter, line])
```


## Предсказание коэффициентов не полиномиальной функции

### Тайна египетских пирамид

Новый перспективный сервис Яндекс Пирамиды ищет стажера со знаниями в ML. В качестве тестового задания предлагается расшифровать найденную совсем недавно древнюю табличку, на которой записана формула, по предположению археологов, лежащая в основе архитектуры египетских пирамид. Возможно, происхождение этой таблички инопланетное!

Известно, что на табличке запечатлена следующая надпись:$$
f(x)=a⋅tg(x)+(b⋅sin(x)+c⋅cos(x))^ 2+d⋅sqrt(x)
$$

При этом часть таблички стерлась и значения коэффициентов $a,b,c,d$ утрачены. Однако известно, что коэффициенты $a,b,c,d$ неотрицательные, а на обратной стороне таблички были обнаружены некоторые известные значения$x$и$f(x)$для построенных пирамид.

У Вас есть прекрасная возможность помочь с решением данной головоломки, найдя неизвестные коэффициенты, и не только попасть на стажировку, но и поучаствовать в совершении прорыва в области археологии и уфологии.

#### Решение задачи в упрощённом виде


```python
def f(a, b, c, d, x):
    return a * np.tan(x) + (b * np.sin(x) + c * np.cos(x)) ** 2 + d * np.sqrt(x)
```


```python
X = np.linspace(0, 10, 100)
target = [np.random.randint(1, 100) + np.random.random() for _ in range(4)]
Y = f(*target, X)

print("Загаданные коэффициенты", target)
go.Figure(data=[go.Scatter(x=X, y=Y, name='Исходные данные', mode="markers")])
```

    Загаданные коэффициенты [73.45936868987789, 85.43404795659721, 59.40337172173047, 97.99520847747495]
    




```python
def loss(coefs, *args):
    y_pred = f(*coefs, X)
    return np.sum((Y - y_pred) ** 2)
```


```python
pred = minimize(fun=loss, x0=np.random.rand(4)).x; pred
```




    array([73.45936872, 85.43404794, 59.40337171, 97.99520915])




```python
scatter1 = go.Scatter(x=X, y=Y, mode='markers', name='Исходные данные')
scatter2 = go.Scatter(x=X, y=f(*pred, X), mode="markers", name='Предсказание')

go.Figure(data=[scatter1, scatter2])
```


На графике не видно исходных данных, так как они почти полностью совпадают с предсказанием

#### Решение задачи в полной виде

Формат ввода:
В первой строке задается целое число $n$ $(10 \leq n \leq 1000)$ - количество известных значений функции $f(x)$. Далее следует $n$ строк, в каждой из которых записана пара действительны чисел $x$ и $f(x)$, разделённые пробелом.

Формат вывода:
Выведите значения неизвестных коэффициентов $a,b,c,d$ в одной строке через пробел с точностью ровно 2 знака после запятой.


```python
def solution_to_the_mystery_of_the_egyptian_pyramids():
    import numpy as np
    from scipy.optimize import minimize
    
    
    def f(a, b, c, d, x):
        return a * np.tan(x) + (b * np.sin(x) + c * np.cos(x)) ** 2 + d * np.sqrt(x)
    
    
    n = int(input())
    
    X = np.zeros(n, dtype=float)
    Y = np.zeros(n, dtype=float)
    
    for i in range(n):
        X[i], Y[i] = map(float, input().split())
        
    
    def loss(coefs, *args):
        y_pred = f(*coefs, X)
        return np.sum((Y - y_pred) ** 2)
    
    
    pred = minimize(fun=loss, x0=np.random.rand(4)).x
    print(" ".join(map(lambda x: "{:.2f}".format(x), pred)))
```
