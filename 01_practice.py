import numpy as np
from numpy import ndarray
import matplotlib.pylab as plt


# https://ai-trend.jp/basic-study/neural-network/num_diff/
# 中心差分近似でf'(x)を求める
def numerical_diff(f, x: float) -> float:
    h = 1e-4
    return (f(x + h) - f(x - h)) / 2 * h


def plot_graph(x: ndarray, y: ndarray):
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, y)
    plt.show()


# y= 0.01x^2 + 0.1x
def function_1(x) -> float:
    return 0.01 * x ** 2 + 0.1 * x


def function_1_summary():
    x: ndarray = np.arange(0.0, 20.0, 0.1)
    y = function_1(x)
    plot_graph(x, y)
    # ここら辺のx'の式にxを入れた値のことを'真の微分'というらしい
    # 約0.2
    print(numerical_diff(function_1, 5))
    # 約0.3
    print(numerical_diff(function_1, 10))


# f(x_0, x_1) = x_0^2 + x_1^2
def function_2(x):
    # 下記のようにも書ける
    # return np.sum(x**2)
    return x[0] ** 2 + x[1] ** 2


# x_0 = 3, x_1=4の時のx_0に対する偏微分∂f/∂x_0
# ∂は\partialと書く
# 偏微分は1変数の微分と同じで、ある場所の傾きを求める
# 複数ある変数の中でターゲットを一つに絞り、他の変数の値は固定する。
def function_tmp1(x0):
    return x0 * x0 + 4.0 ** 2


# 全ての変数の偏微分をベクトルとしてまとめたものをgradient(勾配)という
# 勾配は各地点において低くなる方向を指す
def numerical_gradient(f, x):
    # 0.0001
    h: float = 1e-4
    gradient = np.zeros_like(x)
    # 一つずつ計算している。すご。
    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        gradient[idx] = (fxh1 - fxh2) / (2 * h)
        # 値を元に戻す
        x[idx] = tmp_val
    return gradient


# 勾配降下
# 勾配法の数式x=x - n * (∂f) / (∂x)
# nはlearning rate. 学習率のようなパラメータはハイパーパラメータともいう
def gradient_descent(f, init_x, learning_rate=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        gradient = numerical_gradient(f, x)
        x -= learning_rate * gradient

    return x


if __name__ == '__main__':
    init_x = np.array([-3.0, 4.0])
    # なんかうごかん
    gradient_descent(function_tmp1, init_x, 0.1)
