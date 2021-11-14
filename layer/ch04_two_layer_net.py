from typing import Dict

import numpy as np
from numpy.random import randn


# mnistの手書きの数字データを予測するモデル
class TwoLayerNet:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, weight_init_std=0.01):
        """
        :param input_size: 入力層のニューロン数
        :param hidden_size: 隠れ層のニューロン数
        :param output_size: 出力層のニューロン数
        :param weight_init_std: 重み係数の初期値
        Wは重み
        bはバイアス

        重みパラメータの初期化は大切らしい
        今回はガウス分布西違う乱数での初期化
        """
        self.params: Dict = {
            'W1': weight_init_std * randn(input_size, hidden_size),
            'W2': weight_init_std * randn(input_size, hidden_size),
            'b1': np.zeros(hidden_size),
            'b2': np.zeros(output_size)
        }

    # TODO: あとで基礎的なlayerの処理は親クラスにまとめて、このクラスが継承するようにする
    # あらゆる入力値を0-1に正規化する
    # https://atmarkit.itmedia.co.jp/ait/articles/2003/04/news021.html
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # 複数の出力値の合計が1.0になるように変換する関数。0-1の範囲を出力する
    # https://atmarkit.itmedia.co.jp/ait/articles/2004/08/news016.html
    def softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)  # オーバーフロー対策
        return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

    def sigmoid_grad(self, x):
        return (1.0 - self.sigmoid(x)) * self.sigmoid(x)

    # 交差エントロピー誤差
    def cross_entropy_error(self, y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
        if t.size == y.size:
            t = t.argmax(axis=1)

        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: 画像データ
        :return:
        """
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = self.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = self.softmax(a2)
        return y

    def loss(self, x, t):
        """
        :param x: 画像データ
        :param t: 正解ラベル
        :return:
        """
        y = self.predict(x)
        return self.cross_entropy_error(y, t)

    # 正解率
    # https://data.gunosy.io/entry/2016/08/05/115345
    def accuracy(self, x: np.ndarray, t: np.ndarray) -> float:
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """
        :param x: 入力データ
        :param t: 教師データ
        :return:
        wは重みの勾配
        bはバイアスの勾配
        各重みパラメータの勾配を求める
        再起的に処理をしているのはなぜだろう・・・
        """
        loss_w = lambda w: self.loss(x, t)
        grads = {'W1': self.numerical_gradient(loss_w, self.params['W1']),
                 'b1': self.numerical_gradient(loss_w, self.params['b1']),
                 'W2': self.numerical_gradient(loss_w, self.params['W2']),
                 'b2': self.numerical_gradient(loss_w, self.params['b2'])}
        return grads

    def gradient(self, x, t):
        """
        numerical_gradientの高速版らしい
        誤差逆伝播法を使っているため早いらしい
        :param x:
        :param t:
        :return:
        """
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = self.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = self.softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        dz1 = np.dot(dy, W2.T)
        da1 = self.sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads


if __name__ == '__main__':
    # 28x28=784 で labelが10個.hidden_sizeは隠れ層の個数で適当らしい
    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    print(net.params['W1'].shape)  # (784, 100)
    print(net.params['b1'].shape)  # (100,)
    print(net.params['W2'].shape)  # (784, 100)
    print(net.params['b2'].shape)  # (10,)
    # dummy data. 100枚の28x28
    x = np.random.rand(100, 784)
    # 正解データ。100枚の0-10のlabel
    t = np.random.rand(100, 10)
    # 勾配情報
    grads = net.numerical_gradient(x, t)
    print(grads['W1'].shape)  # (784, 100)
    print(grads['b1'].shape)  # (100,)
    print(grads['W2'].shape)  # (784, 100)
    print(grads['b2'].shape)  # (10,)
