#! -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy import misc
import glob
import imageio
from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3,preprocess_input
import os


if not os.path.exists('samples'):
    os.mkdir('samples')


imgs = glob.glob('../../../data/celebahq/CelebA-HQ/train/*.png')
np.random.shuffle(imgs)
img_dim = 512
z_dim = 128
num_layers = int(np.log2(img_dim)) - 3
max_num_channels = img_dim * 4
f_size = img_dim // 2**(num_layers + 1)
batch_size = 32


def imread(f, mode='gan'):
    x = misc.imread(f, mode='RGB')
    if mode == 'gan':
        # x = misc.imresize(x, (img_dim, img_dim))
        return x.astype(np.float32) / 255 * 2 - 1
    elif mode == 'fid':
        x = misc.imresize(x, (299, 299))
        return x.astype(np.float32)


class img_generator:
    """图片迭代器，方便重复调用
    """
    def __init__(self, imgs, mode='gan', batch_size=64):
        self.imgs = imgs
        self.batch_size = batch_size
        self.mode = mode
        if len(imgs) % batch_size == 0:
            self.steps = len(imgs) // batch_size
        else:
            self.steps = len(imgs) // batch_size + 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        X = []
        while True:
            np.random.shuffle(self.imgs)
            for i,f in enumerate(self.imgs):
                X.append(imread(f, self.mode))
                if len(X) == batch_size or i == len(self.imgs)-1:
                    X = np.array(X)
                    yield X
                    X = []


class FID:
    """基于Python的FID计算
    """
    def __init__(self, x_real,
                 from_generator=False,
                 batch_size=None,
                 steps=None):
        """初始化，把真实样本的统计结果存起来，以便多次测试
        """
        self.base_model = InceptionV3(include_top=False,
                                      pooling='avg')
        self.mu_real,self.sigma_real = self.evaluate_mu_sigma(x_real,
                                                              from_generator,
                                                              batch_size,
                                                              steps)
    def evaluate_mu_sigma(self, x,
                          from_generator=False,
                          batch_size=None,
                          steps=None):
        """根据样本计算均值和协方差矩阵
        """
        if from_generator:
            steps = steps if steps else len(x)
            def _generator():
                for _x in x:
                    _x = preprocess_input(_x.copy())
                    yield _x
            h = self.base_model.predict_generator(_generator(),
                                                  verbose=True,
                                                  steps=steps)
        else:
            x = preprocess_input(x.copy())
            h = self.base_model.predict(x,
                                        verbose=True,
                                        batch_size=batch_size)
        mu = h.mean(0)
        sigma = np.cov(h.T)
        return mu,sigma
    def evaluate(self, x_fake,
                 from_generator=False,
                 batch_size=None,
                 steps=None):
        """计算FID值
        """
        mu_real,sigma_real = self.mu_real,self.sigma_real
        mu_fake,sigma_fake = self.evaluate_mu_sigma(x_fake,
                                                    from_generator,
                                                    batch_size,
                                                    steps)
        mu_diff = mu_real - mu_fake
        sigma_root = sp.linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)[0]
        sigma_diff = sigma_real + sigma_fake - 2 * sigma_root
        return np.real((mu_diff**2).sum() + np.trace(sigma_diff))


# 判别器
x_in = Input(shape=(img_dim, img_dim, 3))
x = x_in

for i in range(num_layers + 1):
    num_channels = max_num_channels // 2**(num_layers - i)
    x = Conv2D(num_channels,
               (5, 5),
               strides=(2, 2),
               use_bias=False,
               padding='same')(x)
    if i > 0:
        x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

x = Flatten()(x)
x = Dense(1, use_bias=False)(x)

d_model = Model(x_in, x)
d_model.summary()


# 生成器
z_in = Input(shape=(z_dim, ))
z = z_in

z = Dense(f_size**2 * max_num_channels)(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Reshape((f_size, f_size, max_num_channels))(z)

for i in range(num_layers):
    num_channels = max_num_channels // 2**(i + 1)
    z = Conv2DTranspose(num_channels,
                        (5, 5),
                        strides=(2, 2),
                        padding='same')(z)
    z = BatchNormalization()(z)
    z = Activation('relu')(z)

z = Conv2DTranspose(3,
                    (5, 5),
                    strides=(2, 2),
                    padding='same')(z)
z = Activation('tanh')(z)

g_model = Model(z_in, z)
g_model.summary()


# 整合模型（训练判别器）
x_in = Input(shape=(img_dim, img_dim, 3))
z_in = Input(shape=(z_dim, ))
g_model.trainable = False

x_real = x_in
x_fake = g_model(z_in)

x_real_score = d_model(x_real)
x_fake_score = d_model(x_fake)

d_train_model = Model([x_in, z_in],
                      [x_real_score, x_fake_score])

d_loss = x_real_score - x_fake_score
d_norm = 10 * K.mean(K.abs(x_real - x_fake), axis=[1, 2, 3])
d_loss = K.mean(- d_loss + 0.5 * d_loss**2 / d_norm)

d_train_model.add_loss(d_loss)
d_train_model.compile(optimizer=Adam(2e-4, 0.5))


# 整合模型（训练生成器）
g_model.trainable = True
d_model.trainable = False

x_real = x_in
x_fake = g_model(z_in)

x_real_score = d_model(x_real)
x_fake_score = d_model(x_fake)

g_train_model = Model([x_in, z_in],
                      [x_real_score, x_fake_score])

g_loss = K.mean(x_real_score - x_fake_score)

g_train_model.add_loss(g_loss)
g_train_model.compile(optimizer=Adam(2e-4, 0.5))


# 检查模型结构
d_train_model.summary()
g_train_model.summary()


# 采样函数
def sample(path, n=9, z_samples=None):
    figure = np.zeros((img_dim * n, img_dim * n, 3))
    if z_samples is None:
        z_samples = np.random.randn(n**2, z_dim)
    for i in range(n):
        for j in range(n):
            z_sample = z_samples[[i * n + j]]
            x_sample = g_model.predict(z_sample)
            digit = x_sample[0]
            figure[i * img_dim:(i + 1) * img_dim,
                   j * img_dim:(j + 1) * img_dim] = digit
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype(int)
    imageio.imwrite(path, figure)


if __name__ == '__main__':

    import json

    iters_per_sample = 100
    fid_per_sample = 1000
    total_iter = 1000000
    n_size = 9
    img_data = img_generator(imgs, 'gan', batch_size).__iter__()
    Z = np.random.randn(n_size**2, z_dim)
    logs = {'fid': [], 'best': 1000}

    print u'初始化FID评估器...'
    fid_evaluator = FID(img_generator(imgs, 'fid', batch_size), True)

    for i in range(total_iter):
        for j in range(2):
            x_sample = img_data.next()
            z_sample = np.random.randn(len(x_sample), z_dim)
            d_loss = d_train_model.train_on_batch(
                [x_sample, z_sample], None)
        for j in range(1):
            x_sample = img_data.next()
            z_sample = np.random.randn(len(x_sample), z_dim)
            g_loss = g_train_model.train_on_batch(
                [x_sample, z_sample], None)
        if i % 10 == 0:
            print 'iter: %s, d_loss: %s, g_loss: %s' % (i, d_loss, g_loss)
        if i % iters_per_sample == 0:
            sample('samples/test_%s.png' % i, n_size, Z)
            g_train_model.save_weights('./g_train_model.weights')
        if i % fid_per_sample == 0:
            def _generator():
                while True:
                    _z_fake = np.random.randn(100, z_dim)
                    _x_fake = g_model.predict(_z_fake,
                                              batch_size=batch_size)
                    _x_fake = np.round((_x_fake + 1) / 2 * 255, 0)
                    _x_fake = np.array([misc.imresize(_x, (299, 299)) for _x in _x_fake])
                    yield _x_fake
            fid = fid_evaluator.evaluate(_generator(), True, steps=100)
            logs['fid'].append((i, fid))
            if fid < logs['best']:
                logs['best'] = fid
                g_train_model.save_weights('./g_train_model.best.weights')
            json.dump(logs, open('logs.txt', 'w'), indent=4)
            print 'iter: %s, fid: %s, best: %s' % (i, fid, logs['best'])
        if i > 10000:
            fid_per_sample = 100
