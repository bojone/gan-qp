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
import os


if not os.path.exists('samples'):
    os.mkdir('samples')


imgs = glob.glob('../../CelebA-HQ/train/*.png')
np.random.shuffle(imgs)
img_dim = 128
z_dim = 128
num_layers = int(np.log2(img_dim)) - 3
max_num_channels = img_dim * 8
f_size = img_dim // 2**(num_layers + 1)
batch_size = 64


def imread(f, mode='gan'):
    x = misc.imread(f, mode='RGB')
    if mode == 'gan':
        x = misc.imresize(x, (img_dim, img_dim))
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
                if len(X) == self.batch_size or i == len(self.imgs)-1:
                    X = np.array(X)
                    yield X
                    X = []


# 编码器1（为了编码）
x_in = Input(shape=(img_dim, img_dim, 3))
x = x_in

for i in range(num_layers + 1):
    num_channels = max_num_channels // 2**(num_layers - i)
    x = Conv2D(num_channels,
               (5, 5),
               strides=(2, 2),
               padding='same')(x)
    if i > 0:
        x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

x = Flatten()(x)
x = Dense(z_dim)(x)

e_model = Model(x_in, x)
e_model.summary()


# 编码器2（为了判别器）
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
x = Dense(z_dim, use_bias=False)(x)

te_model = Model(x_in, x)
te_model.summary()


# 判别器
z_in = Input(shape=(z_dim * 2,))
z = z_in

z = Dense(1024, use_bias=False)(z)
z = LeakyReLU(0.2)(z)
z = Dense(1024, use_bias=False)(z)
z = LeakyReLU(0.2)(z)
z = Dense(1, use_bias=False)(z)

td_model = Model(z_in, z)
td_model.summary()


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
e_model.trainable = False

x_real,z_fake = x_in,z_in
x_fake = g_model(z_fake)
z_real = e_model(x_real)
x_real_encoded = te_model(x_real)
x_fake_encoded = te_model(x_fake)
xz_real = Concatenate()([x_real_encoded, z_real])
xz_fake = Concatenate()([x_fake_encoded, z_fake])
xz_real_score = td_model(xz_real)
xz_fake_score = td_model(xz_fake)

d_train_model = Model([x_in, z_in],
                      [xz_real_score, xz_fake_score])

d_loss = xz_real_score - xz_fake_score
d_loss = d_loss[:, 0]
d_norm = 10 * (K.mean(K.abs(x_real - x_fake), axis=[1, 2, 3]) + K.mean(K.abs(z_real - z_fake), axis=1))
d_loss = K.mean(- d_loss + 0.5 * d_loss**2 / d_norm)

d_train_model.add_loss(d_loss)
d_train_model.compile(optimizer=Adam(2e-4, 0.5))


# 整合模型（训练生成器）
g_model.trainable = True
e_model.trainable = True
td_model.trainable = False
te_model.trainable = False

x_real,z_fake = x_in,z_in
x_fake = g_model(z_fake)
z_real = e_model(x_real)
z_real_ = Lambda(lambda x: K.stop_gradient(x))(z_real)
x_real_ = g_model(z_real_)
x_fake_ = Lambda(lambda x: K.stop_gradient(x))(x_fake)
z_fake_ = e_model(x_fake_)

x_real_encoded = te_model(x_real)
x_fake_encoded = te_model(x_fake)
xz_real = Concatenate()([x_real_encoded, z_real])
xz_fake = Concatenate()([x_fake_encoded, z_fake])
xz_real_score = td_model(xz_real)
xz_fake_score = td_model(xz_fake)

g_train_model = Model([x_in, z_in],
                      [xz_real_score, xz_fake_score])

g_loss = K.mean(xz_real_score - xz_fake_score) + 2 * K.mean(K.square(z_fake - z_fake_)) + 3 * K.mean(K.square(x_real - x_real_))

g_train_model.add_loss(g_loss)
g_train_model.compile(optimizer=Adam(2e-4, 0.5))

g_train_model.metrics_names.append('d_loss')
g_train_model.metrics_tensors.append(K.mean(xz_real_score - xz_fake_score))
g_train_model.metrics_names.append('r_loss')
g_train_model.metrics_tensors.append(2 * K.mean(K.square(z_fake - z_fake_)) + 3 * K.mean(K.square(x_real - x_real_)))

# 检查模型结构
d_train_model.summary()
g_train_model.summary()


# 采样函数
def sample(path, n=8, z_samples=None):
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


# 重构采样函数
def sample_ae(path, n=8):
    figure = np.zeros((img_dim * n, img_dim * n, 3))
    for i in range(n):
        for j in range(n):
            if j % 2 == 0:
                x_sample = [imread(np.random.choice(imgs))]
            else:
                z_sample = e_model.predict(np.array(x_sample))
                x_sample = g_model.predict(z_sample)
            digit = x_sample[0]
            figure[i * img_dim:(i + 1) * img_dim,
                   j * img_dim:(j + 1) * img_dim] = digit
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype(int)
    imageio.imwrite(path, figure)


if __name__ == '__main__':

    iters_per_sample = 100
    total_iter = 1000000
    n_size = 8
    img_data = img_generator(imgs, 'gan', batch_size).__iter__()
    Z = np.random.randn(n_size**2, z_dim)

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
            sample_ae('samples/test_ae_%s.png' % i)
            g_train_model.save_weights('./g_train_model.weights')
