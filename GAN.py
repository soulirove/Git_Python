import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

image_cols = 28
image_rows = 28
channels = 1
image_shape = (image_rows, image_cols, channels)


def generator_creator():

    noise_shape = (100,)

    model = Sequential()

    model.add(Dense(256, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(np.prod(image_shape), activation='tanh'))
    model.add(Reshape(image_shape))

    model.summary()

    noise = Input(shape=noise_shape)
    img = model(noise)

    return Model(noise, img)


def discriminator_creator():

    model = Sequential()

    model.add(Flatten(input_shape=image_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    img = Input(shape=image_shape)
    validity = model(img)

    return Model(img, validity)


def RetainImages(epoch):
    A, B = 5, 5
    noise = np.random.normal(0, 1, (A * B, 100))
    GeneratedImages = generator.predict(noise)

    GeneratedImages = 0.5 * GeneratedImages + 0.5

    fig, axs = plt.subplots(A, B)
    Counter = 0
    for i in range(A):
        for j in range(B):
            axs[i, j].imshow(GeneratedImages[Counter, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            Counter += 1
    fig.savefig(
        "C:/Users/Souli/Desktop/Projects/Python/GAN/images/mnist_%d.png" % epoch)
    plt.close()


def teaching(epochs, batch_size=128, RetainGap=5000):

    (Xteach, _), (_, _) = mnist.load_data()

    Xteach = (Xteach.astype(np.float32) - 127.5) / 127.5

    Xteach = np.expand_dims(Xteach, axis=3)

    HalfAmount = int(batch_size / 2)

    for epoch in range(epochs):

        idx = np.random.randint(0, Xteach.shape[0], HalfAmount)
        imgs = Xteach[idx]

        noise = np.random.normal(0, 1, (HalfAmount, 100))

        GeneratedImages = generator.predict(noise)

        DiscriminatorLossToReal = discriminator.train_on_batch(
            imgs, np.ones((HalfAmount, 1)))
        DiscriminatorLossToFake = discriminator.train_on_batch(
            GeneratedImages, np.zeros((HalfAmount, 1)))
        DiscriminatorLoss = 0.5 * \
            np.add(DiscriminatorLossToReal, DiscriminatorLossToFake)

        noise = np.random.normal(0, 1, (batch_size, 100))

        valid_y = np.array([1] * batch_size)

        GeneratorLoss = combined.train_on_batch(noise, valid_y)

        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %
              (epoch, DiscriminatorLoss[0], 100*DiscriminatorLoss[1], GeneratorLoss))

        if epoch % RetainGap == 0:
            RetainImages(epoch)


optimizer = Adam(0.0002, 0.5)

discriminator = discriminator_creator()
discriminator.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

generator = generator_creator()
generator.compile(loss='binary_crossentropy', optimizer=optimizer)

C = Input(shape=(100,))
image = generator(C)

discriminator.trainable = False

valid = discriminator(image)


combined = Model(C, valid)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)


teaching(epochs=500000, batch_size=32, RetainGap=50000)

generator.save(
    'C:/Users/Souli/Desktop/Projects/Python/GAN/Models/generator_model.h5')
