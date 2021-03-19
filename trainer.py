import time
import numpy as np
import tensorflow as tf
import tensorflow.keras.optimizers as O
import tensorflow.compat.v1.losses as Losses
from log_utils import save_weights, plot_results, print_log

DEFAULT_LR = 1e-3
DEFAULT_BETA_1 = 0.7


class Trainer(object):
    def __init__(self, generator, discriminator, g_optimizer=None, d_optimizer=None, d_output_smoothing=True):
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer or O.Adam(DEFAULT_LR, beta_1=DEFAULT_BETA_1)
        self.d_optimizer = d_optimizer or O.Adam(10 * DEFAULT_LR, beta_1=DEFAULT_BETA_1)
        self.d_output_smoothing = d_output_smoothing

    def train(self, data_generator, epochs, save_path, save_label, per_epoch_plot=10):
        const_images = data_generator[0][0][:10]
        history = []
        params = {}

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            params = self.train_one_epoch(data_generator)
            history.append(list(params.values()))

            data_generator.on_epoch_end()
            save_weights(self.generator, self.discriminator, save_path, save_label)
            if (epoch + 1) % per_epoch_plot == 0:
                plot_results(self.generator, const_images)
            print()

        values = np.array(history).transpose((1, 0))
        return dict(zip(params.keys(), values))

    def train_one_epoch(self, data_generator):
        n_batches = len(data_generator)
        start_time = time.time()
        params = {}

        discriminator_batch_shape = (data_generator.batch_size, *self.discriminator.output_shape[1:])
        y_real_smoothing = tf.ones(discriminator_batch_shape) - tf.random.uniform(discriminator_batch_shape) * 0.2
        y_fake_smoothing = tf.random.uniform(discriminator_batch_shape) * 0.2
        y_real_pure = tf.ones(discriminator_batch_shape)

        for batch_idx, (x_images, y_images) in enumerate(data_generator):
            x_images, y_images = [tf.dtypes.cast(x, tf.float32) for x in [x_images, y_images]]
            y_real = y_real_smoothing[:len(x_images)]
            y_fake = y_fake_smoothing[:len(x_images)]

            # train discriminator
            fake_imgs = self.generator.predict(x_images)
            d_out = self.discriminator_train_on_batch((y_images, y_real), (fake_imgs, y_fake))

            # train generator
            y_real = y_real_pure[:len(x_images)]
            g_out = self.gan_train_on_batch(x_images, y_images, y_real)

            params = {**d_out, **g_out}
            print_log(batch_idx, n_batches, time.time() - start_time, params)
        print_log(n_batches, n_batches, time.time() - start_time, params)
        return params

    def discriminator_train_on_batch(self, real, fake):
        x_real, y_real = real
        x_fake, y_fake = fake
        x = tf.concat([x_real, x_fake], axis=0)
        y = tf.concat([y_real, y_fake], axis=0)

        with tf.GradientTape() as tape:
            output = self.discriminator(x, training=True)
            loss, losses = self.discriminator_loss(y, output)
        grads = tape.gradient(loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        metrics = self.discriminator_metrics(y, output)
        return {**losses, **metrics}

    def discriminator_loss(self, y_true, y_pred):
        loss = Losses.sigmoid_cross_entropy(y_pred, y_true)
        return loss, {'d_loss': loss}

    def discriminator_metrics(self, y_true, y_pred):
        total = tf.reduce_prod(y_true.shape)
        current = tf.reduce_sum(tf.cast(tf.round(y_true) == tf.round(y_pred), tf.int32))
        accuracy = current / total
        return {'d_acc': accuracy}

    def gan_train_on_batch(self, x_images, y_images, y_dis):
        with tf.GradientTape() as tape:
            y_pred_images = self.generator(x_images, training=True)
            output = self.discriminator(y_pred_images, training=False)
            loss, losses = self.generator_loss(y_images, y_pred_images, y_dis, output)
        grads = tape.gradient(loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))

        metrics = self.generator_metrics(y_images, y_pred_images, y_dis, output)
        return {**losses, **metrics}

    def generator_loss(self, y_img_true, y_img_pred, y_dis_true, y_dis_pred):
        loss1 = Losses.mean_squared_error(y_img_true, y_img_pred)
        loss2 = Losses.sigmoid_cross_entropy(y_dis_true, y_dis_pred)
        gan_lambda = 1e-2
        loss = loss1 + gan_lambda * loss2
        return loss, {'loss': loss, 'img_loss': loss1, 'g_loss': loss2}

    def generator_metrics(self, y_img_true, y_img_pred, y_dis_true, y_dis_pred):
        return {}
