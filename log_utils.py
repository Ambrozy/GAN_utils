import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.preprocessing as P


def save_weights(generator, discriminator, save_path, save_label):
    generator.save_weights(f'{save_path}{save_label}-generator.h5')
    discriminator.save_weights(f'{save_path}{save_label}-discriminator.h5')


def load_weights(generator, discriminator, save_path, save_label):
    generator.load_weights(f'{save_path}{save_label}-generator.h5')
    discriminator.load_weights(f'{save_path}{save_label}-discriminator.h5')


def plot_results(generator, const_input):
    fake_imgs = generator.predict(const_input)
    plt.figure(figsize=(18, 2))
    len_imgs = len(fake_imgs)
    for i in range(len_imgs):
        plt.subplot(1, len_imgs, i + 1)
        plt.imshow((fake_imgs[i] + 1) * 0.5)
        plt.axis('off')
    plt.show()


def zero_pad(time_value):
    return '0' + str(time_value) if time_value < 10 else str(time_value)


def format_time(time_value):
    if time_value < 60:
        return str(round(time_value)) + 's'
    minutes = zero_pad(round(time_value // 60))
    seconds = zero_pad(round(time_value % 60))
    return str(minutes) + ':' + str(seconds)


def print_log(current, amount, time_value, params):
    bar_len = 30
    percent = int(bar_len * current / amount)
    if current == amount:
        progressbar = '=' * bar_len
    else:
        progressbar = '=' * (percent - 1) + '>' + '-' * (bar_len - percent)

    message = '\r' + str(current) + '/' + str(amount) + ' [' + progressbar + ']'

    step_time = time_value / (current + 1)
    step_time = f'{round(step_time)}s' if step_time > 1 else f'{round(1000 * step_time)}ms'
    message += f' - {format_time(time_value)} {step_time}/step'

    for key in params:
        param = str(np.cast['float16'](params[key]))
        message += ' - ' + key + ': ' + param

    sys.stdout.write(message)
    sys.stdout.flush()


def plot_history(history, loss_names, d_loss_name='d_loss', g_loss_name='g_loss'):
    num_epochs = range(1, len(history[loss_names[0]]) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(num_epochs, history[d_loss_name], label='discriminator loss')
    plt.plot(num_epochs, history[g_loss_name], label='generator loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss during training')
    plt.title('D/G losses')
    plt.legend()

    plt.subplot(1, 2, 2)
    for loss_key in loss_names:
        plt.plot(num_epochs, history[loss_key], label=loss_key)
    plt.xlabel('Epoch')
    plt.ylabel('Loss during training')
    plt.legend()

    plt.show()


def reconstruct_images(model, x_images, y_images, n_row, n_col, size=2, random=True):
    n = n_row * n_col

    if random:
        x = np.random.choice(x_images.shape[0], size=n, replace=False)
        y = y_images[x]
        x = x_images[x]
    else:
        y = y_images[:n]
        x = x_images[:n]

    recons = model.predict(x)
    dists = np.mean((recons - y) ** 2, axis=(1, 2, 3))
    recons = (recons + 1) * 0.5
    y = (y + 1) * 0.5

    plt.figure(figsize=(8 * size, 4 * size))
    for i in range(n):
        plt.subplot(n_row, 3 * n_col, 3 * i + 1)
        plt.imshow(x[i])
        plt.title('original', fontsize=8)
        plt.axis('off')

        plt.subplot(n_row, 3 * n_col, 3 * i + 2)
        plt.imshow(y[i])
        plt.title('expectation', fontsize=8)
        plt.axis('off')

        plt.subplot(n_row, 3 * n_col, 3 * i + 3)
        plt.imshow(recons[i])
        plt.title('L2: {:.3f}'.format(dists[i]), fontsize=8)
        plt.axis('off')

    plt.show()


def reconstruct_from_file(generator, file_name, target_size, size=1):
    image = P.image.load_img(file_name, target_size=target_size)
    image = np.array(image) / 255.0
    image = image[None, ...]

    recons = generator.predict(image)
    plt.figure(figsize=(4 * size, 4 * size))
    plt.subplot(1, 2, 1)
    plt.imshow(image[0])
    plt.title('original', fontsize=6 + 2 * size)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow((recons[0] + 1) * 0.5)
    plt.title('restored', fontsize=6 + 2 * size)
    plt.axis('off')
    plt.show()
