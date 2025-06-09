import os

import torch.nn
import argparse
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, rgb2gray

import utils
from model.hidden import *
from noise_layers.ilpf import IdealLowPassFilter
from noise_layers.gaussian import GaussianFilter
from noise_layers.identity import Identity
from noise_layers.noiser import Noiser
from PIL import Image
import torchvision.transforms.functional as TF


def randomCrop(img, height, width):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = 0; y = 0
    if (img.shape[1] - width) > 0:
        x = np.random.randint(0, img.shape[1] - width)
    if (img.shape[0] - height) > 0:
        y = np.random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    # img = img[218:218+128, 219:219+128]
    return img


def image_tensor_to_numpy(encoded_images):
    watermarked_images = encoded_images[:encoded_images.shape[0], :, :, :].cpu()

    # scale values to range [0, 1] from original range of [-1, 1]
    watermarked_images = (watermarked_images + 1) / 2
    watermarked_images = watermarked_images.permute(0, 2, 3, 1)

    watermarked_images = watermarked_images.detach().cpu().numpy()

    return watermarked_images[0]


def make_color_histogram():
    pass


def images_diff_plot(cover_image, encoded_image):
    difference_image = np.abs(cover_image - encoded_image)
    difference_image = rgb2gray(difference_image)
    # difference_image = np.clip(difference_image, 0, 255)
    difference_image = ((difference_image - np.min(difference_image)) /
                        (np.max(difference_image) - np.min(difference_image)))

    # TODO: OBLITERATE
    # difference_image_pil = Image.fromarray(np.uint8(difference_image * 255))
    # difference_image_pil.save('difference_image.png')

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1); plt.imshow(cover_image); plt.title("Cover")
    plt.subplot(1, 3, 2); plt.imshow(encoded_image); plt.title("Encoded")
    plt.subplot(1, 3, 3); plt.imshow(difference_image, cmap="gray"); plt.title("Difference")
    plt.show()

def ftransform_rgb_plot(cover_image, encoded_image):
    cover_red_transform = np.fft.fftshift(np.fft.fft2(cover_image[:, :, 0]))
    cover_green_transform = np.fft.fftshift(np.fft.fft2(cover_image[:, :, 1]))
    cover_blue_transform = np.fft.fftshift(np.fft.fft2(cover_image[:, :, 2]))
    encoded_red_transform = np.fft.fftshift(np.fft.fft2(encoded_image[:, :, 0]))
    encoded_green_transform = np.fft.fftshift(np.fft.fft2(encoded_image[:, :, 1]))
    encoded_blue_transform = np.fft.fftshift(np.fft.fft2(encoded_image[:, :, 2]))

    plt.figure(figsize=(10, 7))
    plt.subplot(2, 3, 1)
    plt.imshow(np.log(np.abs(cover_red_transform) + 1), cmap="jet", extent=(-0.5, 0.5, 0.5, -0.5))
    plt.title("Cover - Red")
    plt.subplot(2, 3, 2)
    plt.imshow(np.log(np.abs(cover_green_transform) + 1), cmap="jet", extent=(-0.5, 0.5, 0.5, -0.5))
    plt.title("Cover - Green")
    plt.subplot(2, 3, 3)
    plt.imshow(np.log(np.abs(cover_blue_transform) + 1), cmap="jet", extent=(-0.5, 0.5, 0.5, -0.5))
    plt.title("Cover - Blue")
    plt.subplot(2, 3, 4)
    plt.imshow(np.log(np.abs(encoded_red_transform) + 1), cmap="jet", extent=(-0.5, 0.5, 0.5, -0.5))
    plt.title("Encoded - Red")
    plt.subplot(2, 3, 5)
    plt.imshow(np.log(np.abs(encoded_green_transform) + 1), cmap="jet", extent=(-0.5, 0.5, 0.5, -0.5))
    plt.title("Encoded - Green")
    plt.subplot(2, 3, 6)
    plt.imshow(np.log(np.abs(encoded_blue_transform) + 1), cmap="jet", extent=(-0.5, 0.5, 0.5, -0.5))
    plt.title("Encoded - Blue")
    plt.show()


def ftransform_hsv_plot(cover_image, encoded_image):
    cover_hsv, encoded_hsv = rgb2hsv(cover_image), rgb2hsv(encoded_image)

    cover_hue_transform = np.fft.fftshift(np.fft.fft2(cover_hsv[:, :, 0]))
    cover_saturation_transform = np.fft.fftshift(np.fft.fft2(cover_hsv[:, :, 1]))
    cover_value_transform = np.fft.fftshift(np.fft.fft2(cover_hsv[:, :, 2]))
    encoded_hue_transform = np.fft.fftshift(np.fft.fft2(encoded_hsv[:, :, 0]))
    encoded_saturation_transform = np.fft.fftshift(np.fft.fft2(encoded_hsv[:, :, 1]))
    encoded_value_transform = np.fft.fftshift(np.fft.fft2(encoded_hsv[:, :, 2]))

    plt.figure(figsize=(10, 7))
    plt.subplot(2, 3, 1)
    plt.imshow(np.log(np.abs(cover_hue_transform) + 1), cmap="jet", extent=(-0.5, 0.5, 0.5, -0.5))
    plt.title("Cover - Hue")
    plt.subplot(2, 3, 2)
    plt.imshow(np.log(np.abs(cover_saturation_transform) + 1), cmap="jet", extent=(-0.5, 0.5, 0.5, -0.5))
    plt.title("Cover - Saturation")
    plt.subplot(2, 3, 3)
    plt.imshow(np.log(np.abs(cover_value_transform) + 1), cmap="jet", extent=(-0.5, 0.5, 0.5, -0.5))
    plt.title("Cover - Value")
    plt.subplot(2, 3, 4)
    plt.imshow(np.log(np.abs(encoded_hue_transform) + 1), cmap="jet", extent=(-0.5, 0.5, 0.5, -0.5))
    plt.title("Encoded - Hue")
    plt.subplot(2, 3, 5)
    plt.imshow(np.log(np.abs(encoded_saturation_transform) + 1), cmap="jet", extent=(-0.5, 0.5, 0.5, -0.5))
    plt.title("Encoded - Saturation")
    plt.subplot(2, 3, 6)
    plt.imshow(np.log(np.abs(encoded_value_transform) + 1), cmap="jet", extent=(-0.5, 0.5, 0.5, -0.5))
    plt.title("Encoded - Value")
    plt.show()



def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    parser = argparse.ArgumentParser(description='Test trained models')
    parser.add_argument('--options-file', '-o', default='options-and-config.pickle', type=str,
                        help='The file where the simulation options are stored.')
    parser.add_argument('--checkpoint-file', '-c', required=True, type=str, help='Model checkpoint file')
    parser.add_argument('--batch-size', '-b', default=12, type=int, help='The batch size.')
    parser.add_argument('--source-image', '-s', required=True, type=str,
                        help='The image to watermark')
    # parser.add_argument('--times', '-t', default=10, type=int,
    #                     help='Number iterations (insert watermark->extract).')

    args = parser.parse_args()

    train_options, hidden_config, noise_config = utils.load_options(args.options_file)
    # noiser = Noiser(["JpegPlaceholder"], device)
    noiser = Noiser([GaussianFilter(kernel_size=3, sigma=1)], device)
    # noiser = Noiser(noise_config, device)

    checkpoint = torch.load(args.checkpoint_file, map_location=device)
    hidden_net = Hidden(hidden_config, device, noiser, None)
    utils.model_from_checkpoint(hidden_net, checkpoint)

    # message_errors = []
    # total_losses = []
    # encoder_mse_losses = []
    # decoder_mse_losses = []
    # adversarial_bce_losses = []
    # discriminator_cover_bce_losses = []
    # discriminator_encoded_bce_losses = []
    # for image_dir in os.listdir("dataset/test/test/"):
    image_pil = Image.open(args.source_image)
    # image_pil = Image.open("dataset/test/test/" + image_dir)
    image = randomCrop(np.array(image_pil), hidden_config.H, hidden_config.W)
    image_tensor = TF.to_tensor(image).to(device)
    image_tensor = image_tensor * 2 - 1  # transform from [0, 1] to [-1, 1]
    image_tensor.unsqueeze_(0)

    # for t in range(args.times):
    message = torch.Tensor(np.random.choice([0, 1], (image_tensor.shape[0],
                                                    hidden_config.message_length))).to(device)
    losses, (encoded_images, noised_images, decoded_messages) = hidden_net.validate_on_batch([image_tensor, message])
    decoded_rounded = np.abs(decoded_messages.detach().cpu().numpy().round().clip(0, 1))
    message_detached = message.detach().cpu().numpy()
    print(f'original: {message_detached}')
    print(f'decoded : {decoded_rounded}')
    print(f'error : {np.mean(np.abs(decoded_rounded - message_detached)):.3f}')
    print(f'losses: {losses}')
    # message_errors.append(np.mean(np.abs(decoded_rounded - message_detached)))
    # total_losses.append(losses["loss"])
    # encoder_mse_losses.append(losses["encoder_mse"])
    # decoder_mse_losses.append(losses["dec_mse"])
    # adversarial_bce_losses.append(losses["adversarial_bce"])
    # discriminator_cover_bce_losses.append(losses["discr_cover_bce"])
    # discriminator_encoded_bce_losses.append(losses["discr_encod_bce"])

    # print("Test results (means):")
    # print("Bitwise error:", np.mean(message_errors))
    # print("Encoder MSE:", np.mean(encoder_mse_losses))
    # print("Decoder MSE:", np.mean(decoder_mse_losses))
    # print("Adversarial BCE:", np.mean(adversarial_bce_losses))
    # print("Discriminator BCE (Cover):", np.mean(discriminator_cover_bce_losses))
    # print("Discriminator BCE (Encoded):", np.mean(discriminator_encoded_bce_losses))

    image = image / 255

    # TODO: OBLITERATE
    # image_pil = Image.fromarray(np.uint8(image * 255))
    # image_pil.save("cover_image.png")

    encoded_image = image_tensor_to_numpy(encoded_images)

    # TODO: OBLITERATE
    # encoded_image_pil = np.clip(encoded_image, 0, 1)
    # encoded_image_pil = Image.fromarray(np.uint8(encoded_image_pil * 255))
    # encoded_image_pil.save("encoded_image.png")

    noised_image = image_tensor_to_numpy(noised_images)

    # TODO: OBLITERATE
    # noised_image_pil = ((noised_image - np.min(noised_image)) /
    #                      (np.max(noised_image) - np.min(noised_image)))
    # noised_image_pil = Image.fromarray(np.uint8(noised_image_pil * 255))
    # noised_image_pil.save("noised_image.png")

    # images_diff_plot(image, encoded_image)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(noised_image)
    plt.title("Noised")
    plt.show()
    # ftransform_rgb_plot(image, encoded_image)
    # ftransform_hsv_plot(image, encoded_image)

    # bitwise_avg_err = np.sum(np.abs(decoded_rounded - message.detach().cpu().numpy()))/(image_tensor.shape[0] * messages.shape[1])


if __name__ == '__main__':
    main()
