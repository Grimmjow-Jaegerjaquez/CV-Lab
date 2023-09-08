import cv2
import numpy as np
import matplotlib.pyplot as plt

def unsharp_mask(image, kernel_size = (15, 15), sigma = 1.0, amount = 1.0, threshold = 0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharp = float(amount + 1) * image - float(amount) * blurred
    sharp = np.maximum(sharp, np.zeros(sharp.shape))
    sharp = np.minimum(sharp, 255 * np.ones(sharp.shape))
    sharp = sharp.round().astype(np.uint8)

    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharp, image, where=low_contrast_mask)

    return sharp

def main():
    color_image = cv2.imread("resources\\image.jpg")
    og_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    sharpened_color_image = unsharp_mask(og_image)
    plt.subplot(131), plt.imshow(og_image, cmap='gray'), plt.title('Original')
    # plt.subplot(132), plt.imshow(median_filtered, cmap='gray'), plt.title('Median Filtered')
    plt.subplot(133), plt.imshow(sharpened_color_image, cmap='gray'), plt.title('Sharpened')
    plt.show()


if __name__ == '__main__':
    main()