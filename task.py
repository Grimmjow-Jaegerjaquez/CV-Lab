import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

image1 = cv2.imread(input("Please enter the path of the first image : "))
#image1 = cv2.imread('download.jpeg')
img = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
img = img.tolist()

image2 = cv2.imread(input("Please enter the path of the second image : "))
pic = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
pic = pic.tolist()

#display the changed image
def compare(original, manipulated, title_1="Original", title_2="Manipulated"):
    plt.figure(figsize=(15, 25))
    plt.subplot(1, 2, 1)
    plt.title(title_1)
    plt.imshow(original, cmap = "gray")
    plt.subplot(1, 2, 2)
    plt.title(title_2)
    plt.imshow(manipulated, cmap = "gray")
    plt.show()
    
def print_shape(img):
    try:
        print("Shape of the array is:", len(img), "x", len(img[0]), "x", len(img[0][0]))
    except:
        print("Shape of the array is:", len(img), "x", len(img[0])) 
    
def add_list(img1, img2):
    return [[img1[i][j] + img2[i][j] for j in range(len(img1[0]))] for i in range(len(img1))]

def channel_first(img):
    return [[[img[j][k][i] for k in range(len(img[0]))] for j in range(len(img))] for i in range(len(img[0][0]))]

def show_channel(img):
    channel_wise = channel_first(img)
    plt.figure(figsize=(10, 20))
    plt.subplot(1, 3, 1)
    plt.title("Red")
    plt.imshow(channel_wise[0], cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("Green")
    plt.imshow(channel_wise[1], cmap="gray")
    plt.subplot(1, 3, 3)
    plt.title("Blue")
    plt.imshow(channel_wise[2], cmap="gray")
    plt.show()
    
#color to grey
def convert_to_binary(og_image):
    #og_image = mpimg.imread(path)
    grayscale_image = np.dot(og_image[...,:3], [0.2989, 0.5870, 0.1140])
    return grayscale_image

#invert the color
def invert(img):
    return [[[255 - k for k in j] for j in i] for i in img]

#flip vertical
def flip_vertical(img):
    return [img[-i - 1] for i in range(len(img))]

#flip horizontal
def flip_horizontal(img):
    return [[img[i][-j - 1] for j in range(len(img[0]))] for i in range(len(img))]

#rotate left
def left_rotate(img):
    return [[[img[j][-1-i][k] for k in range(len(img[0][0]))] for j in range(len(img))] for i in range(len(img[0]))]

#rotate_right
def right_rotate(img):
    return [[[img[-1 - j][i][k] for k in range(len(img[0][0]))] for j in range(len(img))] for i in range(len(img[0]))]

#blur
def blur(img, strength = 1):
    arr1 = []
    for i in range(len(img)):
        arr2 = []
        for j in range(len(img[0])):
            arr3 = []
            for k in range(len(img[0][0])):
                arr4 = []
                for x in range(1, strength + 1):
                    a_pixels = 1
                    temp = img[i][j][k]
                    try:
                        temp += img[i + x][j + x][k]
                        a_pixels += 1
                    except:
                        True
                    
                    try:
                        temp += img[i + x][j][k]
                        a_pixels += 1
                    except:
                        True
                    
                    try:
                        temp += img[i + x][j - x][k]
                        a_pixels += 1
                    except:
                        True
                    
                    try:
                        temp += img[i][j - x][k]
                        a_pixels += 1
                    except:
                        True

                    try:
                        temp += img[i - x][j - x][k]
                        a_pixels += 1
                    except:
                        True
                    
                    try:
                        temp += img[i - x][j][k]
                        a_pixels += 1
                    except:
                        True
                    
                    try:
                        temp += img[i - x][j + x][k]
                        a_pixels += 1
                    except:
                        True
                    
                    try:
                        temp += img[i][j + x][k]
                        a_pixels += 1
                    except:
                        True
                    arr4.append(temp / a_pixels)
                arr3.append(int(sum(arr4) / len(arr4)))
            arr2.append(arr3)
        arr1.append(arr2)
    
    return arr1

#lightness
def lightness(img, b=50):
    return [[[int((255 * (b / 100)) + (img[i][j][k] * (1 - (b / 100)))) for k in range(len(img[0][0]))] for j in range(len(img[0]))] for i in range(len(img))]
    
#brightness
def brightness(img, strength):
    return [[[int((510 / (1 + (2.7183**(-strength * img[i][j][k] / 255)))) - 255) for k in range(len(img[0][0]))] for j in range(len(img[0]))] for i in range(len(img))]

#contrast
def contrast(img, strength=0):
    return [[[int(255 / (1 + (2.7183**(-strength * ((img[i][j][k] - 127.5) / 127.5))))) for k in range(len(img[0][0]))] for j in range(len(img[0]))] for i in range(len(img))]
    
#cutmix
def cutmix(path, x1, y1, x2, y2, x3, y3, x4, y4):
    og_image = mpimg.imread(path)
    manipulated_image = np.copy(og_image)
    manipulated_image[y1:y2, x1:x2, :] = og_image[y3:y4, x3:x4, :]
    manipulated_image[y3:y4, x3:x4, :] = og_image[y1:y2, x1:x2, :]
    return manipulated_image

#jigsaw
def jigsaw(original_image1, piece_size):
    #original_image1 = cv2.imread(path)
    original_image = cv2.cvtColor(original_image1, cv2.COLOR_BGR2RGB)
    height, width = original_image.shape[:2]
    num_rows = int(height / piece_size)
    num_cols = int(width / piece_size)

    canvas_height = num_rows * piece_size
    canvas_width = num_cols * piece_size
    jigsaw_canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    piece_indices = np.random.permutation(num_rows * num_cols)

    for piece_idx, canvas_idx in enumerate(piece_indices):
        piece_row = int(piece_idx / num_cols)
        piece_col = int(piece_idx % num_cols)

        canvas_row = int(canvas_idx / num_cols)
        canvas_col = int(canvas_idx % num_cols)

        piece_start_y = piece_row * piece_size
        piece_end_y = (piece_row + 1) * piece_size
        piece_start_x = piece_col * piece_size
        piece_end_x = (piece_col + 1) * piece_size

        canvas_start_y = canvas_row * piece_size
        canvas_end_y = (canvas_row + 1) * piece_size
        canvas_start_x = canvas_col * piece_size
        canvas_end_x = (canvas_col + 1) * piece_size

        jigsaw_canvas[canvas_start_y:canvas_end_y, canvas_start_x:canvas_end_x] = original_image[piece_start_y:piece_end_y, piece_start_x:piece_end_x]
    return jigsaw_canvas

#bounding box
def extract_bounding_box(og_image, x, y, w, h):
    #og_image = cv2.imread(path)
    original_image = cv2.cvtColor(og_image, cv2.COLOR_BGR2RGB)
    manipulated_image = original_image.copy()
    cv2.rectangle(manipulated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return manipulated_image

#mixup
def mixup(image1, image2, alpha):
    #image1 = cv2.imread(path1)
    #image2 = cv2.imread(input("Please enter the path of the second image required : "))
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    
    

    height, width = img1.shape[:2]
    resized_image2 = cv2.resize(img2, (width, height))
    mixup_image = cv2.addWeighted(img1, alpha, resized_image2, 1 - alpha, 0)
    return mixup_image
    

#add noise
def image_to_list_1(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    pixel_matrix = np.array(image)

    return pixel_matrix.tolist()

def add_noise(pixel_matrix, noise_level):
    noisy_matrix = np.array(pixel_matrix)
    noise = np.random.randint(-noise_level, noise_level + 1, size=noisy_matrix.shape)
    noisy_matrix += noise
    noisy_matrix = np.clip(noisy_matrix, 0, 255)

    return noisy_matrix.tolist()

#scaling
def resize(img, size):

    return [[[img[int(len(img) * i / size[0])][int(len(img[0]) * j / size[1])][k] for k in range(3)] for j in range(size[1])] for i in range(size[0])]

#hue and saturation
def adjust_hue_saturation(og_image, hue_factor, saturation_factor):
    hsv_image = cv2.cvtColor(og_image, cv2.COLOR_RGB2HSV)
    hsv_image[..., 0] = (hsv_image[..., 0] + hue_factor) % 180
    hsv_image[..., 1] *= saturation_factor
    manipulated_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    return manipulated_image

#downsacle
def downscale_image(og_image, max_dimension):
    #og_image = mpimg.imread(path)
    og_image = cv2.cvtColor(og_image, cv2.COLOR_BGR2RGB)
    aspect_ratio = og_image.shape[1] / og_image.shape[0]
    if og_image.shape[0] > og_image.shape[1]:
        new_height = max_dimension
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = max_dimension
        new_height = int(new_width / aspect_ratio)
        
    downscaled_image = cv2.resize(og_image, (new_width, new_height))
    return downscaled_image

#cropping
def crop(og_image, x, y, width, height):
    #og_image = mpimg.imread(path)
    og_image = cv2.cvtColor(og_image, cv2.COLOR_BGR2RGB)
    cropped_image = og_image[y : y + height, x : x + width]
    return cropped_image

#erase region
def erase_region(og_image, x, y, width, height):
    # og_image = cv2.imread(path)
    original_image = cv2.cvtColor(og_image, cv2.COLOR_BGR2RGB)
    manipulated_image = original_image.copy()
    cv2.rectangle(manipulated_image, (x, y), (x + width, y + height), (0, 0, 0), -1)
    return manipulated_image

def compare2(original1, original2, manipulated, title_1="Original1", title_2="Original2", title_3 = "Manipulated"):
    plt.figure(figsize=(15, 25))
    plt.subplot(1, 3, 1)
    plt.title(title_1)
    plt.imshow(original1)
    plt.subplot(1, 3, 2)
    plt.title(title_2)
    plt.imshow(original2)
    plt.subplot(1, 3, 3)
    plt.title(title_3)
    plt.imshow(manipulated)
    plt.show()