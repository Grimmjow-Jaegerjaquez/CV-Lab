from PIL import Image

def threshold(img, t):
    w, h = img.size
    binary_img = Image.new("1", (w, h))

    for x in range(h):
        for y in range(w):
            pixel_value = img.getpixel((y, x))
            if pixel_value < t:
                binary_img.putpixel((y, x), 0)
            else:
                binary_img.putpixel((y, x), 255)

    return  binary_img

img = Image.open('resources\\srk.jfif').convert("L")
t = int(input("Enter threshold value : "))

out_img = threshold(img, t)
out_img.save("binary_image.jpg")