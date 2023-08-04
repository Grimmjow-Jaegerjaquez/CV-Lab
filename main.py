from task import *

print("Choose your image manipulation :\n")
print("1. Horizontal Flipping\n"
      "2. Vertical Flipping\n"
      "3. Scaling\n"
      "4. Rotation\n"
      "5. Croppingdw\n"
      "6. Adding Noise\n"
      "7. Contrast\n"
      "8. Brightness\n"
      "9. Hue and Saturation\n"
     "10. Invert Image\n"
     "11. Color to Gray\n"
     "12. Blur\n"
     "13. Down-Scale\n"
     "14. Mask Region/Erase Region\n"
     "15. Mixup\n"
     "16. Bounding Box\n"
     "17. Jigsaw\n")
choice = int(input("Please enter your choice : "))

if choice == 1:
    compare(img, flip_horizontal(img))

elif choice == 2:
    compare(img, flip_vertical(img))

elif choice == 3:
    h = int(input("Please enter required height(Range : 0 to 1000) : "))
    w = int(input("Please enter required width(Range : 0 to 1000) : "))
    if (h >= 0 and h <= 1000) and (w >= 0 and w <= 1000):
        compare(img, resize(img, (h, w)), title_2="Resized")
    else:
        print("Invalid option")

elif choice == 4:
    print("1. To keep the image as it is please enter 0\n"
          "2. To rotate to towards right please enter 90\n"
          "3. To flip the image vertically please enter 180\n"
          "4. To rotate the image towards left please enter 270\n"
          "5. To flip the image horizontally please enter 360\n"
          "Any other input will result in Invalid option")
    direction = int(input("Please enter degree of rotation : "))
    if direction == 0:
        compare(img, img, title_2="Rotated")
    elif direction == 90:
        compare(img, right_rotate(img), title_2="Rotated")
    elif direction == 180:
        compare(img, flip_vertical(img), title_2="Rotated")
    elif direction == 270:
        compare(img, left_rotate(img), title_2="Rotated")
    elif direction == 360:
        compare(img, flip_horizontal(img), title_2="Rotated")
    else:
        print("Invalid option")
    
elif choice == 5:
    x = int(input("Please enter the starting X-coordinate of the crop(Range : 0 to 300) : "))
    y = int(input("Please enter the starting Y-coordinate of the crop(Range : 0 to 180) : "))
    w = int(input("Please enter the width of the crop(Range : 0 to 300) : "))
    h = int(input("Please enter the height of the crop(Range : 0 to 180) : "))
    if x == y == w == h == 0:
        compare(img, crop(image1, x, y, w, h), title_2="Cropped")
    elif (x >= 0 and x <= 300) and (y >= 0 and y <= 180) and (x != w) and (y != h):
        compare(img, crop(image1, x, y, w, h), title_2="Cropped")
    else:
        print("Invalid Option")
    
elif choice == 6:
    noise_f = int(input("Please enter noise level(Range : 0 to 700) : "))
    if noise_f >= 0 and noise_f <= 700:
        compare(img, add_noise(img, noise_f), title_2="Noisy Image")
    
    else:
        print("Invalid Option")
    
elif choice == 7:
    c_level = int(input("Please enter contrast level(Range : 0 to 700) : "))
    if c_level >= 0 and c_level <= 700:
        compare(img, contrast(img, c_level), title_2="Contrast")
    else:
        print("Invalid Option")
    
elif choice == 8:
    b_level = int(input("Please enter brightness level(Range : 0 to 50) : "))
    
    compare(img, brightness(img, b_level), title_2="Bright")
    
elif choice == 9:
    hue = int(input("Please enter hue factor(Range : 0 to 20) : "))
    saturation = int(input("Please enter saturation factor(Range : 0 to 20) : "))
    if (hue >= 0 and hue <= 20) and (saturation >= 0 and saturation <= 20):
        compare(img, adjust_hue_saturation(image1, hue, saturation), title_2="Hue and Saturation")
    else:
        print("Invalid option")
        
elif choice == 10:
    compare(img, invert(img), title_2="Inverted")
    
elif choice == 11:
    compare(img, convert_to_binary(image1), title_1="Original", title_2="Grayscale")
    
elif choice == 12:
    b = int(input("Please enter blur factor(Range : 0 to 20) : "))
    if b == 0:
        compare(img, img, title_2="Blur")
    elif b >= 1 and b <= 20:
        compare(img, blur(img, b), title_2="Blur")
    else:
        print("Invalid Option")
    
elif choice == 13:
    d_scale = int(input("Please enter downsacle factor(Range : 0 to 50) : "))
    compare(img, downscale_image(image1, d_scale), title_2="Downscaled")
    
elif choice == 14:
    x = int(input("Please enter the starting X-coordinate of the erased region(Range : 0 to 300) : "))
    y = int(input("Please enter the starting Y-coordinate of the erased region(Range : 0 to 180) : "))
    w = int(input("Please enter the width of the erased region(Range : 0 to 300) : "))
    h = int(input("Please enter the height of the erased region(Range : 0 to 180) : "))
    if x == y == w == h == 0:
        compare(img, erase_region(image1, x, y, w, h), title_2="Erased Region")
    elif (x >= 0 and x <= 300) and (y >= 0 and y <= 180) and (x != w) and (y != h):
        compare(img, erase_region(image1, x, y, w, h), title_2="Erased Region")
    else:
        print("Invalid Option")
  
#elif choice == 15:
    #compare(img, cutmix('download.jpeg', 100, 100, 300, 300, 400, 100, 600, 300))
    
elif choice == 15:
    a = float(input("Please enter mixup factor(Range 0 to 1) : "))
    if a >= 0 or a <= 1:
        compare2(img, pic, mixup(image1, image2, a), title_1 = "Image1", title_2="Image2",title_3 = "Mixup")
    else:
        print("Invalid option")
    
elif choice == 16:
    x = int(input("Please enter the starting X-coordinate of the bounding box : "))
    y = int(input("Please enter the starting Y-coordinate of the bounding box : "))
    w = int(input("Please enter the width  of the bounding box : "))  
    h = int(input("Please enter the height of the bounding box : "))
    if x == y == w == h == 0:
        compare(img, extract_bounding_box(image1, x, y, w, h))
    elif (x >= 0 and x <= 300) and (y >= 0 and y <= 180) and (x != w) and (y != h):
        compare(img, extract_bounding_box(image1, x, y, w, h))
    else:
        print("Invalid Option")
    
elif choice == 17:
    piece_size = int(input("Please enter piece size(Range : 0 to 80) : "))
    if piece_size >= 0 or piece_size <= 80:
        compare(img, jigsaw(image1, piece_size), title_2="Jigsaw")
    else:
        print("Invalid option")
    
else:
    print("Invalid choice")