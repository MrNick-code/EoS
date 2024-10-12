import os
import cv2

def cut_imgs(input_dir, crop_h, crop_w, output_dir): # 640 / 480
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"): 
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)

            if image is not None:
                height, width, _ = image.shape

                top = (height - crop_h) // 2
                bottom = top + crop_h
                left = (width - crop_w) // 2
                right = left + crop_w

                cropped_image = image[top:bottom, left:right]

                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, cropped_image)

eosl_crop = cut_imgs("D:/Users/mathe/ML/EoS/IMG_DATA/EOSL_low_40",
                    40, 40,
                    "D:/Users/mathe/ML/EoS/IMG_DATA/DATA_CROP/low40_cropL")
