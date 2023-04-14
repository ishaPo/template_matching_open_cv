import cv2 as cv
import os
from os import listdir
import imutils
import numpy as np


folder_dir = "./Dataset_internship"
result_images_dir = './test'
# 6 differnent methods of template matching
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

best_match = None
for scale in np.linspace(0.055, 0.5, 11):  #Pick scale based on your estimate of template to object in the image ratio
    print(scale)

for template_image in os.listdir(result_images_dir):
    i,j = 0, 0 
    # THE TEMPLATE IMAGE
    temp_image = cv.imread(result_images_dir+'/'+template_image,0)
    h,w = temp_image.shape[::] # height and width

    for image in os.listdir(folder_dir):
        # print(image)
        # reading images one by one from entire dataset
        img_rgb = cv.imread(folder_dir+'/'+image,0)
        img = cv.imread(folder_dir+'/'+image,0)

        # copy of img
        img2 = img.copy()

        # TEMPLATE RESIZE
        resized_template = imutils.resize(temp_image, width = int(temp_image.shape[1] * scale))
        result = cv.matchTemplate(img2,temp_image,eval(methods[4]))
        print(result)

        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

        if best_match is None or min_val <= best_match[0]:
            ideal_scale=scale  #Save the ideal scale for printout. 
            h, w = resized_template.shape[::] #Get the size of the scaled template to draw the rectangle. 
            best_match = [min_val, min_loc, ideal_scale]

            #Save the image with a red box around the detected object in the large image. 
            top_left = best_match[1]  #Change to max_loc for all except for TM_SQDIFF
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv.rectangle(img_rgb, top_left, bottom_right, (0, 0,255), 2)
            #Red rectangle with thickness 2.
            num = 0 
            cv.imwrite(f'./Results/{num}matched_resized.jpg', img_rgb)
            num+= 1


        # cv.imshow("Image",img)
        # cv.waitKey(0)
        # perform template matching

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if methods in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv.rectangle(img_rgb,top_left, bottom_right, (0,0,255), 2)

        # cv.imshow('Match',img2)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        cv.imwrite(f'./Results/{j}_{i}.jpg',img_rgb)
        i += 1






print('Success')

