from ple.games.flappybird import FlappyBird
from ple import PLE
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

#Change many pixeles to 64
def change_borders(img):
    img[img ==77] = 64
    return close_borders(img)

#Put the black color to all borders in the image
def close_borders(img):
    img_aux=img.copy()
    start_to_change = False
    cont = 0;
    #Close the up side of the tube
    for i in range(len(img[0])):
        if (img[0][i] == 64):
            cont = cont + 1
            if (cont == 2):
                cont = 0
                if (not start_to_change):
                    start_to_change = True
                else:
                    start_to_change = False
        if (start_to_change):
            img_aux[0,i] = 64
            img_aux[1,i] = 64
    #Left and right border of image
    for i in range(len(img)):
        img_aux[i,len(img[0])-1] = 64
        img_aux[i,len(img[0])-2] = 64
        img_aux[i,0] = 64
        img_aux[i,1] = 64
    #Puts 1 to each 64 or 63(bird)  
    img_aux[img_aux == 63] = 1
    img_aux[img_aux == 64] = 1
    img_aux[img_aux != 1] = 0
    return img_aux

#Exceute all the process to convert the image
def convert_image(img):
    gray_scale_Image= cv.cvtColor(img, cv.COLOR_BGR2GRAY).T
    gray_scale_image_borders = change_borders(gray_scale_Image)
    black_white_image = scipy.ndimage.binary_fill_holes(gray_scale_image_borders).astype(int)
    black_white_image[black_white_image == 1] = 255
    #black_white_image[black_white_image == 0] = 255
    return black_white_image

def run():
    game = FlappyBird()
    p = PLE(game, fps=30, display_screen=True)
    #agent = myAgentHere(allowed_actions=p.getActionSet())

    p.init()
    reward = 0.0

    for i in range(150):
       if p.game_over():
               p.reset_game()

       observation = p.getScreenRGB()
       new_image = convert_image(observation);
       cv.imwrite( "Imagenes/Gray_Image"+str(i)+".jpg", new_image );
       action = None
       reward = p.act(action)

run()



