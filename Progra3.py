from ple.games.flappybird import FlappyBird
from ple import PLE
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

import torch
import torch.nn.functional as F


class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN,self).__init__()

        self.conv1 = torch.nn.Conv2d(1,1,kernel_size=11,stride=1,padding=0)   #input channel, output channel,kernel,stride,padding...
        self.pool = torch.nn.MaxPool2d(200,stride=2,padding=0)

        #Fully connected 
        self.fc1 = torch.nn.Linear(200,200) 
        #self.fc2 = torch.nn.Linear(1024, 2)

    def forward(self,x):
        print("Forward")
        #x = self.pool(F.relu(self.conv1(x)))
        print("pool")
        x = x.view(x.size(0),-1)
        print("view")
        x = F.relu(self.fc1(x.float()))
        print("relu Fully")
        x = self.fc1(x.float())
        print("fin")
        
        return(x)

#Cálculo para saber el tamaño de la salida
def outputSize(in_size,kernel_size,stride,padding):
    return (in_size - kernel_size + 2*padding)/stride + 1
    


########################## MANEJO DE LAS IMÁGENES ################################

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

def resize_Image(img):
    height = img.shape[0]
    width = img.shape[1]
    res = cv.resize(img,(200, 200), interpolation = cv.INTER_NEAREST)
    return res

#Exceute all the process to convert the image
def convert_image(img):
    gray_scale_Image= cv.cvtColor(img, cv.COLOR_BGR2GRAY).T
    gray_scale_image_borders = change_borders(gray_scale_Image)
    black_white_image = scipy.ndimage.binary_fill_holes(gray_scale_image_borders).astype(int)
    black_white_image[black_white_image == 1] = 255
    #black_white_image[black_white_image == 0] = 255
    black_white_image = black_white_image[:404]
    black_white_image = resize_Image(black_white_image)
    return black_white_image


############################## ENTRENAMIENTO #####################################



def run():
    model = CNN()
    game = FlappyBird()
    p = PLE(game, fps=30, display_screen=True)
    #agent = myAgentHere(allowed_actions=p.getActionSet())

    p.init()
    reward = 0.0

    while (True):
    #for i in range(150):
        if p.game_over():
            p.reset_game()

        observation = p.getScreenRGB()         #observations == state
        new_image = convert_image(observation);
        #cv.imwrite( "Imagenes/Gray_Image"+str(i)+".jpg", new_image );     #Guarda cada frame de cuando juega

        prob = np.random.uniform()
        pred_prob = model.forward(torch.from_numpy(new_image))
        if (pred_prob[0][0] >= prob):
            action = 119
        else:
            action = None
            
        #action = 119
        reward = p.act(action)

run()



