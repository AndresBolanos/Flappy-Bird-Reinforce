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

def resize_Image(img):
    height = img.shape[0]
    width = img.shape[1]
    res = cv.resize(img,(100, 100), interpolation = cv.INTER_NEAREST)
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


################################################################
#Red
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #Se crean tres convoluciones
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)                #Entradas(1 imagen), #Kernels, Size_kernel / Convolucion para train
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)               #Entradas, #Kernels, Size_kernel
        
        self.conv2_drop = nn.Dropout2d()

        #Salida de la convolucion, tamano del layer
        self.fc1 = nn.Linear( 9680, 20)                                # Se disminuye el tamano de la capa
        self.fc2 = nn.Linear(20, 2)                                       #Solo dos salidas, arriba o abajo

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        #Hace otro reshape a (#imagenes, reduccion del vector)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])

        #Funcion de activacion
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

CN = Net()              #Instancia la red

def train(model, data):
    model.train()
    output = model(data)
    #hacer backProp

def select_action(model, state):
    #Se convierte la imagen a arreglo de thorch
    state = torch.from_numpy(state)
    #Hace reshape para garantizar un canal de entrada
    #Canales, #Imagenes, ancho, alto
    state= state.reshape(1, 1, state.shape[0], state.shape[1])
    #Convierte los valores a float, asi lo recibe conv2d
    state = state.type(torch.FloatTensor)
    model.train()
    output = model(state)
    return torch.argmax(output, dim=1)[0]
    
def run():
    game = FlappyBird()
    p = PLE(game, fps=30, display_screen=True)

    p.init()
    reward = 0.0
    images = []
    
    while(True):
       if p.game_over():
               p.reset_game()
               #images = []

       observation = p.getScreenRGB()                   #Obtiene una nuev imagen del juego
       new_image = convert_image(observation);   #Convierte la imagen a lo que recibe la red
       #images+=[new_image]                                   #Guarda la imagen para train
    
       #Selecciona la accion a realizar
       res = select_action(CN, new_image)

       #Descarga la imagen
       #cv.imwrite( "Imagenes/Gray_Image"+str(i)+".jpg", new_image );
       
       if (res == 0):
           action = None
       else:
           action = 119
       reward = p.act(action)

run()



