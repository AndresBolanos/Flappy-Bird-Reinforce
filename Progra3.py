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
#Ejemplos
#https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
#https://github.com/pytorch/examples/blob/master/mnist/main.py
#https://github.com/pytorch/examples

#Red
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import namedtuple
import torch.optim as optim
from pathlib import Path

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #Se crean tres convoluciones
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)                #Entradas(1 imagen), #Kernels, Size_kernel / Convolucion para train
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)               #Entradas, #Kernels, Size_kernel
        
        self.conv2_drop = nn.Dropout2d()

        #Salida de la convolucion, tamano del layer
        #self.fc1 = nn.Linear( 9680, 2)                                # Se disminuye el tamano de la capa
        self.fc2 = nn.Linear(9680, 2)                                       #Solo dos salidas, arriba o abajo
        self.fc3 = nn.Linear(9680, 1)                                       #Solo dos salidas, arriba o abajo

        #Guardar informacion
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        #Hace otro reshape a (#imagenes, reduccion del vector)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])

        #Funcion de activacion
        #x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        state_values = self.fc3(x)
        action_scores = self.fc2(x)
        
        return F.softmax(action_scores, dim=1), state_values[0]

model = Net()              #Instancia la red

#Estructura de la accion guardada
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
eps = np.finfo(np.float32).eps.item()
optimizer = optim.Adam(model.parameters(), lr=3e-2)

def select_action(model, state):
    #Se convierte la imagen a arreglo de thorch
    state = torch.from_numpy(state)
    #Hace reshape para garantizar un canal de entrada
    #Canales, #Imagenes, ancho, alto
    state= state.reshape(1, 1, state.shape[0], state.shape[1])
    #Convierte los valores a float, asi lo recibe conv2d
    state = state.type(torch.FloatTensor)
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()

def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.item()
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    #Se guarda el estado de la red
    torch.save(model.state_dict(), "Pesos.txt")
    optimizer.step()
    optimizer.zero_grad()
    del model.rewards[:]
    del model.saved_actions[:]
    
def run():
    # Se cargan los pesos
    file = Path("Pesos.txt")
    if (file.exists()):
        model.load_state_dict(torch.load("Pesos.txt"))
        model.eval()

    #Se instancia el juego 
    game = FlappyBird()
    p = PLE(game, fps=30, display_screen=True, reward_values={"loss": -1.0, "win": 1.0})

    p.init()
    reward = 0.0
    running_reward = 10
    t = 0                   #Tiempo por el que va del episodio
    imagenAnt = []
    
    while(True):
        if reward != 0 or p.game_over():
            if (reward == 1):
                print(str(reward)+"Pasoooo")
            else:
                print(reward)
            #Cada vez q termina un episodio
            last = model.rewards[-1]
            rewards = np.asarray(model.rewards)
            rewards[rewards == 0] = last
            model.rewards = rewards.tolist()
            #print(model.rewards)
            running_reward = running_reward * 0.99 + t * 0.01
            finish_episode()
            t = 0
            if (p.game_over()):
                p.reset_game()
        observation = p.getScreenRGB()                   #Obtiene una nuev imagen del juego
        new_image = convert_image(observation)    #Convierte la imagen a lo que recibe la red

        restaImg = []
        res = 0
        if imagenAnt != []:
            restaImg = new_image-imagenAnt
            res = select_action(model, restaImg)
        imagenAnt = new_image
            
        #Selecciona la accion a realizar
        #Descarga la imagen
        #cv.imwrite( "Imagenes/Gray_Image"+str(i)+".jpg", new_image );
        if (res == 1):
            action = None
        else:
            action = 119
           
        reward = p.act(action)
        #Guardo el reward
        model.rewards.append(reward)
        t = t+1

run()



