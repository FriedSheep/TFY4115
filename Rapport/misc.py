import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

gravity = 9.81
c = 2/5 # c=2/5 for en kule
masse = 0.031 # kg for kulen
radius = 0.011 # m for kulen

def v_y(y_num):
    return np.sqrt((2*gravity*(y[0]-y[y_num])) / (1+c))

def v_x():
    # cs(x) returns y(x)
    return np.sqrt((2*gravity*(y[0]-y)) / (1+c))
    # return np.sqrt((2*gravity*(y[0]-cs(x))) / (1+c))

def v():
    return np.sqrt((10*gravity*(y[0]-y)) / 7)


def krumning():
    # returnerer krumningsradiusen til banen i punktet x
    return d2y / (1 + dy**2)**(3/2)

def sentripetalakselerasjon():
    # returnerer sentripetalakselerasjonen til banen i punktet x
    return (v_x()**2) * krumning()

def helningsvinkel():
    # returnerer helningsvinkelen til banen i punktet x
    return np.arctan(dy)

def friksjonskraft():
    # returnerer friksjonskraften til banen i punktet x
    return masse * gravity * np.sin(helningsvinkel())

def normalkraft():
    # returnerer normalkraften til banen i punktet x
    return masse * (gravity * np.cos(helningsvinkel() + krumning()))


def statiskFriksjonskoeffisient():
    # returnerer friksjonskoeffisienten til banen i punktet x
    return 2*masse*gravity*np.sin(helningsvinkel()) / 7

def sluttfart():
    # returnerer sluttfarten til banen i punktet x
    return v()[-1]


def eulersmethod_x():
    # returnerer en tabell med 1401 verdier for y(x)
    t_n = np.zeros(Nx)
    
    for i in range(1, Nx):
        t_n[i] = t_n[i-1] + (2 * dx) / (v_x()[i-1] + v_x()[i])
    return t_n

def plot(figName, figTitle, xLabel, yLabel, x_values, y_values, graphLabel=""):
    figName = plt.figure()
    plt.plot(x_values, y_values, label= graphLabel)
    plt.title(figTitle)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.legend()
    plt.grid()
    plt.show()

h = 0.200
xfast=np.asarray([0,h,2*h,3*h,4*h,5*h,6*h,7*h])
yfast = [0.293, 0.249, 0.189, 0.222, 0.197, 0.159, 0.153, 0.136]

cs = CubicSpline(xfast, yfast, bc_type='natural')

xmin = 0.000
xmax = 1.401
dx = 0.001

x = np.arange(xmin, xmax, dx) 

Nx = len(x)
y = cs(x)       #y=tabell med 1401 verdier for y(x)
dy = cs(x,1)    #dy=tabell med 1401 verdier for y'(x)
d2y = cs(x,2)   #d2y=tabell med 1401 verdier for y''(x)

baneform = plt.figure('y(x)',figsize=(12,6))
plt.plot(x,y,xfast,yfast,'*')
plt.title('Banens form')
plt.xlabel('$x$ (m)',fontsize=20)
plt.ylabel('$y(x)$ (m)',fontsize=20)
plt.grid()
plt.show()
