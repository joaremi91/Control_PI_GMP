# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 08:17:32 2020

@author: jrequena
"""

"Importar librerias necesarias"

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math 
from array import *
from matplotlib.mlab import psd, csd
from scipy import signal
import control
import control.matlab

#%%Función definida
def encontrar_cercano(vector, valor):
    vector = np.asarray(vector)
    idx = (np.abs(vector - valor)).argmin()
    return vector[idx],idx

#%%
Kp= 2
taup=2
num=[Kp]
den=[taup,1]
sys= signal.TransferFunction(num,den)
t,y=signal.step(sys)
referencia=np.linspace(Kp,Kp,len(t))
#%%
t_establecimiento=0.98+max(y)
Kp_taup=0.632*max(y)
Kp_tau_buscado,idx=encontrar_cercano(y,Kp_taup)
#%%
taup_calculada=t[idx]
Kp_calculada=max(y)
num2=[Kp_calculada]
den2=[taup_calculada,1]
sys2= signal.TransferFunction(num2,den2)
t2,y2=signal.step(sys2)



#%%
plt.figure(1)
plt.plot(t,y,'--c',label='Sistema referencia')
plt.plot(t,referencia,'r',label='Referencia')
plt.plot(t2,y2,':b',label='Sistema calculado')
plt.xlabel('Tiempo (s)')
plt.ylabel('Ganancia')
plt.legend(loc='best')
plt.grid(True)

plt.plot(t[idx],Kp_tau_buscado,'go')
plt.legend(loc='best')
plt.suptitle('Estimación función primer orden(entrada escalón)', fontsize=16)
plt.savefig('Estimación función primer orden(entrada escalón)', dpi = 1080)



