# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 10:14:25 2020

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
from sympy import *
from simple_pid import PID

#%%Funciones definidas
def encontrar_cercano(vector, valor):
    vector = np.asarray(vector)
    idx = (np.abs(vector - valor)).argmin()
    return vector[idx],idx

def primer_valor(vector,valor):
    vector = np.asarray(vector)
    i=1
    z=0
    while i==1:
        if abs(vector[z])<valor:
            i=0
            break
        z=z+1
    return z
            
  
    
"Bloque 1:Sistema de referencia"
#%%
Kp=2
Wn=2
E=0.2
Ts =0.1

num=[Kp*Wn**2]
den=[1,2*E*Wn,Wn**2]
sys= signal.TransferFunction(num,den)
t,y=signal.step(sys)
referencia=np.linspace(Kp,Kp,len(t))

#%%
tp_y=max(y)
tp_y_buscado,idx=encontrar_cercano(y,tp_y)
tp=t[idx]
wp=math.pi/tp

Diferencial=np.diff(y)
plt.figure(1)
plt.plot(Diferencial,'go')
plt.xlabel('Índice')
plt.ylabel('Diferencial')
plt.grid(True)


#%%
"Bloque 2:obtener ecuación trasnferencia"
Valor_dif=0.0005
idx=primer_valor(Diferencial,Valor_dif)

#%%
sigma=4/t[idx]
E_calculada=(sigma**2/(wp**2+sigma**2))**0.5
Wn_calculada=sigma/E_calculada
#%%
Kp_calcu=y[len(y)-1]*Wn_calculada**2
num2=[Kp_calcu]
den2=[1,2*E_calculada*Wn_calculada,Wn_calculada**2]
sys2= signal.TransferFunction(num2,den2)
t2,y2=signal.step(sys2)


#%%
"Bloque 3: pasar la función de trasnferencia a ecuacion en espacio de estados"
#dx/dt=Ax+Bu
#v=Cx+Du
A=np.array([[-2*E_calculada*Wn_calculada, -Wn_calculada**2],[1,0]])
B=np.array([[Kp_calcu],[0]])
C=np.array([0,1])
D=np.array(0)

sys3=signal.StateSpace(A,B,C,D)
t3,y3=signal.step(sys3)

#%%
"Bloque 4:pasar de espacio de estados(continuo) a función de transferencia discreta"

sys4=signal.cont2discrete((A,B,C,D),dt=Ts,method='zoh')
num3,den3=signal.ss2tf(sys4[0],sys4[1],sys4[2],sys4[3])

sys5= signal.TransferFunction(num3,den3,dt=Ts)

#%%
"Bloque 4.2: pasar funcion de trasnferencia continua a funcion de trasnferencia discreta"

H_cont =control.tf( num2 , den2 )
#Pasar de función continua a discreta
H_disc=control.sample_system( H_cont , Ts , method = 'tustin' )#para usar esta función H_cont debe ser una función obtenida mediante "control.tf"




#%%
"Bloque 5:pasar funcion de trasnferencia discreta a ecuación en diferencias"
yk_1=0
yk_2=0
Uk=1
yk=np.zeros([200])
yk[0]=0
for k in range(1, 200,1):
     yk[k]=-sys5.den[1]*yk_1-sys5.den[2]*yk_2+sys5.num[0]*Uk+sys5.num[1]*Uk
     #yk[k]=-0.517*yk_1-0.4325*yk_2+2.262*Uk+1.64*Uk
     yk_2=yk_1
     yk_1=yk[k]
 
Tiempo=np.linspace(0,t[len(t)-1],len(t))  
   
t4,y4= signal.dstep(sys5,n=200)  



#%%
"Bloque 6: Graficos:resultados de la estimación de la ft"
plt.figure(2)
plt.plot(t,y,'--b',label='Sistema referencia')
plt.plot(t2,y2,'--g',label='Sistema calculado')
plt.plot(t3,y3,':k',label='Sistema calculado espacio de estados')
plt.plot(t,referencia,'r',label='Referencia')
plt.plot(t[idx],y[idx],'go',label='Tiempo establecimiento')
plt.plot(t4,y4[0],':y',label='Sistema discretizado')
plt.xlabel('Tiempo (s)')
plt.ylabel('Ganancia')
plt.legend(loc='best')
plt.grid(True)
plt.suptitle('Estimación función segundo orden(entrada escalón)', fontsize=16)
plt.savefig('Resultados estimación función', dpi = 1080)
#%%
from scipy import signal
import matplotlib.pyplot as plt

butter = signal.dlti(*signal.butter(3, 0.5))
t, y = signal.dstep(butter, n=25)
plt.step(t, np.squeeze(y))
plt.grid()
plt.xlabel('n [samples]')
plt.ylabel('Amplitude')


    





