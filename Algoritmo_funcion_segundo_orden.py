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
from sympy import *
from simple_pid import PID
from plot_zplane import zplane 
import cmath

#%%Funciones definidas
def encontrar_cercano(vector, valor):#función para encontrar el valore mas cercano recorriendo el vector
    vector = np.asarray(vector)
    idx = (np.abs(vector - valor)).argmin()
    return vector[idx],idx

def primer_valor(vector,valor):#función que proporciona el índice del primer valor que pasa cierto valor
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
Kp=5
Wn=2
E=0.2
Ts =0.1

num=[Kp*Wn**2]
den=[1,2*E*Wn,Wn**2]
sys= signal.TransferFunction(num,den)
t,y=signal.step(sys)
referencia=np.linspace(Kp,Kp,len(t))

#%%
tp_y=max(y)#pico máximo de la función
tp_y_buscado,idx=encontrar_cercano(y,tp_y)#encontrar en que tiempo llega la función al máximo
tp=t[idx]#tiempo de pico máximo
wp=math.pi/tp#pulsación propia

#se gráfica el diferencial para ver el valor optimo te
Diferencial=np.diff(y)
plt.figure(1)
plt.plot(Diferencial,'go',label='Diferencial')
plt.xlabel('Índice')
plt.ylabel('Diferencial')
plt.legend(loc='best')
plt.grid(True)
plt.grid(True)


#%%
"Bloque 2:obtener ecuación trasnferencia"
Valor_dif=0.0005 #valor estimado en el que el sistema alcanza el establecimiento(pasa el transitorio)
idx=primer_valor(Diferencial,Valor_dif)#primer valor en el que el diferencial vale menos de "Valor_dif"

#%%
sigma=4/t[idx]#factor de decrecimiento
E_calculada=(sigma**2/(wp**2+sigma**2))**0.5#coeficiente de amortiguamiento "xi"
Wn_calculada=sigma/E_calculada
#%%
Kp_calcu=y[len(y)-1]*Wn_calculada**2#ganancia estática
#por deficicón de ft de segundo orden se estable el numerador y denominador como:
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

sys3=signal.StateSpace(A,B,C,D)#ecuacion de estado continua
t3,y3=signal.step(sys3)

#%%
"Bloque 4:pasar de espacio de estados(continuo) a función de transferencia discreta"

sys4=signal.cont2discrete((A,B,C,D),dt=Ts,method='zoh')#ecuacion en espoacio de estados discreta
num3,den3=signal.ss2tf(sys4[0],sys4[1],sys4[2],sys4[3])#numerador y denominador funcion de transferencia en discreto

sys5= signal.TransferFunction(num3,den3,dt=Ts)#funcion de trasnferencia discreta

#%%
"Bloque 4.2: pasar funcion de trasnferencia continua a funcion de trasnferencia discreta"

H_cont =control.tf( num2 , den2 )
#Pasar de función continua a discreta
H_disc=control.sample_system( H_cont , Ts , method = 'zoh' )#para usar esta función H_cont debe ser una función obtenida mediante "control.tf"




#%%
"Bloque 5:pasar funcion de trasnferencia discreta a ecuación en diferencias"
yk_1=0#valor del estado anterior de yk
yk_2=0#valor del estado anterior de yk_1
Uk=1#valor de referencia
n=len(t)
yk=np.zeros([n])
yk[0]=0
Tiempo=np.zeros([n])
Tiempo[0]=0
for k in range(1, n,1):
     yk[k]=-sys5.den[1]*yk_1-sys5.den[2]*yk_2+sys5.num[0]*Uk+sys5.num[1]*Uk
     #yk[k]=-0.517*yk_1-0.4325*yk_2+2.262*Uk+1.64*Uk
     yk_2=yk_1
     yk_1=yk[k]
     Tiempo[k]=Tiempo[k-1]+Ts
 
#Tiempo=np.linspace(0,t[len(t)-1],len(t))  
   
t4,y4= signal.dstep(sys5,n=n)  



#%%
"Bloque 6: Graficos:resultados de la estimación de la ft"
plt.figure(2)
plt.plot(t,y,'--b',label='Sistema referencia')
plt.plot(t2,y2,'--g',label='Sistema calculado')
plt.plot(t3,y3,':k',label='Sistema calculado espacio de estados')
plt.plot(t,referencia,'r',label='Ganancia estática')
plt.plot(t[idx],y[idx],'go',label='Tiempo establecimiento')
plt.plot(Tiempo,yk,':y',label='Sistema discretizado')
plt.xlabel('Tiempo (s)')
plt.ylabel('Ganancia')
plt.legend(loc='best')
plt.grid(True)
plt.suptitle('Estimación función segundo orden(entrada escalón)', fontsize=16)
plt.savefig('Resultados estimación función', dpi = 1080)
#%%
"Bloque 7: graficar sistema en bucle cerrado con P (ecuaciones en diferencias)"
K_ultima=(sys5.den[len(sys5.den)-1]-1)/(-sys5.num[len(sys5.num)-1])
Kc=K_ultima/2

#Calcular el error de posición
Ke=Kc*(sys5.num[0]+sys5.num[1])/(1+sys5.den[1]+sys5.den[2])
ep=1/(1+Ke)

#Variables para realizar el bucle de ecuaciones en diferencias
yk_1=0#valor del estado anterior de yk
yk_2=0#valor del estado anterior de yk_1
Uk_1=0#accion de control -1
Uk_2=0#accion de control -2
r=10#referencia
n=len(t)
ek_1=0
ek_2=0
yk=np.zeros([n])
yk[0]=0
Tiempo=np.zeros([n])
Tiempo[0]=0
for k in range(1, n,1):
     yk[k]=-sys5.den[1]*yk_1 - sys5.den[2]*yk_2 + sys5.num[0]*Uk_1 + sys5.num[1]*Uk_2
     Uk_1=Kc*(r-yk[k])#r-yk es el error
     Uk_2=Kc*(r-yk_1)
     yk_2=yk_1
     yk_1=yk[k]     
     Tiempo[k]=Tiempo[k-1]+Ts
     

#t5,y5= signal.dstep(sys5,n=n)
plt.figure(3)
plt.plot(Tiempo,yk,label='Sistema con regulador P')
plt.xlabel('Tiempo (s)')
plt.ylabel('Salida')
plt.legend(loc='best')
plt.grid(True)
plt.suptitle('Comportamiento del sistema en bucle cerrado regulador P(computacional)', fontsize=16)



#%%
"Bloque 8:graficar sistema en bucle cerrado con P con dstep"
z = control.TransferFunction.z
Kc_cont= control.tf([Kc],[1])#funcion de trasnferencia continua del proporcional
sys_Kc=control.sample_system( Kc_cont , Ts , method = 'zoh' )#funcion de trasnferencia discreta del proporcional
Hz=(sys_Kc*H_disc)/(1+sys_Kc*H_disc)#ecuancion en bucle cerrado
Hz_num=np.array(Hz.num[0][0])
Hz_den=np.array(Hz.den[0][0])

Hz_sys= signal.TransferFunction(Hz_num,Hz_den,dt=Ts)

t_Hz,Hz_Y= signal.dstep(Hz_sys,n=n)


plt.figure(4)
plt.plot(t_Hz,Hz_Y[:][0],label='Sistema con regulador P')
plt.xlabel('Tiempo (s)')
plt.ylabel('Salida')
plt.legend(loc='best')
plt.grid(True)
plt.suptitle('Comportamiento del sistema en bucle cerrado regulador P(función "signal.dstep")', fontsize=16)




plt.figure(5)
plt.grid(True)
plt.suptitle('Lugar de los polos sistema en bucle cerrado regulador P', fontsize=16)
zplane(Hz_num,Hz_den) #lugar de los polos

print(Hz_sys)
#%%
"Bloque 9:obtener el valor del controlador integral Ti"
#control.pole(Hz_sys)

#Se obtienen los polos en bucle cerrado con un controlador P con valor Ku
a_den=sys5.den[0]
b_den=sys5.den[1]
c_den=sys5.den[2]
a_num=sys5.num[0]
b_num=sys5.num[1]
#a_den*z^2+(b_den+Kc*a_num)z+(c_den+Kc*b_num)   ecuacion general
# a        b                                    coeficientes

a=a_den
b=b_den+K_ultima*a_num
c=c_den+K_ultima*b_num
d = (b**2) - (4*a*c)

sol1 = (-b-cmath.sqrt(d))/(2*a)
sol2 = (-b+cmath.sqrt(d))/(2*a)

Wp=(1/Ts)*math.atan(sol2.imag/sol2.real)
Tu=(2*math.pi)/Wp
Ti=Tu*1.5
#%%
"Bloque :formar la función de trasnferencia del PI"
Kc=0.001603
Ti=0.032069


C_PI_cont= control.tf([Kc*Ti,Kc],[Ti,0])#funcion de trasnferencia continua del PI
C_PI_dis=control.sample_system( C_PI_cont , Ts , method = 'zoh' )#funcion de trasnferencia discreta del proporcional
C_PI_dis_num=[Kc,-Kc*((Ti-Ts)/Ti)]
C_PI_dis_den=[1,-1]

#C_PI_dis=signal.TransferFunction(C_PI_dis_num,C_PI_dis_den,dt=Ts)

Hz_PI=(C_PI_dis*H_disc)/(1+C_PI_dis*H_disc)#ecuancion en bucle cerrado

C_PI_dis_num=np.array(Hz_PI.num[0][0])
C_PI_dis_den=np.array(Hz_PI.den[0][0])



C_PI_dis_sys= signal.TransferFunction(C_PI_dis_num,C_PI_dis_den,dt=Ts)

t_C_PI_dis,y_C_PI_dis= signal.dstep(C_PI_dis_sys,n=n)


plt.figure(6)
plt.plot(t_C_PI_dis,y_C_PI_dis[:][0],label='Sistema con regulador P')
plt.xlabel('Tiempo (s)')
plt.ylabel('Salida')
plt.legend(loc='best')
plt.grid(True)
plt.suptitle('Comportamiento del sistema en bucle cerrado regulador PI', fontsize=16)







