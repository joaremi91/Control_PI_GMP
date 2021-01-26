# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 10:09:05 2020

@author: jrequena
"""

"Importar librerias necesarias"


import matplotlib.pyplot as plt
import numpy as np
import math 
from scipy import signal
import control  
import cmath
from matplotlib.widgets import Slider #libreria para plot interactivo



#%%
"Funciones definidas"
#función para encontrar el valore mas cercano recorriendo el vector
def encontrar_cercano(vector, valor):
    vector = np.asarray(vector)
    idx = (np.abs(vector - valor)).argmin()
    return vector[idx],idx

#función que proporciona el índice del primer valor que pasa cierto valor
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
#Funcion para actualizar gráfica(crea una nueva función de trasnferencia del PI)
def actualizar_grafica(Kcc,Tii):
    Kc=Kcc
    Ti=Tii
    n=len(t)
    C_PI_cont= control.tf([Kc*Ti,Kc],[Ti,0])#funcion de trasnferencia continua del PI
    C_PI_dis=control.sample_system( C_PI_cont , Ts , method = 'zoh' )#funcion de trasnferencia discreta del proporcional
    C_PI_dis_num=[Kc,-Kc*((Ti-Ts)/Ti)]
    C_PI_dis_den=[1,-1]
    Hz_PI=(C_PI_dis*H_disc)/(1+C_PI_dis*H_disc)#ecuancion en bucle cerrado
    C_PI_dis_num=np.array(Hz_PI.num[0][0])
    C_PI_dis_den=np.array(Hz_PI.den[0][0])
    C_PI_dis_sys= signal.TransferFunction(C_PI_dis_num,C_PI_dis_den,dt=Ts)
    t_C_PI_dis,y_C_PI_dis= signal.dstep(C_PI_dis_sys,t=t,n=n)
    return  t_C_PI_dis, y_C_PI_dis[:][0]          

#Funció para actualizar Kc
def update_Kc(Kc):
    global Kcc
    Kcc=Kc
    t,y=actualizar_grafica(Kcc,Tii)
    Kp_tuner.set_ydata(y) # set new y-coordinates of the plotted points
    plt.xlim(0, t_C_PI_dis[len(t_C_PI_dis)-1])
    plt.ylim(min(y)-2, max(y)+2)
    fig.canvas.draw_idle()
    print(Kcc,Tii)           # redraw the plot
    
#Funció para actualizar Ti
def update_Ti(Ti):
    global Tii
    Tii=Ti
    t,y=actualizar_grafica(Kcc,Tii)
    Kp_tuner.set_ydata(y) # set new y-coordinates of the plotted points
    #Kp_tuner.set_ydata(np.linspace(1,1,len(t_C_PI_dis)))
    plt.xlim(0, t_C_PI_dis[len(t_C_PI_dis)-1])
    plt.ylim(min(y)-2, max(y)+2)
    fig.canvas.draw_idle() 
    print(Kcc,Tii)     # redraw the plot    
    

#%%
"Bloque 1:Sistema de referencia"
Kp=1
Wn=10
E=0.01
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

Diferencial=np.diff(y)



#%%
"Bloque 2:obtener ecuación trasnferencia"
Valor_dif=0.0005 #valor estimado en el que el sistema alcanza el establecimiento(pasa el transitorio)
idx=primer_valor(Diferencial,Valor_dif)#primer valor en el que el diferencial vale menos de "Valor_dif"


sigma=4/t[idx]#factor de decrecimiento
E_calculada=(sigma**2/(wp**2+sigma**2))**0.5#coeficiente de amortiguamiento "xi"
Wn_calculada=sigma/E_calculada

Kp_calcu=y[len(y)-1]*Wn_calculada**2#ganancia estática
#por deficicón de ft de segundo orden se establece el numerador y denominador como:
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

sys4=signal.cont2discrete((A,B,C,D),dt=Ts,method='zoh')#ecuacion en espacio de estados discreta
num3,den3=signal.ss2tf(sys4[0],sys4[1],sys4[2],sys4[3])#numerador y denominador funcion de transferencia en discreto

sys5= signal.TransferFunction(num3,den3,dt=Ts)#funcion de trasnferencia discreta
#%%
"Bloque 4.2: pasar funcion de trasnferencia continua a funcion de trasnferencia discreta"

H_cont =control.tf( num2 , den2 )
#Pasar de función continua a discreta
H_disc=control.sample_system( H_cont , Ts , method = 'zoh' )#para usar esta función H_cont debe ser una función obtenida mediante "control.tf"
#%%
"Bloque 5: Calcular K última y el valor del controlador P"
K_ultima=(sys5.den[len(sys5.den)-1]-1)/(-sys5.num[len(sys5.num)-1])
Kc=K_ultima/2
#%%
"Bloque 6:obtener el valor del controlador integral Ti"

#Se obtienen los polos en bucle cerrado de un controlador P con valor K última
a_den=sys5.den[0]
b_den=sys5.den[1]
c_den=sys5.den[2]
a_num=sys5.num[0]
b_num=sys5.num[1]
#a_den*z^2+(b_den+Kc*a_num)z+(c_den+Kc*b_num)   ecuacion general
# a        b                         c           coeficientes

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
"Bloque 7:formar la función de trasnferencia del PI"
Kc=K_ultima/2  
Ti=Ti          
n=len(t)

C_PI_cont= control.tf([Kc*Ti,Kc],[Ti,0])#funcion de trasnferencia continua del PI
C_PI_dis=control.sample_system( C_PI_cont , Ts , method = 'zoh' )#funcion de trasnferencia discreta del proporcional
C_PI_dis_num=[Kc,-Kc*((Ti-Ts)/Ti)]
C_PI_dis_den=[1,-1]

#C_PI_dis=signal.TransferFunction(C_PI_dis_num,C_PI_dis_den,dt=Ts)

Hz_PI=(C_PI_dis*H_disc)/(1+C_PI_dis*H_disc)#ecuancion en bucle cerrado

C_PI_dis_num=np.array(Hz_PI.num[0][0])
C_PI_dis_den=np.array(Hz_PI.den[0][0])



C_PI_dis_sys= signal.TransferFunction(C_PI_dis_num,C_PI_dis_den,dt=Ts)

t_C_PI_dis,y_C_PI_dis= signal.dstep(C_PI_dis_sys,t=t,n=n)
#%%
"Bloque 8: PID tuner"
Kcc=Kc
Tii=Ti

Kc_min = 0.00000000    # the minimial value of the paramater a
Kc_max = Kc*2   # the maximal value of the paramater a
Kc_init = Kc

Ti_min = 0.0000000000
Ti_max=Ti*2
Ti_init=Ti

fig = plt.figure(figsize=(15,10))

#se crean las barras generales de la figura con dos objetos:plot y slicer
Kp_ax = plt.axes([0.1, 0.2, 0.8, 0.65])
slider_ax = plt.axes([0.1, 0.05, 0.8, 0.05])
slider_ax2 = plt.axes([0.1, 0.001, 0.8, 0.05])

# en plot_ax se crea dibuja la función con el valor inicial para Ti y Kc
plt.axes(Kp_ax) 
plt.title('PI Tuner App')
Kp_tuner, = plt.plot(t_C_PI_dis, y_C_PI_dis[:][0], 'r',label='Respuesta PI')
plt.grid(True)
plt.xlim(0, t_C_PI_dis[len(t_C_PI_dis)-1])
plt.ylim(min(y)-2, max(y)+2)
plt.legend(loc='best')
plt.xlabel('Tiempo(s)')
plt.ylabel('Velocidad')

# se crean los slider para Ti y Kc
Kc_slider = Slider(slider_ax,      # the axes object containing the slider
                  'Kc',            # the name of the slider parameter
                  Kc_min,          # minimal value of the parameter
                  Kc_max,          # maximal value of the parameter
                  valinit=Kc_init  # initial value of the parameter
                 )
Ti_slider = Slider(slider_ax2,      # the axes object containing the slider
                  'Ti',            # the name of the slider parameter
                  Ti_min,          # minimal value of the parameter
                  Ti_max,          # maximal value of the parameter
                  valinit=Ti_init  # initial value of the parameter
                 )
    
#cada vez que hay un cambio en el slider se llama a esta función
Kc_slider.on_changed(update_Kc)
Ti_slider.on_changed(update_Ti)

plt.show()

