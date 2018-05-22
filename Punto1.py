# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 18:17:01 2017

@author: palom
"""



from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import datetime
import pandas as pd
import scipy.stats
import scipy
from scipy import stats


Pathout= 'C:/Users/palom/Documents/AnalisisDatos/Tarea02/'

#series aleatorias
S1=np.random.uniform(low=1, high=10, size=20)
S2=np.random.uniform(low=1, high=10, size=50)
S3=np.random.uniform(low=1, high=10, size=100)
S4=np.random.uniform(low=1, high=10, size=1000)
S5=np.random.uniform(low=1, high=10, size=10000)

valorextremo= np.concatenate((S1,np.array([50])))
valorextremo2= np.concatenate((S2,np.array([50])))
valorextremo3=np.concatenate((S3,np.array([50])))
valorextremo4= np.concatenate((S4,np.array([50])))
valorextremo5= np.concatenate((S5,np.array([50])))



#media
S1_mean=np.mean(S1)
S2_mean=np.mean(S2)
S3_mean=np.mean(S3)
S4_mean=np.mean(S4)
S5_mean=np.mean(S5)
Ext_mean=np.mean(valorextremo)
Ext_mean2=np.mean(valorextremo2)
Ext_mean3=np.mean(valorextremo3)
Ext_mean4=np.mean(valorextremo4)
Ext_mean5=np.mean(valorextremo5)


#mediana
S1_medi=np.median(S1)
S2_medi=np.median(S2)
S3_medi=np.median(S3)
S4_medi=np.median(S4)
S5_medi=np.median(S5)
Ext_medi=np.median(valorextremo)
Ext_medi2=np.median(valorextremo2)
Ext_medi3=np.median(valorextremo3)
Ext_medi4=np.median(valorextremo4)
Ext_medi5=np.median(valorextremo5)



p=np.percentile(S1,50)

#desviacion estandar
S1_std=np.std(S1)
S2_std=np.std(S2)
S3_std=np.std(S3)
S4_std=np.std(S4)
S5_std=np.std(S5)


#Kurtosis
S1_kur=scipy.stats.kurtosis(S1)
S2_kur=scipy.stats.kurtosis(S2)
S3_kur=scipy.stats.kurtosis(S3)
S4_kur=scipy.stats.kurtosis(S4)
S5_kur=scipy.stats.kurtosis(S5)


#Percentiles

S1_perc_25=np.percentile(S1,25)
S1_perc_75=np.percentile(S1,75)
S2_perc_25=np.percentile(S2,25)
S2_perc_75=np.percentile(S2,75)
S3_perc_25=np.percentile(S3,25)
S3_perc_75=np.percentile(S3,75)
S4_perc_25=np.percentile(S4,25)
S4_perc_75=np.percentile(S4,75)
S5_perc_25=np.percentile(S5,25)
S5_perc_75=np.percentile(S5,75)


#Deciles
DecilesS1=[]
DecilesS2=[]
DecilesS3=[]
DecilesS4=[]
DecilesS5=[]

    
for i in range(10,100,10):
    DecilesS1.append(np.percentile(S1,i))    
    DecilesS2.append(np.percentile(S2,i))   
    DecilesS3.append(np.percentile(S3,i))   
    DecilesS4.append(np.percentile(S4,i))   
    DecilesS5.append(np.percentile(S5,i))
    
    
#Rango intercuartil
IQR1 = S1_perc_75-S1_perc_25
IQR2 = S2_perc_75-S2_perc_25
IQR3 = S3_perc_75-S3_perc_25
IQR4 = S4_perc_75-S4_perc_25
IQR5 = S5_perc_75-S5_perc_25



from statsmodels import robust

#Desviacion absoluta media

MAD1=robust.mad(S1)
MAD2=robust.mad(S2)
MAD3=robust.mad(S3)
MAD4=robust.mad(S4)
MAD5=robust.mad(S5)

#Trimedia

def Trimd(percentil25, mediana, percentil75):
    Trimedia= ((percentil25) + (2*mediana) + (percentil75))/4
    return Trimedia    
    
    
TriM1=Trimd (S1_perc_25, S1_medi, S1_perc_75)    
TriM2=Trimd (S2_perc_25, S2_medi, S2_perc_75)  
TriM3=Trimd (S3_perc_25, S3_medi, S3_perc_75)  
TriM4=Trimd (S4_perc_25, S4_medi, S4_perc_75)  
TriM5=Trimd (S5_perc_25, S5_medi, S5_perc_75)     
    
#TriM_5=stats.tmean(S5) si calculo la trimedia con este m'odulo da un poco diferente
    
#Simetria
    
Skew1=stats.skew(S1)
Skew2=stats.skew(S2)
Skew3=stats.skew(S3)
Skew4=stats.skew(S4)
Skew5=stats.skew(S5)



yk1 = (S1_perc_25 - 2*(S1_medi) +S1_perc_75) / (S1_perc_75- S1_perc_25)
yk2 = (S2_perc_25 - 2*(S2_medi) +S2_perc_75) / (S2_perc_75- S2_perc_25)
yk3 = (S3_perc_25 - 2*(S3_medi) +S3_perc_75) / (S3_perc_75- S3_perc_25)
yk4 = (S4_perc_25 - 2*(S4_medi) +S4_perc_75) / (S4_perc_75- S4_perc_25)
yk5 = (S5_perc_25 - 2*(S5_medi) +S5_perc_75) / (S5_perc_75- S5_perc_25)



#--------------------------



S1_mean=np.array(S1_mean)
S2_mean=np.array(S2_mean)
S3_mean=np.array(S3_mean)
S4_mean=np.array(S4_mean)
S5_mean=np.array(S5_mean)




S1_stdx=np.full(20,S1_std)
S1_meanx=np.full(20,S1_mean)

S3_stdx=np.full(100,S3_std)    #para crear un vector con el mismo valor siempre
S3_meanx=np.full(100,S3_mean)

S4_stdx=np.full(1000,S4_std)    #para crear un vector con el mismo valor siempre
S4_meanx=np.full(1000,S4_mean)

#--------------Figuras-----------



fig=plt.figure(figsize=[10,24])
ax= fig.add_subplot(311)

plt.plot(range(1,21),S1, color= 'b', )
plt.axhline(y=S1_mean, color='m', linestyle='-', label='Media') 
plt.axhline(y=S1_medi, color=('olive'), label = 'Mediana')  
#plt.axhline(y=Ext_mean, color='k', linestyle='-', label='Media') 
#plt.axhline(y=Ext_medi, color=('r'), label = 'Mediana')  
plt.axhline(y=S1_perc_25, color=('gold'), label = 'Cuartiles')  
plt.axhline(y=S1_perc_75, color=('gold')) 
plt.plot(range(1,21),S1_meanx + S1_stdx,'skyblue',label=u'Desviación Estandar', alpha=1)
plt.plot(range(1,21),S1_meanx - S1_stdx,'skyblue')
plt.fill_between( range(1,21),S1_meanx - S1_stdx,S1_meanx + S1_stdx,color='skyblue', alpha=0.1)
plt.xlim(1,20)
plt.ylabel(u'Valores', fontsize=15)
plt.xlabel(u'Número de datos', fontsize=15)
plt.title('Serie 1', fontsize=20)
plt.legend(prop={'size': 10},fontsize=15)
#plt.savefig(Pathout+'Serie1'+'.png')



#plt.figure(figsize=[10,5])
ax= fig.add_subplot(312)
plt.plot(range(1,101),S3, color= 'b', )
plt.axhline(y=S3_mean, color='m', linestyle='-', label='Media') 
plt.axhline(y=S3_medi, color=('olive'), label = 'Mediana')   
plt.axhline(y=S3_perc_25, color=('gold'), label = 'Cuartiles')  
plt.axhline(y=S3_perc_75, color=('gold')) 
plt.plot(range(1,101),S3_meanx + S3_stdx,'skyblue',label=u'Desviación Estandar', alpha=1)
plt.plot(range(1,101),S3_meanx - S3_stdx,'skyblue')
plt.fill_between(range(1,101), S3_meanx - S3_stdx,S3_meanx + S3_stdx,color='skyblue', alpha=0.1)
plt.xlim(1,100)
plt.ylabel(u'Valores', fontsize=15)
plt.xlabel(u'Número de datos', fontsize=15)
plt.title('Serie 3', fontsize=20)
plt.legend(prop={'size': 10},fontsize=15)
#plt.savefig(Pathout+'Serie3'+'.png')


#plt.figure(figsize=[10,5])
ax= fig.add_subplot(313)
plt.plot(range(1,1001),S4, color= 'b', )
plt.axhline(y=S4_mean, color='m', linestyle='-', label='Media') 
plt.axhline(y=S4_medi, color=('olive'), label = 'Mediana')  
plt.axhline(y=S1_perc_25, color=('gold'), label = 'Cuartiles')  
plt.axhline(y=S1_perc_75, color=('gold'))  
plt.plot(range(1,1001),S4_meanx + S4_stdx,'skyblue',label=u'Desviación Estandar', alpha=1)
plt.plot(range(1,1001),S4_meanx - S4_stdx,'skyblue')
plt.fill_between(range(1,1001), S4_meanx - S4_stdx,S4_meanx + S4_stdx,color='skyblue', alpha=0.1)
plt.xlim(1,1000)
plt.ylabel(u'Valores', fontsize=15)
plt.xlabel(u'Número de datos', fontsize=15)
plt.title('Serie 4', fontsize=20)
plt.legend(prop={'size': 10},fontsize=15)
plt.savefig(Pathout+'Seriespunto1'+'.png')



#histograma S1-------------------------------------------------


num_bins=10
hist, bins=np.histogram(S1, bins=num_bins)
hist=hist.astype(float)
pdf=hist.astype(float)
pdf=hist/np.sum(hist)

print len(bins)
print len(pdf)
print pdf
print np.sum(pdf)*np.diff(bins)[0]
print bins
print np.diff(bins)

fig=plt.figure()
ax= fig.add_subplot(111)
temp=pdf/np.sum(pdf)
print temp

bincenters=(bins[1:]+bins[:-1])/2
plt.plot(bincenters, pdf, 'o-', color='b', lw=2, markersize=10 )
for xy in DecilesS1:
    plt.axvline(x=xy, color='mediumblue')
plt.axvline(x=DecilesS1[0], color='mediumblue', label='Deciles')    
plt.axvline(x=S1_perc_25, color='r', label='Q1')
plt.axvline(x=S1_perc_75, color='c',label='Q2')
plt.axvline(x=S1_medi, color='y', label = 'Mediana')   
plt.ylabel(u'Probabilidad', fontsize=15)
plt.xlabel(u'Valores', fontsize=15)
plt.title(u'Histograma Serie 1', fontsize=20)
axes = plt.gca()
#axes.set_xlim([xmin,xmax])
axes.set_ylim([0.0,0.5])
plt.legend()

#histograma S2-----------------------------------------------


num_bins=10
hist, bins=np.histogram(S2, bins=num_bins)
hist=hist.astype(float)
pdf=hist.astype(float)
pdf=hist/np.sum(hist)

#print len(bins)
#print len(pdf)
#print pdf
#print np.sum(pdf)*np.diff(bins)[0]
#print bins
#print np.diff(bins)

fig=plt.figure(figsize=[12,28])
ax= fig.add_subplot(411)
temp=pdf/np.sum(pdf)
print temp

bincenters=(bins[1:]+bins[:-1])/2
plt.plot(bincenters, pdf, 'o-', color='b', lw=2, markersize=10 )
for xy in DecilesS2:
       plt.axvline(x=xy, color='skyblue', )  
plt.axvline(x=DecilesS2[0], color='skyblue', label='Deciles')  
plt.axvline(x=S2_perc_25, color='gold', label='Cuartiles',linewidth= 2)
plt.axvline(x=S2_perc_75, color='gold',linewidth= 2)
plt.axvline(x=S2_medi, color='deeppink', label = 'Mediana',linewidth= 2)  
plt.ylabel(u'Probabilidad', fontsize=15)
plt.xlabel(u'Valores', fontsize=15) 
plt.title(u'Histograma Serie 2', fontsize=20)
axes = plt.gca()
#axes.set_xlim([xmin,xmax])
axes.set_ylim([0.0,0.18])
plt.legend()

#histograma serie3--------------------------------------------------
num_bins=10
hist, bins=np.histogram(S3, bins=num_bins)
hist=hist.astype(float)
pdf=hist.astype(float)
pdf=hist/np.sum(hist)

#print len(bins)
#print len(pdf)
#print pdf
#print np.sum(pdf)*np.diff(bins)[0]
#print bins
#print np.diff(bins)

#fig=plt.figure()
ax= fig.add_subplot(412)
temp=pdf/np.sum(pdf)
print temp

bincenters=(bins[1:]+bins[:-1])/2
plt.plot(bincenters, pdf, 'o-', color='b', lw=2, markersize=10 )
for xy in DecilesS3:
    plt.axvline(x=xy, color='skyblue', )  
plt.axvline(x=DecilesS3[0], color='skyblue', label='Deciles')  
plt.axvline(x=S3_perc_25, color='gold', label='Cuartiles',linewidth= 2)
plt.axvline(x=S3_perc_75, color='gold',linewidth= 2)
plt.axvline(x=S3_medi, color='deeppink', label = 'Mediana',linewidth= 2)   
plt.ylabel(u'Probabilidad', fontsize=15)
plt.xlabel(u'Valores', fontsize=15)
plt.title(u'Histograma Serie 3 ', fontsize=20)
axes = plt.gca()
#axes.set_xlim([xmin,xmax])
axes.set_ylim([0.0,0.18])
plt.legend()

#histograma serie4------------------------------------
num_bins=10
hist, bins=np.histogram(S4, bins=num_bins)
hist=hist.astype(float)
pdf=hist.astype(float)
pdf=hist/np.sum(hist)

#print len(bins)
#print len(pdf)
#print pdf
#print np.sum(pdf)*np.diff(bins)[0]
#print bins
#print np.diff(bins)

#fig=plt.figure(figsize=[20,5])
ax= fig.add_subplot(413)
temp=pdf/np.sum(pdf)
print temp

bincenters=(bins[1:]+bins[:-1])/2
plt.plot(bincenters, pdf, 'o-', color='b', lw=2, markersize=10 )
for xy in DecilesS4:
    plt.axvline(x=xy, color='skyblue', )  
plt.axvline(x=DecilesS4[0], color='skyblue', label='Deciles')  
plt.axvline(x=S4_perc_25, color='gold', label='Cuartiles',linewidth= 2)
plt.axvline(x=S4_perc_75, color='gold',linewidth= 2)
plt.axvline(x=S4_medi, color='deeppink', label = 'Mediana',linewidth= 2)   
plt.ylabel(u'Probabilidad', fontsize=15)
plt.xlabel(u'Valores', fontsize=15)
plt.title(u'Histograma Serie 4 ', fontsize=20)
axes = plt.gca()
#axes.set_xlim([xmin,xmax])
axes.set_ylim([0.0,0.18])
plt.legend()

#histograma serie5-----------------------------------------------
num_bins=10
hist, bins=np.histogram(S5, bins=num_bins)
hist=hist.astype(float)
pdf=hist.astype(float)
pdf=hist/np.sum(hist)

#print len(bins)
#print len(pdf)
#print pdf
#print np.sum(pdf)*np.diff(bins)[0]
#print bins
#print np.diff(bins)

#fig=plt.figure()
ax= fig.add_subplot(414)
temp=pdf/np.sum(pdf)
print temp

bincenters=(bins[1:]+bins[:-1])/2
plt.plot(bincenters, pdf, 'o-', color='b', lw=2, markersize=10 )
for xy in DecilesS5:
    plt.axvline(x=xy, color='skyblue',)   
plt.axvline(x=DecilesS5[0], color='skyblue', label='Deciles')  
plt.axvline(x=S5_perc_25, color='gold', label='Cuartiles', linewidth= 2)
plt.axvline(x=S5_perc_75, color='gold', linewidth= 2)
plt.axvline(x=S5_medi, color='deeppink', label = 'Mediana', linewidth= 2)   
plt.ylabel(u'Probabilidad', fontsize=15)
plt.xlabel(u'Valores', fontsize=15)
plt.title(u'Histograma Serie 5', fontsize=20)
axes = plt.gca()
#axes.set_xlim([xmin,xmax])
axes.set_ylim([0.0,0.18])
plt.legend()
plt.savefig(Pathout+'Histogramas'+'.png')




ejeX=[1, 2, 3, 4, 5]
ejeX=np.array(ejeX)

Medias=[S1_mean,S2_mean,S3_mean,S4_mean,S5_mean]
Medianas=[S1_medi,S2_medi,S3_medi,S4_medi,S5_medi]
Desviaciones=[S1_std,S2_std,S3_std,S4_std,S5_std]
Trimedias=[TriM1, TriM2, TriM3, TriM4, TriM5]
IQR=[IQR1, IQR2, IQR3, IQR4, IQR5]
MAD=[MAD1, MAD2, MAD3, MAD4, MAD5]
Asimetria=[Skew1, Skew2, Skew3, Skew4, Skew5]
YK=[yk1, yk2, yk3, yk4, yk5]
Curtosis=[S1_kur, S2_kur, S3_kur,S4_kur, S5_kur]
extremo=[Ext_mean, Ext_mean2,Ext_mean3,Ext_mean4,Ext_mean5]
extremomd=[Ext_medi, Ext_medi2,Ext_medi3,Ext_medi4,Ext_medi5]

#estadisticos de localizacion 

plt.figure(figsize=[10,7])
plt.rcParams.update({'font.size':14})
plt.plot(ejeX,Medias,'-', color= 'tomato',label ='Media')
plt.plot(ejeX,Medianas,'-', color= 'b', label='Mediana')
plt.plot(ejeX,Trimedias,'-', color= 'aqua', label='Trimedia')
plt.plot(ejeX,extremo,'-', color= 'lime', label='Media valor extremo')
plt.plot(ejeX,extremomd,'-', color= 'orange', label='Mediana valor extremo')
ax=plt.gca()#grab the current axis
ax.set_xticks([1,2,3,4,5])#choose which x locations to have ticks
ax.set_xticklabels(['20','50','100','1000','10000','10000' ]) #set the labels to display at those ticks
plt.title(u'Medidas de Localización', fontsize=20)
plt.xlabel(u'Número de datos en la serie',fontsize=15)
#plt.ylabel('',fontsize=18)
plt.legend(loc=1)
plt.savefig(Pathout+'Medidas de localizacion'+'.png')
#plt.close('all')


#estadisticos de variabilidad
plt.figure(figsize=[10,7])
plt.rcParams.update({'font.size':14})
plt.plot(ejeX,MAD,'-', color= 'crimson',label ='MAD')
plt.plot(ejeX,IQR,'-', color= 'navy', label='IQR')
plt.plot(ejeX,Desviaciones,'-', color= 'lime', label=u'Desviación Estandar')
ax=plt.gca()#grab the current axis
ax.set_xticks([1,2,3,4,5])#choose which x locations to have ticks
ax.set_xticklabels(['20','50','100','1000','10000','10000' ]) #set the labels to display at those ticks
plt.title(u'Medidas de Extensión', fontsize=20)
plt.xlabel(u'Número de datos en la serie',fontsize=15)
#plt.ylabel('',fontsize=18)
plt.legend(loc=1)
plt.savefig(Pathout+'Medidas de variabilidad'+'.png')
#plt.close('all')


#estadisticos de forma
plt.figure(figsize=[10,7])
plt.rcParams.update({'font.size':14})
plt.plot(ejeX,YK,'-', color= 'deeppink',label ='YK')
plt.plot(ejeX, Asimetria,'-', color= 'darkblue', label=u'Asimetría')
plt.plot(ejeX,Curtosis,'-', color= 'c', label=u'Curtosis')
ax=plt.gca()#grab the current axis
ax.set_xticks([1,2,3,4,5])#choose which x locations to have ticks
ax.set_xticklabels(['20','50','100','1000','10000','10000' ]) #set the labels to display at those ticks
plt.title(u'Medidas de Forma', fontsize=20)
plt.xlabel(u'Número de datos en la serie',fontsize=15)
#plt.ylabel('',fontsize=18)
plt.legend(loc=1)
plt.savefig(Pathout+'Medidas de forma'+'.png')
#plt.close('all')









