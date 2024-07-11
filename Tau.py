#%% Analisis para calculo de tau - levantando de archivo resultados.txt y de ciclo
import numpy as np
import matplotlib.pyplot as plt
import fnmatch
import os
import pandas as pd
import chardet 
import re
from scipy.interpolate import interp1d
from uncertainties import ufloat, unumpy 
from scipy.optimize import curve_fit 
from scipy.stats import linregress
from uncertainties import ufloat, unumpy

#%% LECTOR RESULTADOS
def lector_resultados(path): 
    '''
    Para levantar archivos de resultados con columnas :
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	SAR_(W/g)	Tau_(s)	N	xi_M_0
    '''
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']
        
    # Leer las primeras 6 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(6):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                match = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                if match:
                    key = match.group(1)[2:]
                    value = float(match.group(2))
                    meta[key] = value
                    
    # Leer los datos del archivo
    data = pd.read_table(path, header=14,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)
        
    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.to_datetime(data['Time_m'][:],dayfirst=True)
    # delta_t = np.array([dt.total_seconds() for dt in (time-time[0])])
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
     
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)
   
    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)
    
    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,  Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N
#%% LECTOR CICLOS
def lector_ciclos(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()[:6]

    metadata = {'filename': os.path.split(filepath)[-1],
                'Temperatura':float(lines[0].strip().split('_=_')[1]),
        "Concentracion_g/m^3": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[5].strip().split('_=_')[1].split(' ')[0])}
    
    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,3,4),
                        decimal='.',engine='python',
                        dtype={'Tiempo_(s)':'float','Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})  
    t= pd.Series(data['Tiempo_(s)']).to_numpy()
    H = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)*1000 #A/m
    M= pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m
    
    return t,H,M,metadata
#%% TAU PROMEDIO
def Tau_promedio(filepath,recorto_extremos=20):
    '''Dado un path, toma archivo de ciclo M vs H
     Calcula Magnetizacion de Equilibrio, y Tau pesado con dM/dH
     '''
    t,H,M,meta=lector_ciclos(filepath)
     
    indx_max= np.nonzero(H==max(H))[0][0]
    t_mag = t[recorto_extremos:indx_max-recorto_extremos]
    H_mag = H[recorto_extremos:indx_max-recorto_extremos]
    M_mag = M[recorto_extremos:indx_max-recorto_extremos]

    H_demag = H[indx_max+recorto_extremos:-recorto_extremos] 
    # H_demag = np.concatenate((H_demag[:],H_mag[0:1]))

    M_demag = M[indx_max+recorto_extremos:-recorto_extremos]
    # M_demag = np.concatenate((M_demag[:],M_mag[0:1]))

    #INTERPOLACION de M 
    # Verificar que H_mag esté dentro del rango de H_demag
    #H_mag = H_mag[(H_mag >= min(H_demag)) & (H_mag <= max(H_demag))]

    # INTERPOLACION de M solo para los valores dentro del rango
    interpolador = interp1d(H_demag, M_demag,fill_value="extrapolate")
    M_demag_int = interpolador(H_mag)

    # interpolador=interp1d(H_demag, M_demag)
    # M_demag_int = interpolador(H_mag) 
    
    # Derivadas
    dMdH_mag = np.gradient(M_mag,H_mag)
    dMdH_demag_int = np.gradient(M_demag_int,H_mag)
    dHdt= np.gradient(H_mag,t_mag)

    Meq = (M_mag*dMdH_demag_int + M_demag_int*dMdH_mag)/(dMdH_mag+ dMdH_demag_int)
    dMeqdH = np.gradient(Meq,H_mag)

    Tau = (Meq - M_mag)/(dMdH_mag*dHdt )

    Tau_prom = np.sum(Tau*dMeqdH)/np.sum(dMdH_mag)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    #%paso a kA/m y ns
    H_mag/=1e3
    H_demag/=1e3
    Tau *=1e9
    Tau_prom*=1e9
    print(meta['filename'])
    print(Tau_prom,'s')

    fig,(ax1,ax2) = plt.subplots(nrows=2,figsize=(7,6),constrained_layout=True)
    #ax1.plot(H,Tau,'-',label='U')
    ax1.plot(H_mag,Tau,'.-')
    ax1.grid()
    ax1.set_xlabel('H (kA/m)')
    ax1.set_ylabel(r'$\tau$ (s)')
    ax1.text(1/2,1/7,rf'<$\tau$> = {Tau_prom:.1f} ns',ha='center',va='center',
             bbox=dict(alpha=0.8),transform=ax1.transAxes,fontsize=11)

    ax1.grid()
    ax1.set_xlabel('H (A/m)')
    ax1.set_ylabel('$\\tau$ (ns)')
    ax1.set_title(r'$\tau$ vs H', loc='left')
    ax1.grid()

    ax2.plot(H_mag,Meq,'-',label='M$_{equilibrio}$')
    ax2.plot(H_mag,M_mag,label='Mag')
    ax2.plot(H_demag,M_demag,label='Demag')
    ax2.grid()
    ax2.legend()
    ax2.set_title('M vs H', loc='left')
    ax2.set_xlabel('H (kA/m)')
    ax2.set_ylabel('M (A/m)')

    axins = ax2.inset_axes([0.6, 0.12, 0.39, 0.4])
    axins.plot(H_mag,Meq,'.-')
    axins.plot(H_mag, M_mag,'.-')
    axins.plot(H_demag,M_demag,'.-')
    axins.set_xlim(-0.1*max(H_mag),0.1*max(H_mag)) 
    axins.set_ylim(-0.1*max(M_mag),0.1*max(M_mag))
    ax2.indicate_inset_zoom(axins, edgecolor="black")
    axins.grid()
    plt.suptitle(meta['filename'])

    return Meq , H_mag, max(H)/1000, Tau , Tau_prom , fig
#%% 135 05
identif_0='135_20'
dir_0 = os.path.join(os.getcwd(),identif_0)
archivos_resultados = [f for f in os.listdir(dir_0) if  fnmatch.fnmatch(f, '*resultados*')]
archivos_resultados.sort()
filepaths = [os.path.join(dir_0,f) for f in archivos_resultados]
meta_1,files_1,time_1,temperatura_0_1,Mr_1,Hc_1,campo_max_1,mag_max_1,xi_M_0_1,frecuencia_fund_1,magnitud_fund_0_1,dphi_fem_0_1,SAR_0_1,tau_0_1,N1 = lector_resultados(filepaths[0])
meta_2,files_2,time_2,temperatura_0_2,Mr_2,Hc_2,campo_max_2,mag_max_2,xi_M_0_2,frecuencia_fund_2,magnitud_fund_0_2,dphi_fem_0_2,SAR_0_2,tau_0_2,N2 = lector_resultados(filepaths[1])
meta_3,files_3,time_3,temperatura_0_3,Mr_3,Hc_3,campo_max_3,mag_max_3,xi_M_0_3,frecuencia_fund_3,magnitud_fund_0_3,dphi_fem_0_3,SAR_0_3,tau_0_3,N3 = lector_resultados(filepaths[2])

taus_0=np.array([ufloat(np.mean(tau_0_1),np.std(tau_0_1)),ufloat(np.mean(tau_0_2),np.std(tau_0_2)),ufloat(np.mean(tau_0_3),np.std(tau_0_3))])*1e9
SARs_0=np.array([ufloat(np.mean(SAR_0_1),np.std(SAR_0_1)),ufloat(np.mean(SAR_0_2),np.std(SAR_0_2)),ufloat(np.mean(SAR_0_3),np.std(SAR_0_3))])

for i,ar in enumerate(archivos_resultados):
    print('File:',ar,f'- tau: {taus_0[i]:.2f} ns',f'- SAR: {SARs_0[i]:.1f} W/g')
ufloat(np.mean([t.nominal_value for t in taus_0]),np.std([t.nominal_value for t in taus_0]))

print(f'\nPromedio de las {len(taus_0)} medidas:')
tau0 = np.mean(unumpy.uarray([np.mean(tau_0_1),np.mean(tau_0_2),np.mean(tau_0_3)],[np.std(tau_0_1),np.std(tau_0_2),np.std(tau_0_3)]))*1e9
print(f' tau = {tau0} ns')
SAR0 = ufloat(np.mean([S.nominal_value for S in SARs_0]),np.std([S.nominal_value for S in SARs_0]))
print(f' SAR = {SAR0:.2uf} W/g')

fig,ax= plt.subplots(figsize=(9,5),constrained_layout=True)
ax.plot(temperatura_0_1,tau_0_1,'.-',label='1')
ax.plot(temperatura_0_2,tau_0_2,'.-',label='2')
ax.plot(temperatura_0_3,tau_0_3,'.-',label='3')
ax.text(0.95,0.1,rf'<$\tau$> = {tau0:.1uf} ns',bbox=dict(alpha=0.8),transform=ax.transAxes,ha='right', va='bottom')
plt.legend()
plt.grid()
plt.ylabel(r'$\tau$ (s)')
plt.xlabel('T (°C)')
plt.title(r'$\tau$ - '+ identif_0)
plt.savefig('tau_vs_T_'+identif_0+'.png',dpi=300)
plt.show()

archivos_ciclos_0 = [f for f in os.listdir(dir_0) if  fnmatch.fnmatch(f, '*promedio*')]
archivos_ciclos_0.sort()
filepaths_0 = [os.path.join(dir_0,f) for f in archivos_ciclos_0]
for ac in archivos_ciclos_0:
    print(ac)

fig1,ax1=plt.subplots(figsize=(8,6),constrained_layout=True)

for i,fp in enumerate(filepaths_0):
    t,H,M,metadata=lector_ciclos(fp)
    ax1.plot(H,M,label=f'{SARs_0[i]:1f} W/g')

ax1.text(0.95,0.1,f'<SAR> = {SAR0:.2uf} W/g',bbox=dict(alpha=0.8),transform=ax1.transAxes,ha='right', va='bottom')
ax1.set_ylabel('M (A/m)')
ax1.set_xlabel('H (A/m)')
ax1.legend()
ax1.grid()
plt.title('Ciclos promedio - '+identif_0)
plt.savefig('ciclos_promedio_'+identif_0+'.png',dpi=300)
plt.show()

#%% 135 10 
identif_1='135_38'
dir_1 = os.path.join(os.getcwd(),identif_1)
archivos_resultados = [f for f in os.listdir(dir_1) if  fnmatch.fnmatch(f, '*resultados*')]
archivos_resultados.sort()
filepaths = [os.path.join(dir_1,f) for f in archivos_resultados]
meta_1,files_1,time_1,temperatura_1_1,Mr_1,Hc_1,campo_max_1,mag_max_1,xi_M_0_1,frecuencia_fund_1,magnitud_fund_1_1,dphi_fem_1_1,SAR_1_1,tau_1_1,N1 = lector_resultados(filepaths[0])
meta_2,files_2,time_2,temperatura_1_2,Mr_2,Hc_2,campo_max_2,mag_max_2,xi_M_0_2,frecuencia_fund_2,magnitud_fund_1_2,dphi_fem_1_2,SAR_1_2,tau_1_2,N2 = lector_resultados(filepaths[1])
meta_3,files_3,time_3,temperatura_1_3,Mr_3,Hc_3,campo_max_3,mag_max_3,xi_M_0_3,frecuencia_fund_3,magnitud_fund_1_3,dphi_fem_1_3,SAR_1_3,tau_1_3,N3 = lector_resultados(filepaths[2])

taus_1=np.array([ufloat(np.mean(tau_1_1),np.std(tau_1_1)),ufloat(np.mean(tau_1_2),np.std(tau_1_2)),ufloat(np.mean(tau_1_3),np.std(tau_1_3))])*1e9
SARs_1=np.array([ufloat(np.mean(SAR_1_1),np.std(SAR_1_1)),ufloat(np.mean(SAR_1_2),np.std(SAR_1_2)),ufloat(np.mean(SAR_1_3),np.std(SAR_1_3))])

for i,ar in enumerate(archivos_resultados):
    print('File:',ar,f'- tau: {taus_1[i]:.2f} ns',f'- SAR: {SARs_1[i]:.1f} W/g')
ufloat(np.mean([t.nominal_value for t in taus_1]),np.std([t.nominal_value for t in taus_1]))

print(f'\nPromedio de las {len(taus_1)} medidas:')
tau1 = np.mean(unumpy.uarray([np.mean(tau_1_1),np.mean(tau_1_2),np.mean(tau_1_3)],[np.std(tau_1_1),np.std(tau_1_2),np.std(tau_1_3)]))*1e9
print(f' tau = {tau1} ns')
SAR1 = ufloat(np.mean([S.nominal_value for S in SARs_1]),np.std([S.nominal_value for S in SARs_1]))
print(f' SAR = {SAR1:.2uf} W/g')

fig,ax= plt.subplots(figsize=(9,5),constrained_layout=True)
ax.plot(temperatura_1_1,tau_1_1,'.-',label='1')
ax.plot(temperatura_1_2,tau_1_2,'.-',label='2')
ax.plot(temperatura_1_3,tau_1_3,'.-',label='3')
ax.text(0.95,0.1,rf'<$\tau$> = {tau1:.1uf} ns',bbox=dict(alpha=0.8),transform=ax.transAxes,ha='right', va='bottom')
plt.legend()
plt.grid()
plt.ylabel(r'$\tau$ (s)')
plt.xlabel('T (°C)')
plt.title(r'$\tau$ - '+ identif_1)
plt.savefig('tau_vs_T_'+identif_1+'.png',dpi=300)
plt.show()

archivos_ciclos_1 = [f for f in os.listdir(dir_1) if  fnmatch.fnmatch(f, '*promedio*')]
archivos_ciclos_1.sort()
filepaths_1 = [os.path.join(dir_1,f) for f in archivos_ciclos_1]
for ac in archivos_ciclos_1:
    print(ac)

fig1,ax1=plt.subplots(figsize=(8,6),constrained_layout=True)

for i,fp in enumerate(filepaths_1):
    t,H,M,metadata=lector_ciclos(fp)
    ax1.plot(H,M,label=f'{SARs_1[i]:1f} W/g')

ax1.text(0.95,0.1,f'<SAR> = {SAR1:.2uf} W/g',bbox=dict(alpha=0.8),transform=ax1.transAxes,ha='right', va='bottom')
ax1.set_ylabel('M (A/m)')
ax1.set_xlabel('H (A/m)')
ax1.legend()
ax1.grid()
plt.title('Ciclos promedio - '+identif_1)
plt.savefig('ciclos_promedio_'+identif_1+'.png',dpi=300)
plt.show()
#%% 135 15 
identif_2='135_57'
dir_2 = os.path.join(os.getcwd(),identif_2)
archivos_resultados = [f for f in os.listdir(dir_2) if  fnmatch.fnmatch(f, '*resultados*')]
archivos_resultados.sort()
filepaths = [os.path.join(dir_2,f) for f in archivos_resultados]
meta_1,files_1,time_1,temperatura_2_1,Mr_1,Hc_1,campo_max_1,mag_max_1,xi_M_0_1,frecuencia_fund_1,magnitud_fund_2_1,dphi_fem_2_1,SAR_2_1,tau_2_1,N1 = lector_resultados(filepaths[0])
meta_2,files_2,time_2,temperatura_2_2,Mr_2,Hc_2,campo_max_2,mag_max_2,xi_M_0_2,frecuencia_fund_2,magnitud_fund_2_2,dphi_fem_2_2,SAR_2_2,tau_2_2,N2 = lector_resultados(filepaths[1])
meta_3,files_3,time_3,temperatura_2_3,Mr_3,Hc_3,campo_max_3,mag_max_3,xi_M_0_3,frecuencia_fund_3,magnitud_fund_2_3,dphi_fem_2_3,SAR_2_3,tau_2_3,N3 = lector_resultados(filepaths[2])

taus_2=np.array([ufloat(np.mean(tau_2_1),np.std(tau_2_1)),ufloat(np.mean(tau_2_2),np.std(tau_2_2)),ufloat(np.mean(tau_2_3),np.std(tau_2_3))])*1e9
SARs_2=np.array([ufloat(np.mean(SAR_2_1),np.std(SAR_2_1)),ufloat(np.mean(SAR_2_2),np.std(SAR_2_2)),ufloat(np.mean(SAR_2_3),np.std(SAR_2_3))])

for i,ar in enumerate(archivos_resultados):
    print('File:',ar,f'- tau: {taus_2[i]:.2f} ns',f'- SAR: {SARs_2[i]:.1f} W/g')
ufloat(np.mean([t.nominal_value for t in taus_2]),np.std([t.nominal_value for t in taus_2]))

print(f'\nPromedio de las {len(taus_2)} medidas:')
tau2 = np.mean(unumpy.uarray([np.mean(tau_2_1),np.mean(tau_2_2),np.mean(tau_2_3)],[np.std(tau_2_1),np.std(tau_2_2),np.std(tau_2_3)]))*1e9
print(f' tau = {tau2} ns')
SAR2 = ufloat(np.mean([S.nominal_value for S in SARs_2]),np.std([S.nominal_value for S in SARs_2]))
print(f' SAR = {SAR2:.2uf} W/g')

fig,ax= plt.subplots(figsize=(9,5),constrained_layout=True)
ax.plot(temperatura_2_1,tau_2_1,'.-',label='1')
ax.plot(temperatura_2_2,tau_2_2,'.-',label='2')
ax.plot(temperatura_2_3,tau_2_3,'.-',label='3')
ax.text(0.95,0.1,rf'<$\tau$> = {tau1:.1uf} ns',bbox=dict(alpha=0.8),transform=ax.transAxes,ha='right', va='bottom')
plt.legend()
plt.grid()
plt.ylabel(r'$\tau$ (s)')
plt.xlabel('T (°C)')
plt.title(r'$\tau$ - '+ identif_2)
plt.savefig('tau_vs_T_'+identif_2+'.png',dpi=300)
plt.show()

archivos_ciclos_2 = [f for f in os.listdir(dir_2) if  fnmatch.fnmatch(f, '*promedio*')]
archivos_ciclos_2.sort()
filepaths_2 = [os.path.join(dir_2,f) for f in archivos_ciclos_2]
for ac in archivos_ciclos_2:
    print(ac)

fig1,ax1=plt.subplots(figsize=(8,6),constrained_layout=True)

for i,fp in enumerate(filepaths_2):
    t,H,M,metadata=lector_ciclos(fp)
    ax1.plot(H,M,label=f'{SARs_2[i]:1f} W/g')

ax1.text(0.95,0.1,f'<SAR> = {SAR2:.2uf} W/g',bbox=dict(alpha=0.8),transform=ax1.transAxes,ha='right', va='bottom')
ax1.set_ylabel('M (A/m)')
ax1.set_xlabel('H (A/m)')
ax1.legend()
ax1.grid()
plt.title('Ciclos promedio - '+identif_2)
plt.savefig('ciclos_promedio_'+identif_2+'.png',dpi=300)
plt.show()
#%% 265 20
identif_5='265_20'
dir_5 = os.path.join(os.getcwd(),identif_5)
archivos_resultados = [f for f in os.listdir(dir_5) if  fnmatch.fnmatch(f, '*resultados*')]
archivos_resultados.sort()
filepaths = [os.path.join(dir_5,f) for f in archivos_resultados]
meta_1,files_1,time_1,temperatura_5_1,Mr_1,Hc_1,campo_max_1,mag_max_1,xi_M_5_1,frecuencia_fund_1,magnitud_fund_5_1,dphi_fem_5_1,SAR_5_1,tau_5_1,N1 = lector_resultados(filepaths[0])
meta_2,files_2,time_2,temperatura_5_2,Mr_2,Hc_2,campo_max_2,mag_max_2,xi_M_5_2,frecuencia_fund_2,magnitud_fund_5_2,dphi_fem_5_2,SAR_5_2,tau_5_2,N2 = lector_resultados(filepaths[1])
meta_3,files_3,time_3,temperatura_5_3,Mr_3,Hc_3,campo_max_3,mag_max_3,xi_M_5_3,frecuencia_fund_3,magnitud_fund_5_3,dphi_fem_5_3,SAR_5_3,tau_5_3,N3 = lector_resultados(filepaths[2])

taus_5=np.array([ufloat(np.mean(tau_5_1),np.std(tau_5_1)),ufloat(np.mean(tau_5_2),np.std(tau_5_2)),ufloat(np.mean(tau_5_3),np.std(tau_5_3))])*1e9
SARs_5=np.array([ufloat(np.mean(SAR_5_1),np.std(SAR_5_1)),ufloat(np.mean(SAR_5_2),np.std(SAR_5_2)),ufloat(np.mean(SAR_5_3),np.std(SAR_5_3))])

for i,ar in enumerate(archivos_resultados):
    print('File:',ar,f'- tau: {taus_5[i]:.2f} ns',f'- SAR: {SARs_5[i]:.1f} W/g')
ufloat(np.mean([t.nominal_value for t in taus_5]),np.std([t.nominal_value for t in taus_5]))

print(f'\nPromedio de las {len(taus_5)} medidas:')
tau0 = np.mean(unumpy.uarray([np.mean(tau_5_1),np.mean(tau_5_2),np.mean(tau_5_3)],[np.std(tau_5_1),np.std(tau_5_2),np.std(tau_5_3)]))*1e9
print(f' tau = {tau0} ns')
SAR0 = ufloat(np.mean([S.nominal_value for S in SARs_5]),np.std([S.nominal_value for S in SARs_5]))
print(f' SAR = {SAR0:.2uf} W/g')

fig,ax= plt.subplots(figsize=(9,5),constrained_layout=True)
ax.plot(temperatura_5_1,tau_5_1,'.-',label='1')
ax.plot(temperatura_5_2,tau_5_2,'.-',label='2')
ax.plot(temperatura_5_3,tau_5_3,'.-',label='3')
ax.text(0.95,0.1,rf'<$\tau$> = {tau1:.1uf} ns',bbox=dict(alpha=0.8),transform=ax.transAxes,ha='right', va='bottom')
plt.legend()
plt.grid()
plt.ylabel(r'$\tau$ (s)')
plt.xlabel('T (°C)')
plt.title(r'$\tau$ - '+ identif_5)
plt.savefig('tau_vs_T_'+identif_5+'.png',dpi=300)
plt.show()

archivos_ciclos_5 = [f for f in os.listdir(dir_5) if  fnmatch.fnmatch(f, '*promedio*')]
archivos_ciclos_5.sort()
filepaths_5 = [os.path.join(dir_5,f) for f in archivos_ciclos_5]
for ac in archivos_ciclos_5:
    print(ac)

fig1,ax1=plt.subplots(figsize=(8,6),constrained_layout=True)

for i,fp in enumerate(filepaths_5):
    t,H,M,metadata=lector_ciclos(fp)
    ax1.plot(H,M,label=f'{SARs_5[i]:1f} W/g')

ax1.text(0.95,0.1,f'<SAR> = {SAR0:.2uf} W/g',bbox=dict(alpha=0.8),transform=ax1.transAxes,ha='right', va='bottom')
ax1.set_ylabel('M (A/m)')
ax1.set_xlabel('H (A/m)')
ax1.legend()
ax1.grid()
plt.title('Ciclos promedio - '+identif_5)
plt.savefig('ciclos_promedio_'+identif_5+'.png',dpi=300)
plt.show()



#%% 265 38 
identif_3='265_38'
dir_3 = os.path.join(os.getcwd(),identif_3)
archivos_resultados_3 = [f for f in os.listdir(dir_3) if  fnmatch.fnmatch(f, '*resultados*')]
archivos_resultados_3.sort()
filepaths = [os.path.join(dir_3,f) for f in archivos_resultados_3]

meta_1,files_1,time_1,temperatura_3_1,Mr_1,Hc_1,campo_max_1,mag_max_1,xi_M_0_1,frecuencia_fund_3_1,magnitud_fund_3_1,dphi_fem_3_1,SAR_3_1,tau_3_1,N1 = lector_resultados(filepaths[0])
meta_2,files_2,time_2,temperatura_3_2,Mr_2,Hc_2,campo_max_2,mag_max_2,xi_M_0_2,frecuencia_fund_3_2,magnitud_fund_3_2,dphi_fem_3_2,SAR_3_2,tau_3_2,N2 = lector_resultados(filepaths[1])
meta_3,files_3,time_3,temperatura_3_3,Mr_3,Hc_3,campo_max_3,mag_max_3,xi_M_0_3,frecuencia_fund_3_3,magnitud_fund_3_3,dphi_fem_3_3,SAR_3_3,tau_3_3,N3 = lector_resultados(filepaths[2])

taus_3=np.array([ufloat(np.mean(tau_3_1),np.std(tau_3_1)),ufloat(np.mean(tau_3_2),np.std(tau_3_2)),ufloat(np.mean(tau_3_3),np.std(tau_3_3))])*1e9
SARs_3=np.array([ufloat(np.mean(SAR_3_1),np.std(SAR_3_1)),ufloat(np.mean(SAR_3_2),np.std(SAR_3_2)),ufloat(np.mean(SAR_3_3),np.std(SAR_3_3))])

for i,ar in enumerate(archivos_resultados):
    print('File:',ar,f'- tau: {taus_3[i]:.2f} ns',f'- SAR: {SARs_3[i]:.1f} W/g')
ufloat(np.mean([t.nominal_value for t in taus_3]),np.std([t.nominal_value for t in taus_3]))

print(f'\nPromedio de las {len(taus_3)} medidas:')
tau3 = np.mean(unumpy.uarray([np.mean(tau_3_1),np.mean(tau_3_2),np.mean(tau_3_3)],[np.std(tau_3_1),np.std(tau_3_2),np.std(tau_3_3)]))*1e9
print(f' tau = {tau3} ns')
SAR3 = ufloat(np.mean([S.nominal_value for S in SARs_3]),np.std([S.nominal_value for S in SARs_3]))
print(f' SAR = {SAR3:.2uf} W/g')

fig,ax= plt.subplots(figsize=(9,5),constrained_layout=True)
ax.plot(temperatura_3_1,tau_3_1,'.-',label='1')
ax.plot(temperatura_3_2,tau_3_2,'.-',label='2')
ax.plot(temperatura_3_3,tau_3_3,'.-',label='3')
ax.text(0.95,0.1,rf'<$\tau$> = {tau3:.1uf} ns',bbox=dict(alpha=0.8),transform=ax.transAxes,ha='right', va='bottom')
plt.legend()
plt.grid()
plt.ylabel(r'$\tau$ (s)')
plt.xlabel('Indx')
plt.title(identif_3)
plt.savefig('tau_vs_T_'+identif_3+'.png',dpi=300)
plt.show()
#CICLOS
archivos_ciclos_3 = [f for f in os.listdir(dir_3) if  fnmatch.fnmatch(f, '*promedio*')]
archivos_ciclos_3.sort()
filepaths_3 = [os.path.join(dir_3,f) for f in archivos_ciclos_3]
for ac in archivos_ciclos_3:
    print(ac)

fig2,ax2=plt.subplots(figsize=(8,6),constrained_layout=True)
for i,fp in enumerate(filepaths_3):
    t,H,M,metadata=lector_ciclos(fp)
    ax2.plot(H,M,label=f'{SARs_3[i]:1f} W/g')
ax2.text(0.95,0.1,f'<SAR> = {SAR3:.2uf} W/g',bbox=dict(alpha=0.8),transform=ax2.transAxes,ha='right', va='bottom')

ax2.set_ylabel('M (A/m)')
ax2.set_xlabel('H (A/m)')
ax2.legend()
ax2.grid()
plt.title('Ciclos promedio - '+identif_3)
plt.savefig('ciclos_promedio_'+identif_3+'.png',dpi=300)
plt.show()
#%% 265 15
identif_4='265_57'
dir_4 = os.path.join(os.getcwd(),identif_4)
archivos_resultados = [f for f in os.listdir(dir_4) if  fnmatch.fnmatch(f, '*resultados*')]
archivos_resultados.sort()
filepaths = [os.path.join(dir_4,f) for f in archivos_resultados]

meta_1,files_1,time_1,temperatura_4_1,Mr_1,Hc_1,campo_max_1,mag_max_1,xi_M_0_1,frecuencia_fund_4_1,magnitud_fund_4_1,dphi_fem_4_1,SAR_4_1,tau_4_1,N1 = lector_resultados(filepaths[0])
meta_2,files_2,time_2,temperatura_4_2,Mr_2,Hc_2,campo_max_2,mag_max_2,xi_M_0_2,frecuencia_fund_4_2,magnitud_fund_4_2,dphi_fem_4_2,SAR_4_2,tau_4_2,N2 = lector_resultados(filepaths[1])
meta_3,files_3,time_3,temperatura_4_3,Mr_3,Hc_3,campo_max_3,mag_max_3,xi_M_0_3,frecuencia_fund_4_3,magnitud_fund_4_3,dphi_fem_4_3,SAR_4_3,tau_4_3,N3 = lector_resultados(filepaths[2])

taus_4=np.array([ufloat(np.mean(tau_4_1),np.std(tau_4_1)),ufloat(np.mean(tau_4_2),np.std(tau_4_2)),ufloat(np.mean(tau_4_3),np.std(tau_4_3))])*1e9
SARs_4=np.array([ufloat(np.mean(SAR_4_1),np.std(SAR_4_1)),ufloat(np.mean(SAR_4_2),np.std(SAR_4_2)),ufloat(np.mean(SAR_4_3),np.std(SAR_4_3))])

for i,ar in enumerate(archivos_resultados):
    print('File:',ar,f'- tau: {taus_4[i]:.2f} ns',f'- SAR: {SARs_4[i]:.1f} W/g')
ufloat(np.mean([t.nominal_value for t in taus_4]),np.std([t.nominal_value for t in taus_4]))

print(f'\nPromedio de las {len(taus_4)} medidas:')
tau4 = np.mean(unumpy.uarray([np.mean(tau_4_1),np.mean(tau_4_2),np.mean(tau_4_3)],[np.std(tau_4_1),np.std(tau_4_2),np.std(tau_4_3)]))*1e9
print(f' tau = {tau4} ns')
SAR4 = ufloat(np.mean([S.nominal_value for S in SARs_4]),np.std([S.nominal_value for S in SARs_4]))
print(f' SAR = {SAR4:.2uf} W/g')

fig,ax= plt.subplots(figsize=(9,5),constrained_layout=True)
ax.plot(temperatura_4_1,tau_4_1,'.-',label='1')
ax.plot(temperatura_4_2,tau_4_2,'.-',label='2')
ax.plot(temperatura_4_3,tau_4_3,'.-',label='3')
ax.text(0.95,0.1,rf'<$\tau$> = {tau4:.1uf} ns',bbox=dict(alpha=0.8),transform=ax.transAxes,ha='right', va='bottom')
plt.legend()
plt.grid()
plt.ylabel(r'$\tau$ (s)')
plt.xlabel('Indx')
plt.title(identif_4)
plt.savefig('tau_vs_T_'+identif_4+'.png',dpi=300)
plt.show()
#%CICLOS
archivos_ciclos_4 = [f for f in os.listdir(dir_4) if  fnmatch.fnmatch(f, '*promedio*')]
archivos_ciclos_4.sort()
filepaths_4 = [os.path.join(dir_4,f) for f in archivos_ciclos_4]
for ac in archivos_ciclos_4:
    print(ac)

fig2,ax2=plt.subplots(figsize=(8,6),constrained_layout=True)
for i,fp in enumerate(filepaths_4):
    t,H,M,metadata=lector_ciclos(fp)
    ax2.plot(H,M,label=f'{SARs_4[i]:1f} W/g')
ax2.text(0.95,0.1,f'<SAR> = {SAR4:.2uf} W/g',bbox=dict(alpha=0.8),transform=ax2.transAxes,ha='right', va='bottom')
ax2.set_ylabel('M (A/m)')
ax2.set_xlabel('H (A/m)')
ax2.legend()
ax2.grid()
plt.title('Ciclos promedio - '+identif_4)
plt.savefig('ciclos_promedio_'+identif_4+'.png',dpi=300)
plt.show()


#%% PLOTEO PARAMETROS alrededor de la Transicion de Fase
# 135 20
fig,((ax0,ax1,ax2),(ax3,ax4,ax5),(ax9,ax10,ax11),(ax6,ax7,ax8))= plt.subplots(ncols=3,nrows=4,figsize=(13,12),constrained_layout=True,sharex='col',sharey='row')
ax0.plot(temperatura_0_1,np.sin(dphi_fem_0_1),'.-',color='tab:blue',label='1')
ax1.plot(temperatura_0_2,np.sin(dphi_fem_0_2),'.-',color='tab:orange',label='2')
ax2.plot(temperatura_0_3,np.sin(dphi_fem_0_3),'.-',color='tab:green',label='3')

ax3.plot(temperatura_0_1,magnitud_fund_0_1,'.-',color='tab:blue',label='1')
ax4.plot(temperatura_0_2,magnitud_fund_0_2,'.-',color='tab:orange',label='2')
ax5.plot(temperatura_0_3,magnitud_fund_0_3,'.-',color='tab:green',label='3')

ax9.plot(temperatura_0_1,tau_0_1*1e9,'.-',color='tab:blue',label='1')
ax10.plot(temperatura_0_2,tau_0_2*1e9,'.-',color='tab:orange',label='2')
ax11.plot(temperatura_0_3,tau_0_3*1e9,'.-',color='tab:green',label='3')

ax6.plot(temperatura_0_1,SAR_0_1,'.-',color='tab:blue',label='1')
ax7.plot(temperatura_0_2,SAR_0_2,'.-',color='tab:orange',label='2')
ax8.plot(temperatura_0_3,SAR_0_3,'.-',color='tab:green',label='3')

ax0.set_ylabel('$\sin \Delta \phi$')
ax0.set_title('$\sin (\Delta \phi)$',loc='left')
ax3.set_ylabel('|c$_0$|')
ax3.set_title('Magnitud 1° armónico',loc='left')
ax6.set_title('SAR',loc='left')
ax6.set_ylabel('SAR (W/g)')
ax9.set_title('Tiempo de relajacion',loc='left')
ax9.set_ylabel(r'$\tau$ (ns)')
for a in [ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11]:
    a.set_xlim(-4,4)
    a.grid()
    a.legend()

for a in [ax6,ax7,ax8]:
    a.set_xlabel('T (°C)')

plt.suptitle(f'NE@citrico - {identif_0.split("_")[0]} kHz - {identif_0.split("_")[1]} kA/m',fontsize=16)
plt.savefig('sinphi_mag_SAR_tau_vs_T_'+identif_0+'.png',dpi=300)
plt.show()#%%

# 135 38
fig,((ax0,ax1,ax2),(ax3,ax4,ax5),(ax9,ax10,ax11),(ax6,ax7,ax8))= plt.subplots(ncols=3,nrows=4,figsize=(13,12),constrained_layout=True,sharex='col',sharey='row')
ax0.plot(temperatura_1_1,np.sin(dphi_fem_1_1),'.-',color='tab:blue',label='1')
ax1.plot(temperatura_1_2,np.sin(dphi_fem_1_2),'.-',color='tab:orange',label='2')
ax2.plot(temperatura_1_3,np.sin(dphi_fem_1_3),'.-',color='tab:green',label='3')

ax3.plot(temperatura_1_1,magnitud_fund_1_1,'.-',color='tab:blue',label='1')
ax4.plot(temperatura_1_2,magnitud_fund_1_2,'.-',color='tab:orange',label='2')
ax5.plot(temperatura_1_3,magnitud_fund_1_3,'.-',color='tab:green',label='3')

ax9.plot(temperatura_1_1,tau_1_1*1e9,'.-',color='tab:blue',label='1')
ax10.plot(temperatura_1_2,tau_1_2*1e9,'.-',color='tab:orange',label='2')
ax11.plot(temperatura_1_3,tau_1_3*1e9,'.-',color='tab:green',label='3')

ax6.plot(temperatura_1_1,SAR_1_1,'.-',color='tab:blue',label='1')
ax7.plot(temperatura_1_2,SAR_1_2,'.-',color='tab:orange',label='2')
ax8.plot(temperatura_1_3,SAR_1_3,'.-',color='tab:green',label='3')

ax0.set_ylabel('$\sin \Delta \phi$')
ax0.set_title('$\sin (\Delta \phi)$',loc='left')
ax3.set_ylabel('|c$_0$|')
ax3.set_title('Magnitud 1° armónico',loc='left')
ax6.set_title('SAR',loc='left')
ax6.set_ylabel('SAR (W/g)')
ax9.set_title('Tiempo de relajacion',loc='left')
ax9.set_ylabel(r'$\tau$ (ns)')
for a in [ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11]:
    a.set_xlim(-4,4)
    a.grid()
    a.legend()

for a in [ax6,ax7,ax8]:
    a.set_xlabel('T (°C)')

plt.suptitle(f'NE@citrico - {identif_1.split("_")[0]} kHz - {identif_1.split("_")[1]} kA/m',fontsize=16)
plt.savefig('sinphi_mag_SAR_tau_vs_T_'+identif_1+'.png',dpi=300)
plt.show()

#% 135 57
fig,((ax0,ax1,ax2),(ax3,ax4,ax5),(ax9,ax10,ax11),(ax6,ax7,ax8))= plt.subplots(ncols=3,nrows=4,figsize=(13,12),constrained_layout=True,sharex='col',sharey='row')
ax0.plot(temperatura_2_1,np.sin(dphi_fem_2_1),'.-',color='tab:blue',label='1')
ax1.plot(temperatura_2_2,np.sin(dphi_fem_2_2),'.-',color='tab:orange',label='2')
ax2.plot(temperatura_2_3,np.sin(dphi_fem_2_3),'.-',color='tab:green',label='3')

ax3.plot(temperatura_2_1,magnitud_fund_2_1,'.-',color='tab:blue',label='1')
ax4.plot(temperatura_2_2,magnitud_fund_2_2,'.-',color='tab:orange',label='2')
ax5.plot(temperatura_2_3,magnitud_fund_2_3,'.-',color='tab:green',label='3')

ax9.plot(temperatura_2_1,tau_2_1*1e9,'.-',color='tab:blue',label='1')
ax10.plot(temperatura_2_2,tau_2_2*1e9,'.-',color='tab:orange',label='2')
ax11.plot(temperatura_2_3,tau_2_3*1e9,'.-',color='tab:green',label='3')

ax6.plot(temperatura_2_1,SAR_2_1,'.-',color='tab:blue',label='1')
ax7.plot(temperatura_2_2,SAR_2_2,'.-',color='tab:orange',label='2')
ax8.plot(temperatura_2_3,SAR_2_3,'.-',color='tab:green',label='3')

ax0.set_ylabel('$\sin \Delta \phi$')
ax0.set_title('$\sin (\Delta \phi)$',loc='left')
ax3.set_ylabel('|c$_0$|')
ax3.set_title('Magnitud 1° armónico',loc='left')
ax6.set_title('SAR',loc='left')
ax6.set_ylabel('SAR (W/g)')
ax9.set_title('Tiempo de relajacion',loc='left')
ax9.set_ylabel(r'$\tau$ (ns)')
for a in [ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11]:
    a.set_xlim(-4,4)
    a.grid()
    a.legend()

for a in [ax6,ax7,ax8]:
    a.set_xlabel('T (°C)')

plt.suptitle(f'NE@citrico - {identif_2.split("_")[0]} kHz - {identif_2.split("_")[1]} kA/m',fontsize=16)
plt.savefig('sinphi_mag_SAR_tau_vs_T_'+identif_2+'.png',dpi=300)
plt.show()

#%265 38
fig,((ax0,ax1,ax2),(ax3,ax4,ax5),(ax9,ax10,ax11),(ax6,ax7,ax8))= plt.subplots(ncols=3,nrows=4,figsize=(13,12),constrained_layout=True,sharex='col',sharey='row')

ax0.plot(temperatura_3_1,np.sin(dphi_fem_3_1),'.-',color='tab:blue',label='1')
ax1.plot(temperatura_3_2,np.sin(dphi_fem_3_2),'.-',color='tab:orange',label='2')
ax2.plot(temperatura_3_3,np.sin(dphi_fem_3_3),'.-',color='tab:green',label='3')

ax3.plot(temperatura_3_1,magnitud_fund_3_1,'.-',color='tab:blue',label='1')
ax4.plot(temperatura_3_2,magnitud_fund_3_2,'.-',color='tab:orange',label='2')
ax5.plot(temperatura_3_3,magnitud_fund_3_3,'.-',color='tab:green',label='3')

ax9.plot(temperatura_3_1,tau_3_1*1e9,'.-',color='tab:blue',label='1')
ax10.plot(temperatura_3_2,tau_3_2*1e9,'.-',color='tab:orange',label='2')
ax11.plot(temperatura_3_3,tau_3_3*1e9,'.-',color='tab:green',label='3')

ax6.plot(temperatura_3_1,SAR_3_1,'.-',color='tab:blue',label='1')
ax7.plot(temperatura_3_2,SAR_3_2,'.-',color='tab:orange',label='2')
ax8.plot(temperatura_3_3,SAR_3_3,'.-',color='tab:green',label='3')

ax0.set_ylabel('$\sin \Delta \phi$')
ax0.set_title('$\sin (\Delta \phi)$',loc='left')
ax3.set_ylabel('|c$_0$|')
ax3.set_title('Magnitud 1° armónico',loc='left')
ax6.set_title('SAR',loc='left')
ax6.set_ylabel('SAR (W/g)')
ax9.set_title('Tiempo de relajacion',loc='left')
ax9.set_ylabel(r'$\tau$ (ns)')
for a in [ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11]:
    a.set_xlim(-4,4)
    a.grid()
    a.legend()

for a in [ax6,ax7,ax8]:
    a.set_xlabel('T (°C)')

plt.suptitle(f'NE@citrico - {identif_3.split("_")[0]} kHz - {identif_3.split("_")[1]} kA/m',fontsize=16)
plt.savefig('sinphi_mag_SAR_tau_vs_T_'+identif_3+'.png',dpi=300)
plt.show()
#% 265 57
fig,((ax0,ax1,ax2),(ax3,ax4,ax5),(ax9,ax10,ax11),(ax6,ax7,ax8))= plt.subplots(ncols=3,nrows=4,figsize=(13,12),constrained_layout=True,sharex='col',sharey='row')

ax0.plot(temperatura_4_1,np.sin(dphi_fem_4_1),'.-',color='tab:blue',label='1')
ax1.plot(temperatura_4_2,np.sin(dphi_fem_4_2),'.-',color='tab:orange',label='2')
ax2.plot(temperatura_4_3,np.sin(dphi_fem_4_3),'.-',color='tab:green',label='3')

ax3.plot(temperatura_4_1,magnitud_fund_4_1,'.-',color='tab:blue',label='1')
ax4.plot(temperatura_4_2,magnitud_fund_4_2,'.-',color='tab:orange',label='2')
ax5.plot(temperatura_4_3,magnitud_fund_4_3,'.-',color='tab:green',label='3')

ax9.plot(temperatura_4_1,tau_4_1*1e9,'.-',color='tab:blue',label='1')
ax10.plot(temperatura_4_2,tau_4_2*1e9,'.-',color='tab:orange',label='2')
ax11.plot(temperatura_4_3,tau_4_3*1e9,'.-',color='tab:green',label='3')

ax6.plot(temperatura_4_1,SAR_4_1,'.-',color='tab:blue',label='1')
ax7.plot(temperatura_4_2,SAR_4_2,'.-',color='tab:orange',label='2')
ax8.plot(temperatura_4_3,SAR_4_3,'.-',color='tab:green',label='3')

ax0.set_ylabel('$\sin \Delta \phi$')
ax0.set_title('$\sin (\Delta \phi)$',loc='left')
ax3.set_ylabel('|c$_0$|')
ax3.set_title('Magnitud 1° armónico',loc='left')
ax6.set_title('SAR',loc='left')
ax6.set_ylabel('SAR (W/g)')
ax9.set_title('Tiempo de relajacion',loc='left')
ax9.set_ylabel(r'$\tau$ (ns)')
for a in [ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11]:
    a.set_xlim(-4,4)
    a.grid()
    a.legend()

for a in [ax6,ax7,ax8]:
    a.set_xlabel('T (°C)')

plt.suptitle(f'NE@citrico - {identif_4.split("_")[0]} kHz - {identif_4.split("_")[1]} kA/m',fontsize=16)
plt.savefig('sinphi_mag_SAR_tau_vs_T_'+identif_4+'.png',dpi=300)
plt.show()

#% 265 20
fig,((ax0,ax1,ax2),(ax3,ax4,ax5),(ax9,ax10,ax11),(ax6,ax7,ax8))= plt.subplots(ncols=3,nrows=4,figsize=(13,12),constrained_layout=True,sharex='col',sharey='row')

ax0.plot(temperatura_5_1,np.sin(dphi_fem_5_1),'.-',color='tab:blue',label='1')
ax1.plot(temperatura_5_2,np.sin(dphi_fem_5_2),'.-',color='tab:orange',label='2')
ax2.plot(temperatura_5_3,np.sin(dphi_fem_5_3),'.-',color='tab:green',label='3')

ax3.plot(temperatura_5_1,magnitud_fund_5_1,'.-',color='tab:blue',label='1')
ax4.plot(temperatura_5_2,magnitud_fund_5_2,'.-',color='tab:orange',label='2')
ax5.plot(temperatura_5_3,magnitud_fund_5_3,'.-',color='tab:green',label='3')

ax9.plot(temperatura_5_1,tau_5_1*1e9,'.-',color='tab:blue',label='1')
ax10.plot(temperatura_5_2,tau_5_2*1e9,'.-',color='tab:orange',label='2')
ax11.plot(temperatura_5_3,tau_5_3*1e9,'.-',color='tab:green',label='3')

ax6.plot(temperatura_5_1,SAR_5_1,'.-',color='tab:blue',label='1')
ax7.plot(temperatura_5_2,SAR_5_2,'.-',color='tab:orange',label='2')
ax8.plot(temperatura_5_3,SAR_5_3,'.-',color='tab:green',label='3')

ax0.set_ylabel('$\sin \Delta \phi$')
ax0.set_title('$\sin (\Delta \phi)$',loc='left')
ax3.set_ylabel('|c$_0$|')
ax3.set_title('Magnitud 1° armónico',loc='left')
ax6.set_title('SAR',loc='left')
ax6.set_ylabel('SAR (W/g)')
ax9.set_title('Tiempo de relajacion',loc='left')
ax9.set_ylabel(r'$\tau$ (ns)')
for a in [ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11]:
    a.set_xlim(-4,4)
    a.grid()
    a.legend()

for a in [ax6,ax7,ax8]:
    a.set_xlabel('T (°C)')

plt.suptitle(f'NE@citrico - {identif_5.split("_")[0]} kHz - {identif_5.split("_")[1]} kA/m',fontsize=16)
plt.savefig('sinphi_mag_SAR_tau_vs_T_'+identif_5+'.png',dpi=300)
plt.show()

#%% Obtengo ciclos representativos
#primero los del pico en la transicion de fase
# print('max tau en TF:')
# indx_max_1= np.nonzero(tau_1_1==max(tau_1_1))[0][0]
# print(indx_max_1,files_1[indx_max_1],temperatura_1[indx_max_1])

# indx_max_2= np.nonzero(tau_1_2==max(tau_1_2))[0][0]
# print(indx_max_2,files_2[indx_max_2],temperatura_2[indx_max_2])

# indx_max_3= np.nonzero(tau_1_3==max(tau_1_3))[0][0]
# print(indx_max_3,files_3[indx_max_3],temperatura_3[indx_max_3])

#%% PLOT ALL TAUS
colors = [
    ['#00ace6','#0099cc','#0086b3'  ],  # tonos de azul
    ['#ff6600', '#e65c00', '#cc5200'],  # tonos de naranja
    ['#00cc44', '#00b33c', '#009933'],  # tonos de verde
    ['#ff0000', '#e60000', '#b30000'],  # tonos de rojo
    ['#ff00ff', '#cc00cc', '#990099'],  # tonos purpura
    ['#0059b3', '#004d99', '#004080'] # tonos azul oscuro
]

fig,(ax1,ax2)=plt.subplots(2,1,figsize=(14,8),sharex=True,constrained_layout=True)
#Tau

ax1.plot(temperatura_0_1, tau_0_1 * 1e9, 'o-', zorder=-1,label=identif_0, color=colors[0][0])
ax1.plot(temperatura_0_2, tau_0_2 * 1e9, 'o-', zorder=0,label=identif_0, color=colors[0][1])
ax1.plot(temperatura_0_3, tau_0_3 * 1e9, 'o-', zorder=1,label=identif_0, color=colors[0][2])

ax1.plot(temperatura_1_1, tau_1_1 * 1e9, 's-', zorder=-1,label=identif_1, color=colors[1][0])
ax1.plot(temperatura_1_2, tau_1_2 * 1e9, 's-', zorder=0,label=identif_1, color=colors[1][1])
ax1.plot(temperatura_1_3, tau_1_3 * 1e9, 's-', zorder=1,label=identif_1, color=colors[1][2])

ax1.plot(temperatura_2_1, tau_2_1 * 1e9, 'p-', zorder=-1,label=identif_2, color=colors[2][0])
ax1.plot(temperatura_2_2, tau_2_2 * 1e9, 'p-', zorder=0,label=identif_2, color=colors[2][1])
ax1.plot(temperatura_2_3, tau_2_3 * 1e9, 'p-', zorder=1,label=identif_2, color=colors[2][2])

ax1.plot(temperatura_5_1, tau_5_1 * 1e9, 'v-', zorder=-1,label=identif_5, color=colors[3][0])
ax1.plot(temperatura_5_2, tau_5_2 * 1e9, 'v-', zorder=0,label=identif_5, color=colors[3][1])
ax1.plot(temperatura_5_3, tau_5_3 * 1e9, 'v-', zorder=1,label=identif_5, color=colors[3][2])

ax1.plot(temperatura_3_1, tau_3_1 * 1e9, '^-', zorder=-1,label=identif_3, color=colors[4][0])
ax1.plot(temperatura_3_2, tau_3_2 * 1e9, '^-', zorder=0,label=identif_3, color=colors[4][1])
ax1.plot(temperatura_3_3, tau_3_3 * 1e9, '^-', zorder=1,label=identif_3, color=colors[4][2])

ax1.plot(temperatura_4_1, tau_4_1 * 1e9, 'P-', zorder=-1,label=identif_4, color=colors[5][0])
ax1.plot(temperatura_4_2, tau_4_2 * 1e9, 'P-', zorder=0,label=identif_4, color=colors[5][1])
ax1.plot(temperatura_4_3, tau_4_3 * 1e9, 'P-', zorder=1,label=identif_4, color=colors[5][2])

#SAR

ax2.plot(temperatura_0_1, SAR_0_1, 'o-', zorder=1,label=identif_0, color=colors[0][0])
ax2.plot(temperatura_0_2, SAR_0_2, 'o-', zorder=2,label=identif_0, color=colors[0][1])
ax2.plot(temperatura_0_3, SAR_0_3, 'o-', zorder=3,label=identif_0, color=colors[0][2])

ax2.plot(temperatura_1_1, SAR_1_1, 's-', zorder=1,label=identif_1, color=colors[1][0])
ax2.plot(temperatura_1_2, SAR_1_2, 's-', zorder=2,label=identif_1, color=colors[1][1])
ax2.plot(temperatura_1_3, SAR_1_3, 's-', zorder=3,label=identif_1, color=colors[1][2])

ax2.plot(temperatura_2_1, SAR_2_1, 'v-', zorder=1,label=identif_2, color=colors[2][0])
ax2.plot(temperatura_2_2, SAR_2_2, 'v-', zorder=2,label=identif_2, color=colors[2][1])
ax2.plot(temperatura_2_3, SAR_2_3, 'v-', zorder=3,label=identif_2, color=colors[2][2])

ax2.plot(temperatura_5_1, SAR_5_1, 'p-', zorder=1,label=identif_5, color=colors[3][0])
ax2.plot(temperatura_5_2, SAR_5_2, 'p-', zorder=2,label=identif_5, color=colors[3][1])
ax2.plot(temperatura_5_3, SAR_5_3, 'p-', zorder=3,label=identif_5, color=colors[3][2])

ax2.plot(temperatura_3_1, SAR_3_1, '^-', zorder=1,label=identif_3, color=colors[4][0])
ax2.plot(temperatura_3_2, SAR_3_2, '^-', zorder=2,label=identif_3, color=colors[4][1])
ax2.plot(temperatura_3_3, SAR_3_3, '^-', zorder=3,label=identif_3, color=colors[4][2])

ax2.plot(temperatura_4_1, SAR_4_1, 'P-', zorder=1,label=identif_4, color=colors[5][0])
ax2.plot(temperatura_4_2, SAR_4_2, 'P-', zorder=2,label=identif_4, color=colors[5][1])
ax2.plot(temperatura_4_3, SAR_4_3, 'P-', zorder=3,label=identif_4, color=colors[5][2])

ax2.set_xticks(np.arange(-50,30,5))

# ax1.axhline(tau1.nominal_value,0,1,c='tab:red',label=f'{tau1} ns')
# ax1.axhspan(tau1.nominal_value-tau1.std_dev,tau1.nominal_value+tau1.std_dev,alpha=0.5,color='tab:red')
ax1.set_ylabel(r'$\tau$ (ns)')  
ax2.set_ylabel('SAR (W/g)')

ax1.set_title('Tiempo de relajación',loc='left')
ax2.set_title('SAR',loc='left')
ax2.set_xlabel('T (°C)')
plt.suptitle('NE@citrico 5X - congelado c/Campo Perpendicular',fontsize=15)
for ax in [ax1,ax2]:
    #ax.set_ylabel(r'$\tau$ (ns)')
    ax.legend(ncol=6)
    ax.grid(zorder=0)

plt.savefig('tau_SAR_vs_T__all_cong_c_campo_P.png',dpi=300)


#%% AJUSTES TAU 
#%% 265 kHz 57 kA/m   CC vs SC
identif_4_sc='265_57_CsC'
dir_4_sc = os.path.join(os.getcwd(),identif_4_sc)
archivos_resultados = [f for f in os.listdir(dir_4_sc) if  fnmatch.fnmatch(f, '*resultados*')]
archivos_resultados.sort()
filepaths_sc = [os.path.join(dir_4_sc,f) for f in archivos_resultados]

_,files_4_1_sc,time_4_1_sc,temp_4_1_sc,_,_,_,_,_,_,_,_,SAR_4_1_sc,tau_4_1_sc,_ = lector_resultados(filepaths_sc[0])
_,files_4_2_sc,time_4_2_sc,temp_4_2_sc,_,_,_,_,_,_,_,_,SAR_4_2_sc,tau_4_2_sc,_ = lector_resultados(filepaths_sc[1])
_,files_4_3_sc,time_4_3_sc,temp_4_3_sc,_,_,_,_,_,_,_,_,SAR_4_3_sc,tau_4_3_sc,_ = lector_resultados(filepaths_sc[2])

tau_4_1_sc=tau_4_1_sc*1e9
tau_4_2_sc=tau_4_2_sc*1e9
tau_4_3_sc=tau_4_3_sc*1e9

identif_4_cc='265_57_CcC'
dir_4_cc = os.path.join(os.getcwd(),identif_4_cc)
archivos_resultados = [f for f in os.listdir(dir_4_cc) if  fnmatch.fnmatch(f, '*resultados*')]
archivos_resultados.sort()
filepaths_cc = [os.path.join(dir_4_cc,f) for f in archivos_resultados]

_,files_4_1_cc,time_4_1_cc,temp_4_1_cc,_,_,_,_,_,_,_,_,SAR_4_1_cc,tau_4_1_cc,_ = lector_resultados(filepaths_cc[0])
_,files_4_2_cc,time_4_2_cc,temp_4_2_cc,_,_,_,_,_,_,_,_,SAR_4_2_cc,tau_4_2_cc,_ = lector_resultados(filepaths_cc[1])
_,files_4_3_cc,time_4_3_cc,temp_4_3_cc,_,_,_,_,_,_,_,_,SAR_4_3_cc,tau_4_3_cc,_ = lector_resultados(filepaths_cc[2])

tau_4_1_cc=tau_4_1_cc*1e9
tau_4_2_cc=tau_4_2_cc*1e9
tau_4_3_cc=tau_4_3_cc*1e9

temp_all_sc = np.concatenate((temp_4_1_sc, temp_4_2_sc, temp_4_3_sc))
tau_all_sc = np.concatenate((tau_4_1_sc, tau_4_2_sc, tau_4_3_sc))

temp_all_cc = np.concatenate((temp_4_1_cc, temp_4_2_cc, temp_4_3_cc))
tau_all_cc = np.concatenate((tau_4_1_cc, tau_4_2_cc, tau_4_3_cc))
#%
fig,ax= plt.subplots(figsize=(9,5),constrained_layout=True)

# ax.plot(temp_all_sc,tau_all_sc,'o-',label='1 sc')
# ax.plot(temp_all_cc,tau_all_cc,'o-',label='1 cc')
ax.plot(temp_4_1_sc,tau_4_1_sc,'o-',c='tab:blue',label='1 sc')
ax.plot(temp_4_2_sc,tau_4_2_sc,'o-',c='tab:blue',label='2 sc')
ax.plot(temp_4_3_sc,tau_4_3_sc,'o-',c='tab:blue',label='3 sc')

ax.plot(temp_4_1_cc,tau_4_1_cc,'s-',c='tab:orange',label='1 H$_{⟂}$')
ax.plot(temp_4_2_cc,tau_4_2_cc,'s-',c='tab:orange',label='2 H$_{⟂}$')
ax.plot(temp_4_3_cc,tau_4_3_cc,'s-',c='tab:orange',label='3 H$_{⟂}$')

plt.legend(ncol=2)
plt.grid()
plt.ylabel(r'$\tau$ (s)')
plt.xlabel('Temperatura (ºC)')
plt.title(identif_4_sc +' - '+identif_4_cc)
plt.xlim(-10,10)
plt.savefig('tau_vs_T_comparativa_'+identif_4_sc+' '+identif_4_cc + '_zoom.png',dpi=300)
plt.show()
#%
def lineal(x,a,b):
    return a*x+b

lim_ajuste_i= -6
lim_ajuste_tf = -4.0
lim_ajuste_f= 0
#% SC
x_1_i_sc=temp_4_1_sc[np.nonzero(temp_4_1_sc<=lim_ajuste_i)]
x_1_tf_sc=temp_4_1_sc[np.nonzero((temp_4_1_sc>lim_ajuste_tf) & (temp_4_1_sc<lim_ajuste_f))]
x_1_f_sc=temp_4_1_sc[np.nonzero(temp_4_1_sc>=lim_ajuste_f)]

y_1_i_sc=tau_4_1_sc[np.nonzero(temp_4_1_sc<=lim_ajuste_i)]
y_1_tf_sc=tau_4_1_sc[np.nonzero((temp_4_1_sc>lim_ajuste_tf) & (temp_4_1_sc<lim_ajuste_f))]
y_1_f_sc=tau_4_1_sc[np.nonzero(temp_4_1_sc>=lim_ajuste_f)]

x_2_i_sc=temp_4_2_sc[np.nonzero(temp_4_2_sc<=lim_ajuste_i)]
x_2_tf_sc=temp_4_2_sc[np.nonzero((temp_4_2_sc>lim_ajuste_tf) & (temp_4_2_sc<lim_ajuste_f))]
x_2_f_sc=temp_4_2_sc[np.nonzero(temp_4_2_sc>=lim_ajuste_f)]

y_2_i_sc=tau_4_2_sc[np.nonzero(temp_4_2_sc<=lim_ajuste_i)]
y_2_tf_sc=tau_4_2_sc[np.nonzero((temp_4_2_sc>lim_ajuste_tf) & (temp_4_2_sc<lim_ajuste_f))]
y_2_f_sc=tau_4_2_sc[np.nonzero(temp_4_2_sc>=lim_ajuste_f)]

x_3_i_sc=temp_4_3_sc[np.nonzero(temp_4_3_sc<=lim_ajuste_i)]
x_3_tf_sc=temp_4_3_sc[np.nonzero((temp_4_3_sc>lim_ajuste_tf) & (temp_4_3_sc<lim_ajuste_f))]
x_3_f_sc=temp_4_3_sc[np.nonzero(temp_4_3_sc>=lim_ajuste_f)]

y_3_i_sc=tau_4_3_sc[np.nonzero(temp_4_3_sc<=lim_ajuste_i)]
y_3_tf_sc=tau_4_3_sc[np.nonzero((temp_4_3_sc>lim_ajuste_tf) & (temp_4_3_sc<lim_ajuste_f))]
y_3_f_sc=tau_4_3_sc[np.nonzero(temp_4_3_sc>=lim_ajuste_f)]

x_all_i_sc = np.concatenate((x_1_i_sc, x_2_i_sc, x_3_i_sc))
y_all_i_sc = np.concatenate((y_1_i_sc, y_2_i_sc, y_3_i_sc))
x_all_tf_sc = np.concatenate((x_1_tf_sc, x_2_tf_sc, x_3_tf_sc))
y_all_tf_sc = np.concatenate((y_1_tf_sc, y_2_tf_sc, y_3_tf_sc))
x_all_f_sc = np.concatenate((x_1_f_sc, x_2_f_sc, x_3_f_sc))
y_all_f_sc = np.concatenate((y_1_f_sc, y_2_f_sc, y_3_f_sc))

# Realizar el ajuste lineal
slope_i_sc, intercept_i_sc, r_value_i_sc, p_value_i_sc, std_err_i_sc = linregress(x_all_i_sc, y_all_i_sc)
slope_tf_sc, intercept_tf_sc, r_value_tf_sc, p_value_tf_sc, std_err_tf_sc = linregress(x_all_tf_sc, y_all_tf_sc)
slope_f_sc, intercept_f_sc, r_value_f_sc, p_value_f_sc, std_err_f_sc = linregress(x_all_f_sc, y_all_f_sc)
a_i_sc = ufloat(slope_i_sc,std_err_i_sc)
a_tf_sc = ufloat(slope_tf_sc,std_err_tf_sc)
a_f_sc = ufloat(slope_f_sc,std_err_f_sc)

# Mostrar los resultados del ajuste lineal
print('FF congelado SIN campo')
print(f'Pendiente_i: {a_i_sc}') 
print(f'Pendiente_tf: {a_tf_sc}')
print(f'Pendiente_f: {a_f_sc}')

x_sc_i=np.linspace(-50,lim_ajuste_i,1000)
y_sc_i=lineal(x_sc_i,slope_i_sc,intercept_i_sc)
x_sc_tf=np.linspace(lim_ajuste_tf,lim_ajuste_f,1000)
y_sc_tf=lineal(x_sc_tf,slope_tf_sc,intercept_tf_sc)
x_sc_f=np.linspace(lim_ajuste_f,25,1000)
y_sc_f=lineal(x_sc_f,slope_f_sc,intercept_f_sc)

#% CC
x_1_i_cc=temp_4_1_cc[np.nonzero(temp_4_1_cc<=lim_ajuste_i)]
x_1_tf_cc=temp_4_1_cc[np.nonzero((temp_4_1_cc>lim_ajuste_tf) & (temp_4_1_cc<lim_ajuste_f))]
x_1_f_cc=temp_4_1_cc[np.nonzero(temp_4_1_cc>=lim_ajuste_f)]

y_1_i_cc=tau_4_1_cc[np.nonzero(temp_4_1_cc<=lim_ajuste_i)]
y_1_tf_cc=tau_4_1_cc[np.nonzero((temp_4_1_cc>lim_ajuste_tf) & (temp_4_1_cc<lim_ajuste_f))]
y_1_f_cc=tau_4_1_cc[np.nonzero(temp_4_1_cc>=lim_ajuste_f)]

x_2_i_cc=temp_4_2_cc[np.nonzero(temp_4_2_cc<=lim_ajuste_i)]
x_2_tf_cc=temp_4_2_cc[np.nonzero((temp_4_2_cc>lim_ajuste_tf) & (temp_4_2_cc<lim_ajuste_f))]
x_2_f_cc=temp_4_2_cc[np.nonzero(temp_4_2_cc>=lim_ajuste_f)]

y_2_i_cc=tau_4_2_cc[np.nonzero(temp_4_2_cc<=lim_ajuste_i)]
y_2_tf_cc=tau_4_2_cc[np.nonzero((temp_4_2_cc>lim_ajuste_tf) & (temp_4_2_cc<lim_ajuste_f))]
y_2_f_cc=tau_4_2_cc[np.nonzero(temp_4_2_cc>=lim_ajuste_f)]

x_3_i_cc=temp_4_3_cc[np.nonzero(temp_4_3_cc<=lim_ajuste_i)]
x_3_tf_cc=temp_4_3_cc[np.nonzero((temp_4_3_cc>lim_ajuste_tf) & (temp_4_3_cc<lim_ajuste_f))]
x_3_f_cc=temp_4_3_cc[np.nonzero(temp_4_3_cc>=lim_ajuste_f)]

y_3_i_cc=tau_4_3_cc[np.nonzero(temp_4_3_cc<=lim_ajuste_i)]
y_3_tf_cc=tau_4_3_cc[np.nonzero((temp_4_3_cc>lim_ajuste_tf) & (temp_4_3_cc<lim_ajuste_f))]
y_3_f_cc=tau_4_3_cc[np.nonzero(temp_4_3_cc>=lim_ajuste_f)]

x_all_i_cc = np.concatenate((x_1_i_cc, x_2_i_cc, x_3_i_cc))
y_all_i_cc = np.concatenate((y_1_i_cc, y_2_i_cc, y_3_i_cc))
x_all_tf_cc = np.concatenate((x_1_tf_cc, x_2_tf_cc, x_3_tf_cc))
y_all_tf_cc = np.concatenate((y_1_tf_cc, y_2_tf_cc, y_3_tf_cc))
x_all_f_cc = np.concatenate((x_1_f_cc, x_2_f_cc, x_3_f_cc))
y_all_f_cc = np.concatenate((y_1_f_cc, y_2_f_cc, y_3_f_cc))

# Realizar el ajuste lineal
slope_i_cc, intercept_i_cc, r_value_i_cc, p_value_i_cc, std_err_i_cc = linregress(x_all_i_cc, y_all_i_cc)
slope_tf_cc, intercept_tf_cc, r_value_tf_cc, p_value_tf_cc, std_err_tf_cc = linregress(x_all_tf_cc, y_all_tf_cc)
slope_f_cc, intercept_f_cc, r_value_f_cc, p_value_f_cc, std_err_f_cc = linregress(x_all_f_cc, y_all_f_cc)
a_i_cc = ufloat(slope_i_cc,std_err_i_cc)
a_tf_cc = ufloat(slope_tf_cc,std_err_tf_cc)
a_f_cc = ufloat(slope_f_cc,std_err_f_cc)

# Mostrar los resultados del ajuste lineal
print('FF congelado CON campo')
print(f'Pendiente_i: {a_i_cc}') 
print(f'Pendiente_tf: {a_tf_cc}')
print(f'Pendiente_f: {a_f_cc}')

x_cc_i=np.linspace(-50,lim_ajuste_i,1000)
y_cc_i=lineal(x_cc_i,slope_i_cc,intercept_i_cc)
x_cc_tf=np.linspace(lim_ajuste_tf,lim_ajuste_f,1000)
y_cc_tf=lineal(x_cc_tf,slope_tf_cc,intercept_tf_cc)
x_cc_f=np.linspace(lim_ajuste_f,25,1000)
y_cc_f=lineal(x_cc_f,slope_f_cc,intercept_f_cc)

#%
fig,ax= plt.subplots(figsize=(9,5),constrained_layout=True)
ax.plot(x_all_i_sc, y_all_i_sc, 'o',label=f'i: [{min(x_all_i_sc)} ; {lim_ajuste_i}]°C')
ax.plot(x_sc_i,y_sc_i,label=f'{a_i_sc:.1uf} ns/°C')
ax.plot(x_all_tf_sc, y_all_tf_sc,'o', label=f'tf: [{lim_ajuste_i} ; {lim_ajuste_f}]°C')
ax.plot(x_sc_tf,y_sc_tf,label=f'{a_tf_sc:.1uf} ns/°C')
ax.plot(x_all_f_sc, y_all_f_sc, 'o',label=f'f: [{lim_ajuste_f} ; {max(x_all_f_sc)}]°C')
ax.plot(x_sc_f,y_sc_f,label=f'{a_f_sc:.1uf} ns/°C')

ax.plot(x_all_i_cc, y_all_i_cc, 'o',label=f'i: [{min(x_all_i_cc)} ; {lim_ajuste_i}]°C')
ax.plot(x_cc_i,y_cc_i,label=f'{a_i_cc:.1uf} ns/°C')
ax.plot(x_all_tf_cc, y_all_tf_cc,'o', label=f'tf: [{lim_ajuste_i} ; {lim_ajuste_f}]°C')
ax.plot(x_cc_tf,y_cc_tf,label=f'{a_tf_cc:.1uf} ns/°C')
ax.plot(x_all_f_cc, y_all_f_cc, 'o',label=f'f: [{lim_ajuste_f} ; {max(x_all_f_cc)}]°C')
ax.plot(x_cc_f,y_cc_f,label=f'{a_f_cc:.1uf} ns/°C')

plt.legend(ncol=2)
plt.grid()
plt.ylabel(r'$\tau$ (s)')
plt.xlabel('Temperatura (°C)')
plt.title(identif_4_sc + identif_4_sc)
plt.savefig('tau_vs_T_ajustes.png',dpi=300)
plt.show()

x_1_i_cc=temp_4_1_cc[np.nonzero(temp_4_1_cc<=lim_ajuste_i)]
x_1_tf_cc=temp_4_1_cc[np.nonzero((temp_4_1_cc>lim_ajuste_tf) & (temp_4_1_cc<lim_ajuste_f))]
x_1_f_cc=temp_4_1_cc[np.nonzero(temp_4_1_cc>=lim_ajuste_f)]

y_1_i_cc=tau_4_1_cc[np.nonzero(temp_4_1_cc<=lim_ajuste_i)]
y_1_tf_cc=tau_4_1_cc[np.nonzero((temp_4_1_cc>lim_ajuste_tf) & (temp_4_1_cc<lim_ajuste_f))]
y_1_f_cc=tau_4_1_cc[np.nonzero(temp_4_1_cc>=lim_ajuste_f)]

x_2_i_cc=temp_4_2_cc[np.nonzero(temp_4_2_cc<=lim_ajuste_i)]
x_2_tf_cc=temp_4_2_cc[np.nonzero((temp_4_2_cc>lim_ajuste_tf) & (temp_4_2_cc<lim_ajuste_f))]
x_2_f_cc=temp_4_2_cc[np.nonzero(temp_4_2_cc>=lim_ajuste_f)]

y_2_i_cc=tau_4_2_cc[np.nonzero(temp_4_2_cc<=lim_ajuste_i)]
y_2_tf_cc=tau_4_2_cc[np.nonzero((temp_4_2_cc>lim_ajuste_tf) & (temp_4_2_cc<lim_ajuste_f))]
y_2_f_cc=tau_4_2_cc[np.nonzero(temp_4_2_cc>=lim_ajuste_f)]

x_3_i_cc=temp_4_3_cc[np.nonzero(temp_4_3_cc<=lim_ajuste_i)]
x_3_tf_cc=temp_4_3_cc[np.nonzero((temp_4_3_cc>lim_ajuste_tf) & (temp_4_3_cc<lim_ajuste_f))]
x_3_f_cc=temp_4_3_cc[np.nonzero(temp_4_3_cc>=lim_ajuste_f)]

y_3_i_cc=tau_4_3_cc[np.nonzero(temp_4_3_cc<=lim_ajuste_i)]
y_3_tf_cc=tau_4_3_cc[np.nonzero((temp_4_3_cc>lim_ajuste_tf) & (temp_4_3_cc<lim_ajuste_f))]
y_3_f_cc=tau_4_3_cc[np.nonzero(temp_4_3_cc>=lim_ajuste_f)]

x_all_i_cc = np.concatenate((x_1_i_cc, x_2_i_cc, x_3_i_cc))
y_all_i_cc = np.concatenate((y_1_i_cc, y_2_i_cc, y_3_i_cc))
x_all_tf_cc = np.concatenate((x_1_tf_cc, x_2_tf_cc, x_3_tf_cc))
y_all_tf_cc = np.concatenate((y_1_tf_cc, y_2_tf_cc, y_3_tf_cc))
x_all_f_cc = np.concatenate((x_1_f_cc, x_2_f_cc, x_3_f_cc))
y_all_f_cc = np.concatenate((y_1_f_cc, y_2_f_cc, y_3_f_cc))

# Realizar el ajuste lineal
slope_i_cc, intercept_i_cc, r_value_i_cc, p_value_i_cc, std_err_i_cc = linregress(x_all_i_cc, y_all_i_cc)
slope_tf_cc, intercept_tf_cc, r_value_tf_cc, p_value_tf_cc, std_err_tf_cc = linregress(x_all_tf_cc, y_all_tf_cc)
slope_f_cc, intercept_f_cc, r_value_f_cc, p_value_f_cc, std_err_f_cc = linregress(x_all_f_cc, y_all_f_cc)
a_i_cc = ufloat(slope_i_cc,std_err_i_cc)
a_tf_cc = ufloat(slope_tf_cc,std_err_tf_cc)
a_f_cc = ufloat(slope_f_cc,std_err_f_cc)

# Mostrar los resultados del ajuste lineal
print(f'Pendiente_i: {a_i_cc}') 
print(f'Pendiente_tf: {a_tf_cc}')
print(f'Pendiente_f: {a_f_cc}')
# print(f"Intersección_i: {intercept_i} - Intersección_f: {intercept_f}")

x_aux_i=np.linspace(-50,lim_ajuste_i,1000)
y_aux_i=lineal(x_aux_i,slope_i_cc,intercept_i_cc)
x_aux_tf=np.linspace(lim_ajuste_tf,lim_ajuste_f,1000)
y_aux_tf=lineal(x_aux_tf,slope_tf_cc,intercept_tf_cc)
x_aux_f=np.linspace(lim_ajuste_f,25,1000)
y_aux_f=lineal(x_aux_f,slope_f_cc,intercept_f_cc)

fig,ax= plt.subplots(figsize=(9,5),constrained_layout=True)
ax.plot(x_all_i_cc, y_all_i_cc, '.',label=f'i: [{min(x_all_i_cc)} ; {lim_ajuste_i}]°C')
ax.plot(x_aux_i,y_aux_i,label=f'{a_i_cc:.1uf} ns/°C')
ax.plot(x_all_tf_cc, y_all_tf_cc,'.', label=f'tf: [{lim_ajuste_i} ; {lim_ajuste_f}]°C')
ax.plot(x_aux_tf,y_aux_tf,label=f'{a_tf_cc:.1uf} ns/°C')
ax.plot(x_all_f_cc, y_all_f_cc, '.',label=f'f: [{lim_ajuste_f} ; {max(x_all_f_cc)}]°C')
ax.plot(x_aux_f,y_aux_f,label=f'{a_f_cc:.1uf} ns/°C')

plt.legend(ncol=3)
plt.grid()
plt.ylabel(r'$\tau$ (s)')
plt.xlabel('Indx')
plt.title(identif_4_cc)
#plt.savefig('tau_vs_T_'+identif_4+'.png',dpi=300)
plt.show()

#%%%

#%% 265 kHz 57 kA/m   CC vs SC
identif_4_sc='265_20_CsC'
dir_4_sc = os.path.join(os.getcwd(),identif_4_sc)
archivos_resultados = [f for f in os.listdir(dir_4_sc) if  fnmatch.fnmatch(f, '*resultados*')]
archivos_resultados.sort()
filepaths_sc = [os.path.join(dir_4_sc,f) for f in archivos_resultados]

_,files_4_1_sc,time_4_1_sc,temp_4_1_sc,_,_,_,_,_,_,_,_,SAR_4_1_sc,tau_4_1_sc,_ = lector_resultados(filepaths_sc[0])
_,files_4_2_sc,time_4_2_sc,temp_4_2_sc,_,_,_,_,_,_,_,_,SAR_4_2_sc,tau_4_2_sc,_ = lector_resultados(filepaths_sc[1])
_,files_4_3_sc,time_4_3_sc,temp_4_3_sc,_,_,_,_,_,_,_,_,SAR_4_3_sc,tau_4_3_sc,_ = lector_resultados(filepaths_sc[2])

tau_4_1_sc=tau_4_1_sc*1e9
tau_4_2_sc=tau_4_2_sc*1e9
tau_4_3_sc=tau_4_3_sc*1e9

identif_4_cc='265_20_CcC'
dir_4_cc = os.path.join(os.getcwd(),identif_4_cc)
archivos_resultados = [f for f in os.listdir(dir_4_cc) if  fnmatch.fnmatch(f, '*resultados*')]
archivos_resultados.sort()
filepaths_cc = [os.path.join(dir_4_cc,f) for f in archivos_resultados]

_,files_4_1_cc,time_4_1_cc,temp_4_1_cc,_,_,_,_,_,_,_,_,SAR_4_1_cc,tau_4_1_cc,_ = lector_resultados(filepaths_cc[0])
_,files_4_2_cc,time_4_2_cc,temp_4_2_cc,_,_,_,_,_,_,_,_,SAR_4_2_cc,tau_4_2_cc,_ = lector_resultados(filepaths_cc[1])
_,files_4_3_cc,time_4_3_cc,temp_4_3_cc,_,_,_,_,_,_,_,_,SAR_4_3_cc,tau_4_3_cc,_ = lector_resultados(filepaths_cc[2])

tau_4_1_cc=tau_4_1_cc*1e9
tau_4_2_cc=tau_4_2_cc*1e9
tau_4_3_cc=tau_4_3_cc*1e9

temp_all_sc = np.concatenate((temp_4_1_sc, temp_4_2_sc, temp_4_3_sc))
tau_all_sc = np.concatenate((tau_4_1_sc, tau_4_2_sc, tau_4_3_sc))

temp_all_cc = np.concatenate((temp_4_1_cc, temp_4_2_cc, temp_4_3_cc))
tau_all_cc = np.concatenate((tau_4_1_cc, tau_4_2_cc, tau_4_3_cc))
#%
fig,ax= plt.subplots(figsize=(9,5),constrained_layout=True)

# ax.plot(temp_all_sc,tau_all_sc,'o-',label='1 sc')
# ax.plot(temp_all_cc,tau_all_cc,'o-',label='1 cc')
ax.plot(temp_4_1_sc,tau_4_1_sc,'o-',c='tab:blue',label='1 sc')
ax.plot(temp_4_2_sc,tau_4_2_sc,'o-',c='tab:blue',label='2 sc')
ax.plot(temp_4_3_sc,tau_4_3_sc,'o-',c='tab:blue',label='3 sc')

ax.plot(temp_4_1_cc,tau_4_1_cc,'s-',c='tab:orange',label='1 H$_{⟂}$')
ax.plot(temp_4_2_cc,tau_4_2_cc,'s-',c='tab:orange',label='2 H$_{⟂}$')
ax.plot(temp_4_3_cc,tau_4_3_cc,'s-',c='tab:orange',label='3 H$_{⟂}$')

plt.legend(ncol=2)
plt.grid()
plt.ylabel(r'$\tau$ (s)')
plt.xlabel('Temperatura (ºC)')
plt.title(identif_4_sc +' - '+identif_4_cc)
# plt.xlim(-10,10)
plt.savefig('tau_vs_T_comparativa_'+identif_4_sc+' '+identif_4_cc + '.png',dpi=300)
plt.show()

#%%
lim_ajuste_i= -6
lim_ajuste_tf = -4.0
lim_ajuste_f= 0
#% SC
x_1_i_sc=temp_4_1_sc[np.nonzero(temp_4_1_sc<=lim_ajuste_i)]
x_1_tf_sc=temp_4_1_sc[np.nonzero((temp_4_1_sc>lim_ajuste_tf) & (temp_4_1_sc<lim_ajuste_f))]
x_1_f_sc=temp_4_1_sc[np.nonzero(temp_4_1_sc>=lim_ajuste_f)]

y_1_i_sc=tau_4_1_sc[np.nonzero(temp_4_1_sc<=lim_ajuste_i)]
y_1_tf_sc=tau_4_1_sc[np.nonzero((temp_4_1_sc>lim_ajuste_tf) & (temp_4_1_sc<lim_ajuste_f))]
y_1_f_sc=tau_4_1_sc[np.nonzero(temp_4_1_sc>=lim_ajuste_f)]

x_2_i_sc=temp_4_2_sc[np.nonzero(temp_4_2_sc<=lim_ajuste_i)]
x_2_tf_sc=temp_4_2_sc[np.nonzero((temp_4_2_sc>lim_ajuste_tf) & (temp_4_2_sc<lim_ajuste_f))]
x_2_f_sc=temp_4_2_sc[np.nonzero(temp_4_2_sc>=lim_ajuste_f)]

y_2_i_sc=tau_4_2_sc[np.nonzero(temp_4_2_sc<=lim_ajuste_i)]
y_2_tf_sc=tau_4_2_sc[np.nonzero((temp_4_2_sc>lim_ajuste_tf) & (temp_4_2_sc<lim_ajuste_f))]
y_2_f_sc=tau_4_2_sc[np.nonzero(temp_4_2_sc>=lim_ajuste_f)]

x_3_i_sc=temp_4_3_sc[np.nonzero(temp_4_3_sc<=lim_ajuste_i)]
x_3_tf_sc=temp_4_3_sc[np.nonzero((temp_4_3_sc>lim_ajuste_tf) & (temp_4_3_sc<lim_ajuste_f))]
x_3_f_sc=temp_4_3_sc[np.nonzero(temp_4_3_sc>=lim_ajuste_f)]

y_3_i_sc=tau_4_3_sc[np.nonzero(temp_4_3_sc<=lim_ajuste_i)]
y_3_tf_sc=tau_4_3_sc[np.nonzero((temp_4_3_sc>lim_ajuste_tf) & (temp_4_3_sc<lim_ajuste_f))]
y_3_f_sc=tau_4_3_sc[np.nonzero(temp_4_3_sc>=lim_ajuste_f)]

x_all_i_sc = np.concatenate((x_1_i_sc, x_2_i_sc, x_3_i_sc))
y_all_i_sc = np.concatenate((y_1_i_sc, y_2_i_sc, y_3_i_sc))
x_all_tf_sc = np.concatenate((x_1_tf_sc, x_2_tf_sc, x_3_tf_sc))
y_all_tf_sc = np.concatenate((y_1_tf_sc, y_2_tf_sc, y_3_tf_sc))
x_all_f_sc = np.concatenate((x_1_f_sc, x_2_f_sc, x_3_f_sc))
y_all_f_sc = np.concatenate((y_1_f_sc, y_2_f_sc, y_3_f_sc))

# Realizar el ajuste lineal
slope_i_sc, intercept_i_sc, r_value_i_sc, p_value_i_sc, std_err_i_sc = linregress(x_all_i_sc, y_all_i_sc)
slope_tf_sc, intercept_tf_sc, r_value_tf_sc, p_value_tf_sc, std_err_tf_sc = linregress(x_all_tf_sc, y_all_tf_sc)
slope_f_sc, intercept_f_sc, r_value_f_sc, p_value_f_sc, std_err_f_sc = linregress(x_all_f_sc, y_all_f_sc)
a_i_sc = ufloat(slope_i_sc,std_err_i_sc)
a_tf_sc = ufloat(slope_tf_sc,std_err_tf_sc)
a_f_sc = ufloat(slope_f_sc,std_err_f_sc)

# Mostrar los resultados del ajuste lineal
print('FF congelado SIN campo')
print(f'Pendiente_i: {a_i_sc}') 
print(f'Pendiente_tf: {a_tf_sc}')
print(f'Pendiente_f: {a_f_sc}')

x_sc_i=np.linspace(-50,lim_ajuste_i,1000)
y_sc_i=lineal(x_sc_i,slope_i_sc,intercept_i_sc)
x_sc_tf=np.linspace(lim_ajuste_tf,lim_ajuste_f,1000)
y_sc_tf=lineal(x_sc_tf,slope_tf_sc,intercept_tf_sc)
x_sc_f=np.linspace(lim_ajuste_f,25,1000)
y_sc_f=lineal(x_sc_f,slope_f_sc,intercept_f_sc)

#% CC
x_1_i_cc=temp_4_1_cc[np.nonzero(temp_4_1_cc<=lim_ajuste_i)]
x_1_tf_cc=temp_4_1_cc[np.nonzero((temp_4_1_cc>lim_ajuste_tf) & (temp_4_1_cc<lim_ajuste_f))]
x_1_f_cc=temp_4_1_cc[np.nonzero(temp_4_1_cc>=lim_ajuste_f)]

y_1_i_cc=tau_4_1_cc[np.nonzero(temp_4_1_cc<=lim_ajuste_i)]
y_1_tf_cc=tau_4_1_cc[np.nonzero((temp_4_1_cc>lim_ajuste_tf) & (temp_4_1_cc<lim_ajuste_f))]
y_1_f_cc=tau_4_1_cc[np.nonzero(temp_4_1_cc>=lim_ajuste_f)]

x_2_i_cc=temp_4_2_cc[np.nonzero(temp_4_2_cc<=lim_ajuste_i)]
x_2_tf_cc=temp_4_2_cc[np.nonzero((temp_4_2_cc>lim_ajuste_tf) & (temp_4_2_cc<lim_ajuste_f))]
x_2_f_cc=temp_4_2_cc[np.nonzero(temp_4_2_cc>=lim_ajuste_f)]

y_2_i_cc=tau_4_2_cc[np.nonzero(temp_4_2_cc<=lim_ajuste_i)]
y_2_tf_cc=tau_4_2_cc[np.nonzero((temp_4_2_cc>lim_ajuste_tf) & (temp_4_2_cc<lim_ajuste_f))]
y_2_f_cc=tau_4_2_cc[np.nonzero(temp_4_2_cc>=lim_ajuste_f)]

x_3_i_cc=temp_4_3_cc[np.nonzero(temp_4_3_cc<=lim_ajuste_i)]
x_3_tf_cc=temp_4_3_cc[np.nonzero((temp_4_3_cc>lim_ajuste_tf) & (temp_4_3_cc<lim_ajuste_f))]
x_3_f_cc=temp_4_3_cc[np.nonzero(temp_4_3_cc>=lim_ajuste_f)]

y_3_i_cc=tau_4_3_cc[np.nonzero(temp_4_3_cc<=lim_ajuste_i)]
y_3_tf_cc=tau_4_3_cc[np.nonzero((temp_4_3_cc>lim_ajuste_tf) & (temp_4_3_cc<lim_ajuste_f))]
y_3_f_cc=tau_4_3_cc[np.nonzero(temp_4_3_cc>=lim_ajuste_f)]

x_all_i_cc = np.concatenate((x_1_i_cc, x_2_i_cc, x_3_i_cc))
y_all_i_cc = np.concatenate((y_1_i_cc, y_2_i_cc, y_3_i_cc))
x_all_tf_cc = np.concatenate((x_1_tf_cc, x_2_tf_cc, x_3_tf_cc))
y_all_tf_cc = np.concatenate((y_1_tf_cc, y_2_tf_cc, y_3_tf_cc))
x_all_f_cc = np.concatenate((x_1_f_cc, x_2_f_cc, x_3_f_cc))
y_all_f_cc = np.concatenate((y_1_f_cc, y_2_f_cc, y_3_f_cc))

# Realizar el ajuste lineal
slope_i_cc, intercept_i_cc, r_value_i_cc, p_value_i_cc, std_err_i_cc = linregress(x_all_i_cc, y_all_i_cc)
slope_tf_cc, intercept_tf_cc, r_value_tf_cc, p_value_tf_cc, std_err_tf_cc = linregress(x_all_tf_cc, y_all_tf_cc)
slope_f_cc, intercept_f_cc, r_value_f_cc, p_value_f_cc, std_err_f_cc = linregress(x_all_f_cc, y_all_f_cc)
a_i_cc = ufloat(slope_i_cc,std_err_i_cc)
a_tf_cc = ufloat(slope_tf_cc,std_err_tf_cc)
a_f_cc = ufloat(slope_f_cc,std_err_f_cc)

# Mostrar los resultados del ajuste lineal
print('FF congelado CON campo')
print(f'Pendiente_i: {a_i_cc}') 
print(f'Pendiente_tf: {a_tf_cc}')
print(f'Pendiente_f: {a_f_cc}')

x_cc_i=np.linspace(-50,lim_ajuste_i,1000)
y_cc_i=lineal(x_cc_i,slope_i_cc,intercept_i_cc)
x_cc_tf=np.linspace(lim_ajuste_tf,lim_ajuste_f,1000)
y_cc_tf=lineal(x_cc_tf,slope_tf_cc,intercept_tf_cc)
x_cc_f=np.linspace(lim_ajuste_f,25,1000)
y_cc_f=lineal(x_cc_f,slope_f_cc,intercept_f_cc)

#%%
fig,ax= plt.subplots(figsize=(9,5),constrained_layout=True)
ax.plot(x_all_i_sc, y_all_i_sc, 'o',label=f'i: [{min(x_all_i_sc)} ; {lim_ajuste_i}]°C')
ax.plot(x_sc_i,y_sc_i,label=f'{a_i_sc:.1uf} ns/°C')
ax.plot(x_all_tf_sc, y_all_tf_sc,'o', label=f'tf: [{lim_ajuste_i} ; {lim_ajuste_f}]°C')
ax.plot(x_sc_tf,y_sc_tf,label=f'{a_tf_sc:.1uf} ns/°C')
ax.plot(x_all_f_sc, y_all_f_sc, 'o',label=f'f: [{lim_ajuste_f} ; {max(x_all_f_sc)}]°C')
ax.plot(x_sc_f,y_sc_f,label=f'{a_f_sc:.1uf} ns/°C')

ax.plot(x_all_i_cc, y_all_i_cc, 'o',label=f'i: [{min(x_all_i_cc)} ; {lim_ajuste_i}]°C')
ax.plot(x_cc_i,y_cc_i,label=f'{a_i_cc:.1uf} ns/°C')
ax.plot(x_all_tf_cc, y_all_tf_cc,'o', label=f'tf: [{lim_ajuste_i} ; {lim_ajuste_f}]°C')
ax.plot(x_cc_tf,y_cc_tf,label=f'{a_tf_cc:.1uf} ns/°C')
ax.plot(x_all_f_cc, y_all_f_cc, 'o',label=f'f: [{lim_ajuste_f} ; {max(x_all_f_cc)}]°C')
ax.plot(x_cc_f,y_cc_f,label=f'{a_f_cc:.1uf} ns/°C')

plt.legend(ncol=2)
plt.grid()
plt.ylabel(r'$\tau$ (s)')
plt.xlabel('Temperatura (°C)')
plt.title(identif_4_sc +' - ' + identif_4_sc)
plt.savefig('tau_vs_T_ajustes_265_20.png',dpi=300)
plt.show()
#%
x_1_i_cc=temp_4_1_cc[np.nonzero(temp_4_1_cc<=lim_ajuste_i)]
x_1_tf_cc=temp_4_1_cc[np.nonzero((temp_4_1_cc>lim_ajuste_tf) & (temp_4_1_cc<lim_ajuste_f))]
x_1_f_cc=temp_4_1_cc[np.nonzero(temp_4_1_cc>=lim_ajuste_f)]

y_1_i_cc=tau_4_1_cc[np.nonzero(temp_4_1_cc<=lim_ajuste_i)]
y_1_tf_cc=tau_4_1_cc[np.nonzero((temp_4_1_cc>lim_ajuste_tf) & (temp_4_1_cc<lim_ajuste_f))]
y_1_f_cc=tau_4_1_cc[np.nonzero(temp_4_1_cc>=lim_ajuste_f)]

x_2_i_cc=temp_4_2_cc[np.nonzero(temp_4_2_cc<=lim_ajuste_i)]
x_2_tf_cc=temp_4_2_cc[np.nonzero((temp_4_2_cc>lim_ajuste_tf) & (temp_4_2_cc<lim_ajuste_f))]
x_2_f_cc=temp_4_2_cc[np.nonzero(temp_4_2_cc>=lim_ajuste_f)]

y_2_i_cc=tau_4_2_cc[np.nonzero(temp_4_2_cc<=lim_ajuste_i)]
y_2_tf_cc=tau_4_2_cc[np.nonzero((temp_4_2_cc>lim_ajuste_tf) & (temp_4_2_cc<lim_ajuste_f))]
y_2_f_cc=tau_4_2_cc[np.nonzero(temp_4_2_cc>=lim_ajuste_f)]

x_3_i_cc=temp_4_3_cc[np.nonzero(temp_4_3_cc<=lim_ajuste_i)]
x_3_tf_cc=temp_4_3_cc[np.nonzero((temp_4_3_cc>lim_ajuste_tf) & (temp_4_3_cc<lim_ajuste_f))]
x_3_f_cc=temp_4_3_cc[np.nonzero(temp_4_3_cc>=lim_ajuste_f)]

y_3_i_cc=tau_4_3_cc[np.nonzero(temp_4_3_cc<=lim_ajuste_i)]
y_3_tf_cc=tau_4_3_cc[np.nonzero((temp_4_3_cc>lim_ajuste_tf) & (temp_4_3_cc<lim_ajuste_f))]
y_3_f_cc=tau_4_3_cc[np.nonzero(temp_4_3_cc>=lim_ajuste_f)]

x_all_i_cc = np.concatenate((x_1_i_cc, x_2_i_cc, x_3_i_cc))
y_all_i_cc = np.concatenate((y_1_i_cc, y_2_i_cc, y_3_i_cc))
x_all_tf_cc = np.concatenate((x_1_tf_cc, x_2_tf_cc, x_3_tf_cc))
y_all_tf_cc = np.concatenate((y_1_tf_cc, y_2_tf_cc, y_3_tf_cc))
x_all_f_cc = np.concatenate((x_1_f_cc, x_2_f_cc, x_3_f_cc))
y_all_f_cc = np.concatenate((y_1_f_cc, y_2_f_cc, y_3_f_cc))

# Realizar el ajuste lineal
slope_i_cc, intercept_i_cc, r_value_i_cc, p_value_i_cc, std_err_i_cc = linregress(x_all_i_cc, y_all_i_cc)
slope_tf_cc, intercept_tf_cc, r_value_tf_cc, p_value_tf_cc, std_err_tf_cc = linregress(x_all_tf_cc, y_all_tf_cc)
slope_f_cc, intercept_f_cc, r_value_f_cc, p_value_f_cc, std_err_f_cc = linregress(x_all_f_cc, y_all_f_cc)
a_i_cc = ufloat(slope_i_cc,std_err_i_cc)
a_tf_cc = ufloat(slope_tf_cc,std_err_tf_cc)
a_f_cc = ufloat(slope_f_cc,std_err_f_cc)

# Mostrar los resultados del ajuste lineal
print(f'Pendiente_i: {a_i_cc}') 
print(f'Pendiente_tf: {a_tf_cc}')
print(f'Pendiente_f: {a_f_cc}')
# print(f"Intersección_i: {intercept_i} - Intersección_f: {intercept_f}")

x_aux_i=np.linspace(-50,lim_ajuste_i,1000)
y_aux_i=lineal(x_aux_i,slope_i_cc,intercept_i_cc)
x_aux_tf=np.linspace(lim_ajuste_tf,lim_ajuste_f,1000)
y_aux_tf=lineal(x_aux_tf,slope_tf_cc,intercept_tf_cc)
x_aux_f=np.linspace(lim_ajuste_f,25,1000)
y_aux_f=lineal(x_aux_f,slope_f_cc,intercept_f_cc)

# fig,ax= plt.subplots(figsize=(9,5),constrained_layout=True)
# ax.plot(x_all_i_cc, y_all_i_cc, '.',label=f'i: [{min(x_all_i_cc)} ; {lim_ajuste_i}]°C')
# ax.plot(x_aux_i,y_aux_i,label=f'{a_i_cc:.1uf} ns/°C')
# ax.plot(x_all_tf_cc, y_all_tf_cc,'.', label=f'tf: [{lim_ajuste_i} ; {lim_ajuste_f}]°C')
# ax.plot(x_aux_tf,y_aux_tf,label=f'{a_tf_cc:.1uf} ns/°C')
# ax.plot(x_all_f_cc, y_all_f_cc, '.',label=f'f: [{lim_ajuste_f} ; {max(x_all_f_cc)}]°C')
# ax.plot(x_aux_f,y_aux_f,label=f'{a_f_cc:.1uf} ns/°C')

# plt.legend(ncol=3)
# plt.grid()
# plt.ylabel(r'$\tau$ (s)')
# plt.xlabel('Indx')
# plt.title(identif_4_cc)
# #plt.savefig('tau_vs_T_'+identif_4+'.png',dpi=300)
# plt.show()

# %%
#%% 135 kHz 20 kA/m   CC vs SC
identif_4_sc='135_20_CsC'
dir_4_sc = os.path.join(os.getcwd(),identif_4_sc)
archivos_resultados = [f for f in os.listdir(dir_4_sc) if  fnmatch.fnmatch(f, '*resultados*')]
archivos_resultados.sort()
filepaths_sc = [os.path.join(dir_4_sc,f) for f in archivos_resultados]

_,files_4_1_sc,time_4_1_sc,temp_4_1_sc,_,_,_,_,_,_,_,_,SAR_4_1_sc,tau_4_1_sc,_ = lector_resultados(filepaths_sc[0])
_,files_4_2_sc,time_4_2_sc,temp_4_2_sc,_,_,_,_,_,_,_,_,SAR_4_2_sc,tau_4_2_sc,_ = lector_resultados(filepaths_sc[1])
_,files_4_3_sc,time_4_3_sc,temp_4_3_sc,_,_,_,_,_,_,_,_,SAR_4_3_sc,tau_4_3_sc,_ = lector_resultados(filepaths_sc[2])

tau_4_1_sc=tau_4_1_sc*1e9
tau_4_2_sc=tau_4_2_sc*1e9
tau_4_3_sc=tau_4_3_sc*1e9

identif_4_cc='135_20_CcC'
dir_4_cc = os.path.join(os.getcwd(),identif_4_cc)
archivos_resultados = [f for f in os.listdir(dir_4_cc) if  fnmatch.fnmatch(f, '*resultados*')]
archivos_resultados.sort()
filepaths_cc = [os.path.join(dir_4_cc,f) for f in archivos_resultados]

_,files_4_1_cc,time_4_1_cc,temp_4_1_cc,_,_,_,_,_,_,_,_,SAR_4_1_cc,tau_4_1_cc,_ = lector_resultados(filepaths_cc[0])
_,files_4_2_cc,time_4_2_cc,temp_4_2_cc,_,_,_,_,_,_,_,_,SAR_4_2_cc,tau_4_2_cc,_ = lector_resultados(filepaths_cc[1])
_,files_4_3_cc,time_4_3_cc,temp_4_3_cc,_,_,_,_,_,_,_,_,SAR_4_3_cc,tau_4_3_cc,_ = lector_resultados(filepaths_cc[2])

tau_4_1_cc=tau_4_1_cc*1e9
tau_4_2_cc=tau_4_2_cc*1e9
tau_4_3_cc=tau_4_3_cc*1e9

temp_all_sc = np.concatenate((temp_4_1_sc, temp_4_2_sc, temp_4_3_sc))
tau_all_sc = np.concatenate((tau_4_1_sc, tau_4_2_sc, tau_4_3_sc))

temp_all_cc = np.concatenate((temp_4_1_cc, temp_4_2_cc, temp_4_3_cc))
tau_all_cc = np.concatenate((tau_4_1_cc, tau_4_2_cc, tau_4_3_cc))
#%
fig,ax= plt.subplots(figsize=(9,5),constrained_layout=True)

# ax.plot(temp_all_sc,tau_all_sc,'o-',label='1 sc')
# ax.plot(temp_all_cc,tau_all_cc,'o-',label='1 cc')
ax.plot(temp_4_1_sc,tau_4_1_sc,'o-',c='tab:blue',label='1 sc')
ax.plot(temp_4_2_sc,tau_4_2_sc,'o-',c='tab:blue',label='2 sc')
ax.plot(temp_4_3_sc,tau_4_3_sc,'o-',c='tab:blue',label='3 sc')

ax.plot(temp_4_1_cc,tau_4_1_cc,'s-',c='tab:orange',label='1 H$_{⟂}$')
ax.plot(temp_4_2_cc,tau_4_2_cc,'s-',c='tab:orange',label='2 H$_{⟂}$')
ax.plot(temp_4_3_cc,tau_4_3_cc,'s-',c='tab:orange',label='3 H$_{⟂}$')

plt.legend(ncol=2)
plt.grid()
plt.ylabel(r'$\tau$ (ns)')
plt.xlabel('Temperatura (ºC)')
plt.title(identif_4_sc +' - '+identif_4_cc)
plt.xlim(-10,10)
plt.savefig('tau_vs_T_comparativa_'+identif_4_sc+' '+identif_4_cc + '_zoom.png',dpi=300)
plt.show()

#%%
lim_ajuste_i= -6
lim_ajuste_tf = -4.0
lim_ajuste_f= 0
#% SC
x_1_i_sc=temp_4_1_sc[np.nonzero(temp_4_1_sc<=lim_ajuste_i)]
x_1_tf_sc=temp_4_1_sc[np.nonzero((temp_4_1_sc>lim_ajuste_tf) & (temp_4_1_sc<lim_ajuste_f))]
x_1_f_sc=temp_4_1_sc[np.nonzero(temp_4_1_sc>=lim_ajuste_f)]

y_1_i_sc=tau_4_1_sc[np.nonzero(temp_4_1_sc<=lim_ajuste_i)]
y_1_tf_sc=tau_4_1_sc[np.nonzero((temp_4_1_sc>lim_ajuste_tf) & (temp_4_1_sc<lim_ajuste_f))]
y_1_f_sc=tau_4_1_sc[np.nonzero(temp_4_1_sc>=lim_ajuste_f)]

x_2_i_sc=temp_4_2_sc[np.nonzero(temp_4_2_sc<=lim_ajuste_i)]
x_2_tf_sc=temp_4_2_sc[np.nonzero((temp_4_2_sc>lim_ajuste_tf) & (temp_4_2_sc<lim_ajuste_f))]
x_2_f_sc=temp_4_2_sc[np.nonzero(temp_4_2_sc>=lim_ajuste_f)]

y_2_i_sc=tau_4_2_sc[np.nonzero(temp_4_2_sc<=lim_ajuste_i)]
y_2_tf_sc=tau_4_2_sc[np.nonzero((temp_4_2_sc>lim_ajuste_tf) & (temp_4_2_sc<lim_ajuste_f))]
y_2_f_sc=tau_4_2_sc[np.nonzero(temp_4_2_sc>=lim_ajuste_f)]

x_3_i_sc=temp_4_3_sc[np.nonzero(temp_4_3_sc<=lim_ajuste_i)]
x_3_tf_sc=temp_4_3_sc[np.nonzero((temp_4_3_sc>lim_ajuste_tf) & (temp_4_3_sc<lim_ajuste_f))]
x_3_f_sc=temp_4_3_sc[np.nonzero(temp_4_3_sc>=lim_ajuste_f)]

y_3_i_sc=tau_4_3_sc[np.nonzero(temp_4_3_sc<=lim_ajuste_i)]
y_3_tf_sc=tau_4_3_sc[np.nonzero((temp_4_3_sc>lim_ajuste_tf) & (temp_4_3_sc<lim_ajuste_f))]
y_3_f_sc=tau_4_3_sc[np.nonzero(temp_4_3_sc>=lim_ajuste_f)]

x_all_i_sc = np.concatenate((x_1_i_sc, x_2_i_sc, x_3_i_sc))
y_all_i_sc = np.concatenate((y_1_i_sc, y_2_i_sc, y_3_i_sc))
x_all_tf_sc = np.concatenate((x_1_tf_sc, x_2_tf_sc, x_3_tf_sc))
y_all_tf_sc = np.concatenate((y_1_tf_sc, y_2_tf_sc, y_3_tf_sc))
x_all_f_sc = np.concatenate((x_1_f_sc, x_2_f_sc, x_3_f_sc))
y_all_f_sc = np.concatenate((y_1_f_sc, y_2_f_sc, y_3_f_sc))

# Realizar el ajuste lineal
slope_i_sc, intercept_i_sc, r_value_i_sc, p_value_i_sc, std_err_i_sc = linregress(x_all_i_sc, y_all_i_sc)
slope_tf_sc, intercept_tf_sc, r_value_tf_sc, p_value_tf_sc, std_err_tf_sc = linregress(x_all_tf_sc, y_all_tf_sc)
slope_f_sc, intercept_f_sc, r_value_f_sc, p_value_f_sc, std_err_f_sc = linregress(x_all_f_sc, y_all_f_sc)
a_i_sc = ufloat(slope_i_sc,std_err_i_sc)
a_tf_sc = ufloat(slope_tf_sc,std_err_tf_sc)
a_f_sc = ufloat(slope_f_sc,std_err_f_sc)

# Mostrar los resultados del ajuste lineal
print('FF congelado SIN campo')
print(f'Pendiente_i: {a_i_sc}') 
print(f'Pendiente_tf: {a_tf_sc}')
print(f'Pendiente_f: {a_f_sc}')

x_sc_i=np.linspace(-50,lim_ajuste_i,1000)
y_sc_i=lineal(x_sc_i,slope_i_sc,intercept_i_sc)
x_sc_tf=np.linspace(lim_ajuste_tf,lim_ajuste_f,1000)
y_sc_tf=lineal(x_sc_tf,slope_tf_sc,intercept_tf_sc)
x_sc_f=np.linspace(lim_ajuste_f,25,1000)
y_sc_f=lineal(x_sc_f,slope_f_sc,intercept_f_sc)

#% CC
x_1_i_cc=temp_4_1_cc[np.nonzero(temp_4_1_cc<=lim_ajuste_i)]
x_1_tf_cc=temp_4_1_cc[np.nonzero((temp_4_1_cc>lim_ajuste_tf) & (temp_4_1_cc<lim_ajuste_f))]
x_1_f_cc=temp_4_1_cc[np.nonzero(temp_4_1_cc>=lim_ajuste_f)]

y_1_i_cc=tau_4_1_cc[np.nonzero(temp_4_1_cc<=lim_ajuste_i)]
y_1_tf_cc=tau_4_1_cc[np.nonzero((temp_4_1_cc>lim_ajuste_tf) & (temp_4_1_cc<lim_ajuste_f))]
y_1_f_cc=tau_4_1_cc[np.nonzero(temp_4_1_cc>=lim_ajuste_f)]

x_2_i_cc=temp_4_2_cc[np.nonzero(temp_4_2_cc<=lim_ajuste_i)]
x_2_tf_cc=temp_4_2_cc[np.nonzero((temp_4_2_cc>lim_ajuste_tf) & (temp_4_2_cc<lim_ajuste_f))]
x_2_f_cc=temp_4_2_cc[np.nonzero(temp_4_2_cc>=lim_ajuste_f)]

y_2_i_cc=tau_4_2_cc[np.nonzero(temp_4_2_cc<=lim_ajuste_i)]
y_2_tf_cc=tau_4_2_cc[np.nonzero((temp_4_2_cc>lim_ajuste_tf) & (temp_4_2_cc<lim_ajuste_f))]
y_2_f_cc=tau_4_2_cc[np.nonzero(temp_4_2_cc>=lim_ajuste_f)]

x_3_i_cc=temp_4_3_cc[np.nonzero(temp_4_3_cc<=lim_ajuste_i)]
x_3_tf_cc=temp_4_3_cc[np.nonzero((temp_4_3_cc>lim_ajuste_tf) & (temp_4_3_cc<lim_ajuste_f))]
x_3_f_cc=temp_4_3_cc[np.nonzero(temp_4_3_cc>=lim_ajuste_f)]

y_3_i_cc=tau_4_3_cc[np.nonzero(temp_4_3_cc<=lim_ajuste_i)]
y_3_tf_cc=tau_4_3_cc[np.nonzero((temp_4_3_cc>lim_ajuste_tf) & (temp_4_3_cc<lim_ajuste_f))]
y_3_f_cc=tau_4_3_cc[np.nonzero(temp_4_3_cc>=lim_ajuste_f)]

x_all_i_cc = np.concatenate((x_1_i_cc, x_2_i_cc, x_3_i_cc))
y_all_i_cc = np.concatenate((y_1_i_cc, y_2_i_cc, y_3_i_cc))
x_all_tf_cc = np.concatenate((x_1_tf_cc, x_2_tf_cc, x_3_tf_cc))
y_all_tf_cc = np.concatenate((y_1_tf_cc, y_2_tf_cc, y_3_tf_cc))
x_all_f_cc = np.concatenate((x_1_f_cc, x_2_f_cc, x_3_f_cc))
y_all_f_cc = np.concatenate((y_1_f_cc, y_2_f_cc, y_3_f_cc))

# Realizar el ajuste lineal
slope_i_cc, intercept_i_cc, r_value_i_cc, p_value_i_cc, std_err_i_cc = linregress(x_all_i_cc, y_all_i_cc)
slope_tf_cc, intercept_tf_cc, r_value_tf_cc, p_value_tf_cc, std_err_tf_cc = linregress(x_all_tf_cc, y_all_tf_cc)
slope_f_cc, intercept_f_cc, r_value_f_cc, p_value_f_cc, std_err_f_cc = linregress(x_all_f_cc, y_all_f_cc)
a_i_cc = ufloat(slope_i_cc,std_err_i_cc)
a_tf_cc = ufloat(slope_tf_cc,std_err_tf_cc)
a_f_cc = ufloat(slope_f_cc,std_err_f_cc)

# Mostrar los resultados del ajuste lineal
print('FF congelado CON campo')
print(f'Pendiente_i: {a_i_cc}') 
print(f'Pendiente_tf: {a_tf_cc}')
print(f'Pendiente_f: {a_f_cc}')

x_cc_i=np.linspace(-50,lim_ajuste_i,1000)
y_cc_i=lineal(x_cc_i,slope_i_cc,intercept_i_cc)
x_cc_tf=np.linspace(lim_ajuste_tf,lim_ajuste_f,1000)
y_cc_tf=lineal(x_cc_tf,slope_tf_cc,intercept_tf_cc)
x_cc_f=np.linspace(lim_ajuste_f,25,1000)
y_cc_f=lineal(x_cc_f,slope_f_cc,intercept_f_cc)

#%%
fig,ax= plt.subplots(figsize=(9,5),constrained_layout=True)
ax.plot(x_all_i_sc, y_all_i_sc, 'o',label=f'i: [{min(x_all_i_sc)} ; {lim_ajuste_i}]°C')
ax.plot(x_sc_i,y_sc_i,label=f'{a_i_sc:.1uf} ns/°C')
ax.plot(x_all_tf_sc, y_all_tf_sc,'o', label=f'tf: [{lim_ajuste_i} ; {lim_ajuste_f}]°C')
ax.plot(x_sc_tf,y_sc_tf,label=f'{a_tf_sc:.1uf} ns/°C')
ax.plot(x_all_f_sc, y_all_f_sc, 'o',label=f'f: [{lim_ajuste_f} ; {max(x_all_f_sc)}]°C')
ax.plot(x_sc_f,y_sc_f,label=f'{a_f_sc:.1uf} ns/°C')

ax.plot(x_all_i_cc, y_all_i_cc, 'o',label=f'i: [{min(x_all_i_cc)} ; {lim_ajuste_i}]°C')
ax.plot(x_cc_i,y_cc_i,label=f'{a_i_cc:.1uf} ns/°C')
ax.plot(x_all_tf_cc, y_all_tf_cc,'o', label=f'tf: [{lim_ajuste_i} ; {lim_ajuste_f}]°C')
ax.plot(x_cc_tf,y_cc_tf,label=f'{a_tf_cc:.1uf} ns/°C')
ax.plot(x_all_f_cc, y_all_f_cc, 'o',label=f'f: [{lim_ajuste_f} ; {max(x_all_f_cc)}]°C')
ax.plot(x_cc_f,y_cc_f,label=f'{a_f_cc:.1uf} ns/°C')

plt.legend(ncol=2)
plt.grid()
plt.ylabel(r'$\tau$ (s)')
plt.xlabel('Temperatura (°C)')
plt.title(identif_4_sc +' - ' + identif_4_sc)
plt.savefig('tau_vs_T_ajustes_265_20.png',dpi=300)
plt.show()
#%
x_1_i_cc=temp_4_1_cc[np.nonzero(temp_4_1_cc<=lim_ajuste_i)]
x_1_tf_cc=temp_4_1_cc[np.nonzero((temp_4_1_cc>lim_ajuste_tf) & (temp_4_1_cc<lim_ajuste_f))]
x_1_f_cc=temp_4_1_cc[np.nonzero(temp_4_1_cc>=lim_ajuste_f)]

y_1_i_cc=tau_4_1_cc[np.nonzero(temp_4_1_cc<=lim_ajuste_i)]
y_1_tf_cc=tau_4_1_cc[np.nonzero((temp_4_1_cc>lim_ajuste_tf) & (temp_4_1_cc<lim_ajuste_f))]
y_1_f_cc=tau_4_1_cc[np.nonzero(temp_4_1_cc>=lim_ajuste_f)]

x_2_i_cc=temp_4_2_cc[np.nonzero(temp_4_2_cc<=lim_ajuste_i)]
x_2_tf_cc=temp_4_2_cc[np.nonzero((temp_4_2_cc>lim_ajuste_tf) & (temp_4_2_cc<lim_ajuste_f))]
x_2_f_cc=temp_4_2_cc[np.nonzero(temp_4_2_cc>=lim_ajuste_f)]

y_2_i_cc=tau_4_2_cc[np.nonzero(temp_4_2_cc<=lim_ajuste_i)]
y_2_tf_cc=tau_4_2_cc[np.nonzero((temp_4_2_cc>lim_ajuste_tf) & (temp_4_2_cc<lim_ajuste_f))]
y_2_f_cc=tau_4_2_cc[np.nonzero(temp_4_2_cc>=lim_ajuste_f)]

x_3_i_cc=temp_4_3_cc[np.nonzero(temp_4_3_cc<=lim_ajuste_i)]
x_3_tf_cc=temp_4_3_cc[np.nonzero((temp_4_3_cc>lim_ajuste_tf) & (temp_4_3_cc<lim_ajuste_f))]
x_3_f_cc=temp_4_3_cc[np.nonzero(temp_4_3_cc>=lim_ajuste_f)]

y_3_i_cc=tau_4_3_cc[np.nonzero(temp_4_3_cc<=lim_ajuste_i)]
y_3_tf_cc=tau_4_3_cc[np.nonzero((temp_4_3_cc>lim_ajuste_tf) & (temp_4_3_cc<lim_ajuste_f))]
y_3_f_cc=tau_4_3_cc[np.nonzero(temp_4_3_cc>=lim_ajuste_f)]

x_all_i_cc = np.concatenate((x_1_i_cc, x_2_i_cc, x_3_i_cc))
y_all_i_cc = np.concatenate((y_1_i_cc, y_2_i_cc, y_3_i_cc))
x_all_tf_cc = np.concatenate((x_1_tf_cc, x_2_tf_cc, x_3_tf_cc))
y_all_tf_cc = np.concatenate((y_1_tf_cc, y_2_tf_cc, y_3_tf_cc))
x_all_f_cc = np.concatenate((x_1_f_cc, x_2_f_cc, x_3_f_cc))
y_all_f_cc = np.concatenate((y_1_f_cc, y_2_f_cc, y_3_f_cc))

# Realizar el ajuste lineal
slope_i_cc, intercept_i_cc, r_value_i_cc, p_value_i_cc, std_err_i_cc = linregress(x_all_i_cc, y_all_i_cc)
slope_tf_cc, intercept_tf_cc, r_value_tf_cc, p_value_tf_cc, std_err_tf_cc = linregress(x_all_tf_cc, y_all_tf_cc)
slope_f_cc, intercept_f_cc, r_value_f_cc, p_value_f_cc, std_err_f_cc = linregress(x_all_f_cc, y_all_f_cc)
a_i_cc = ufloat(slope_i_cc,std_err_i_cc)
a_tf_cc = ufloat(slope_tf_cc,std_err_tf_cc)
a_f_cc = ufloat(slope_f_cc,std_err_f_cc)

# Mostrar los resultados del ajuste lineal
print(f'Pendiente_i: {a_i_cc}') 
print(f'Pendiente_tf: {a_tf_cc}')
print(f'Pendiente_f: {a_f_cc}')
# print(f"Intersección_i: {intercept_i} - Intersección_f: {intercept_f}")

x_aux_i=np.linspace(-50,lim_ajuste_i,1000)
y_aux_i=lineal(x_aux_i,slope_i_cc,intercept_i_cc)
x_aux_tf=np.linspace(lim_ajuste_tf,lim_ajuste_f,1000)
y_aux_tf=lineal(x_aux_tf,slope_tf_cc,intercept_tf_cc)
x_aux_f=np.linspace(lim_ajuste_f,25,1000)
y_aux_f=lineal(x_aux_f,slope_f_cc,intercept_f_cc)

# fig,ax= plt.subplots(figsize=(9,5),constrained_layout=True)
# ax.plot(x_all_i_cc, y_all_i_cc, '.',label=f'i: [{min(x_all_i_cc)} ; {lim_ajuste_i}]°C')
# ax.plot(x_aux_i,y_aux_i,label=f'{a_i_cc:.1uf} ns/°C')
# ax.plot(x_all_tf_cc, y_all_tf_cc,'.', label=f'tf: [{lim_ajuste_i} ; {lim_ajuste_f}]°C')
# ax.plot(x_aux_tf,y_aux_tf,label=f'{a_tf_cc:.1uf} ns/°C')
# ax.plot(x_all_f_cc, y_all_f_cc, '.',label=f'f: [{lim_ajuste_f} ; {max(x_all_f_cc)}]°C')
# ax.plot(x_aux_f,y_aux_f,label=f'{a_f_cc:.1uf} ns/°C')

# plt.legend(ncol=3)
# plt.grid()
# plt.ylabel(r'$\tau$ (s)')
# plt.xlabel('Indx')
# plt.title(identif_4_cc)
# #plt.savefig('tau_vs_T_'+identif_4+'.png',dpi=300)
# plt.show()

# %%
