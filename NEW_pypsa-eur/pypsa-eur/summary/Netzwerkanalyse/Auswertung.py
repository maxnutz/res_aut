# installed packages: sys, subprocess
# path structure: pypsa-ordner /summary/Netzwerkanalyse
import yaml
import os
ordnerGES = os.getcwd() # working directory
with open(ordnerGES + '/config.yaml', 'r') as stream:
	config = yaml.safe_load(stream)

def install_prerequisites(install):
    import subprocess
    import sys
    reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
    installed_packages = [r.decode().split('==')[0] for r in reqs.split()]
    if 'pandas' not in installed_packages:
        print('package pandas needs to be installed\n')
        subprocess.run(['conda install -c anaconda pandas'], shell = True, universal_newlines = True)
    if 'pypsa' not in installed_packages:
        print('package pypsa needs to be installed\n')
        subprocess.run(['conda install -c conda-forge pypsa'], shell = True, universal_newlines = True)
    if 'numpy' not in installed_packages:
        print('new package numpy needs to be installed\n')
        subprocess.run(['conda install numpy'], shell = True, universal_newlines = True)
    if 'matplotlib' not in installed_packages:
        print('new package matplotlib needs to be installed\n')
        subprocess.run(['conda install -c conda-forge matplotlib'], shell = True, universal_newlines = True)
    if 'datetime' not in installed_packages: # datetime zum installieren findet das programm nicht!
        print('package datetime needs to be installed\n')
        subprocess.run(['conda install -c trentonoliphant datetime'], shell = True, universal_newlines = True)
    if 'xarray' not in installed_packages:
        print('package xarray needs to be installed\n')
        subprocess.run(['conda install -c anaconda xarray'], shell = True, universal_newlines = True)
    if 'openpyxl' not in installed_packages:
        print('package openpyxl needs to be installed\n')
        subprocess.run(['pip install openpyxl'], shell = True, universal_newlines = True)


install = config.get('install_packages')
if install == True:
    print('installing packages')
    install_prerequisites(install)


#installing prerequisites

import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import date, datetime, timedelta
import datetime as dt
from scipy.fft import fft, fftfreq
import math 
import operator
import os.path as osp
from FLUCCOplus import plots
import seaborn as sns

#import network and loads
pypsa_file = config.get('network').get('filename')
pypsa_file = ordnerGES + '/' + pypsa_file
n = pypsa.Network(pypsa_file)

###############################################################################
## computation of Residual load: load, not checked by EE or storage
renewables = ['solar', 'wind', 'ror', 'biomass']
stores = ['H2O', 'battery']#######
store_units = ['PHS', 'hydro']########
load = n.loads_t.p.filter(like='AT')
renload = n.generators_t.p.filter(like = 'geothermal').filter(like = 'AT').sum(axis=1)
for x in renewables: 
    renload += n.generators_t.p.filter(like = x).filter(like = 'AT').sum(axis=1)
for x in stores:
    renload += n.stores_t.p.filter(like=x).filter(like='AT').sum(axis=1)
for x in store_units:
    renload += n.storage_units_t.p.filter(like=x).filter(like='AT').sum(axis=1)
renload[renload<0]=0
renload = renload.to_frame(name = 'AT0 0')
res = load - renload
pres = np.maximum(res, 0)
nres = np.minimum(res, 0)
resWort = "Last, die nicht durch Strom erzeugt mit erneuerbaren Energieträgern oder Strom aus Speichern in Österreich gedeckt wird"
####################################################################################

## global variables and file system ###################################################
jahr = str(res.index[1].year)  #config.get('network').get('year')
#ordner = os.getcwd() #'/home/max/Dokumente/FH/Master_thesis/Netzmodell/pypsa-eur/summary/Netzwerkanalyse/result_files/'
if not os.path.exists(ordnerGES + '/result_files/'):
    os.mkdir(ordnerGES + '/result_files/')
    print('result_files done')
if not os.path.exists(ordnerGES + '/result_files/Plots'):
    os.mkdir(ordnerGES + '/result_files/Plots')
    print('result_files/Plots done')
files = ['Basic_Data', 'Frequenzanalyse', 'Lastverschiebung', 'Nulldurchgang', 'Residuallast', 'Zeitdauern', 'Jahresdauerlinie', 'Schaltsignale']
ordner = ordnerGES + '/result_files/'
for x in files:
    if not os.path.exists(ordner + 'Plots/' + x):
       os.mkdir(ordner + 'Plots/' + x)

#if not os.path.exists(ordner + 'Plots/Lastverschiebung'):
#    os.mkdir(ordner + 'Plots/Lastverschiebung')
#if not os.path.exists(ordner + 'Plots/Frequenzanalyse'):
#    os.mkdir(ordner + 'Plots/Frequenzanalyse')
#if not os.path.exists(ordner + 'Plots/Basic_Data'):
#    os.mkdir(ordner + 'Plots/Basic_Data')
excel_path = ordner + 'all_results.xlsx'
monate = [0, 8760/12, 2*8760/12, 3*8760/12, 4*8760/12, 5*8760/12, 6*8760/12, 7*8760/12, 8*8760/12, 9*8760/12, 10*8760/12, 11*8760/12, 8760]
months = ['Jänner', 'Februar', 'März', 'April', 'Mai', 'Juni', 'Juli', 'August', 'September', 'Oktober', 'November', 'Dezember']
#einjahr = []
einjahr = pd.date_range(start=date(int(jahr), 1, 1),end=date(int(jahr), 12, 31)).to_pydatetime().tolist()
silvester = datetime.strptime(jahr+'/'+'12'+'/31 23:00:00', '%Y/%m/%d %H:%M:%S')
#dayrun = datetime.strptime(jahr+'/'+'01'+'/01', '%Y/%m/%d')
#while dayrun <= datetime.strptime(jahr+'/'+'12'+'/31', '%Y/%m/%d'):
#    einjahr.append(dayrun)
#    dayrun += timedelta (days = 1)

## defining the result-excel-file and storage-places
df = pd.DataFrame([[date.today()], [datetime.now().strftime("%H:%M")], [jahr], [], ['Residuallast'],['Nulldurchgänge'],['Residuallast ja/nein'], ['Jahresdauerlinie'], ['Zeiträume einer EE/RES - Phase'], ['Einfache Modellierung einer Lastverschiebung', 'Es wird davon ausgegangen, dass die neg. RES in die nächste Phase der pos. RES verschoben werden kann.']],
                  index=["Erstellt am", "um", "Referenzjahr", " ", "RES", "ND", "RL", "JDL", "ZR", "LV"])
with pd.ExcelWriter(excel_path, datetime_format = 'DD.MM.YYYY') as writer:
    df.to_excel(writer, sheet_name = 'Allgemeines')
signaldf = pd.DataFrame(res['AT0 0'].values[:], index = res.index, columns = ['Residuallast'])
#jedes mal, wenn dieser Code-Teil durchlaufen wird, wird ein neues Excel-File angelegt
###############################################################################

## basic data ####################################################################################
def basic_data():
    plot_series(load, 'Stromverbrauch in Österreich', 'MW')
    plot_series(renload, 'Erzeugung erneuerbarer Energie in Österreich', 'MW')
    fourier(load, 'Last')
    fourier(renload, 'Erzeugung erneuerbarer Energie')
    autgens = n.generators_t.p.filter(like = 'AT')
    ITs=autgens.filter(like='solar').sum(axis=1).sum(axis=0)
    ITw=autgens.filter(like='wind').sum(axis=1).sum(axis=0)
    ITr=autgens.filter(like='ror').sum(axis=1).sum(axis=0)
	#gas zusammen
    ITg=autgens.filter(like='CCGT').sum(axis=1).sum(axis=0)+n.generators_t.p.filter(like = 'IT').filter(like='PCGT').sum(axis=1).sum(axis=0)
	#rest
    ITn=autgens.filter(like='nuclear').sum(axis=1).sum(axis=0)
    ITb=autgens.filter(like='biomass').sum(axis=1).sum(axis=0)
    ITgt=autgens.filter(like='geothermal').sum(axis=1).sum(axis=0)
    ITo=autgens.filter(like='oil').sum(axis=1).sum(axis=0)
    ITl=autgens.filter(like='lignite').sum(axis=1).sum(axis=0)
    ITow=autgens.filter(like='offwind').sum(axis=1).sum(axis=0)
    ITc=autgens.filter(like='coal').sum(axis=1).sum(axis=0)
    techs = ['solar','wind','ROR','coal','gas','rest']
    tech_erzeugung = pd.Series([ITs, ITw, ITr, ITc, ITg, (ITn+ITb+ITgt+ITo+ITl+ITow)], index=techs, name = 'Österreich')
    tech_cap=(n.generators[n.generators.bus.eq('AT0 0')].groupby('carrier').sum().filter(['p_nom_opt'])['p_nom_opt']+n.generators[n.generators.bus.eq('IT1 0')].groupby('carrier').sum().filter(['p_nom_opt'])['p_nom_opt'])
    tech_erzeugung = tech_erzeugung.sort_values(kind = 'quicksort', ascending = False)
    tech_cap = tech_cap.sort_values(kind = 'quicksort', ascending = False)
    plot_techs(tech_erzeugung, 'Technologien nach Stromerzeugung pro Jahr', 'MWh')
    plot_techs(tech_cap, 'Installierte Leistung nach Technologie', 'MW')
    ## zusammenfassendes txt
    with open(osp.dirname(osp.dirname(osp.dirname(osp.dirname(ordner)))) + '/config.yaml') as stream:
    	pypsaconfig = yaml.safe_load(stream)
    Iscenario = ('Scenario nach*: ' + config.get('network').get('scenario') + '\nReferenzjahr*: ' + config.get('network').get('year'))
    Iloads = ('\n\nLast:\n  - Referenzjahr des Verbrauchs: ' + str(res.index[1].year) + 
    		'\n  - Lineare Skalierung: ' + str(pypsaconfig.get('load').get('scaling_factor')) +
    		'\n  - zeitliche Anpassungen: ' + str(pypsaconfig.get('load').get('time_adjustments')) +
    		'\n  - Berechnungsbasis der Residuallast: ' + resWort)
    Itechs = ('\n\nTechnologie nach Stromerzeugung [MWh] :' +
    		'\n  - Solar: ' + str(round(ITs,0)) +
    		'\n  - Wind: ' + str(round(ITw,0)) +
    		'\n  - Wasser: ' + str(round(ITr,0)) +
    		'\n  - Nicht-erneuerbar: ' + str(round(ITg + ITo + ITl + ITc,0)))
    Icaps = ('\n\nInstallierte Kapazitäten [MW] : ' +
    		'\n  - Solar: ' + str(round(tech_cap['solar'],0)) +
    		'\n  - Wind: ' + str(round(tech_cap['onwind'],0)) +
    		'\n  - Wasser: ' + str(round(tech_cap['ror'],0)) + 
    		'\n\nSpeicher:\n')
    Ispeicher = (pd.DataFrame(data = [n.stores[n.stores.bus.eq('AT0 0 H2')].filter(['e_nom_opt']).values[0], n.stores[n.stores.bus.eq('AT0 0 battery')].filter(['e_nom_opt']).values[0], n.storage_units[n.storage_units.bus.eq('AT0 0')].groupby('carrier').sum().filter(['p_nom_opt']).values[0], n.storage_units[n.storage_units.bus.eq('AT0 0')].groupby('carrier').sum().filter(['p_nom_opt']).values[1]], index = ['H2','battery', 'PHS', 'hydro'], columns = ['nom. Power [MW]']))
    Igborder = ('\n\nGeografische Grenzen:' +
    		'\n  - Betrachtete  Länder: ' + str(pypsaconfig.get('countries')) +
    		'\n  - Geografische Genauigkeit: ' + str(pypsaconfig.get('scenario').get('clusters')) + ' Knotenpunkte für ' +
    		str(len(pypsaconfig.get('countries'))) + ' Länder. Dabei wird jedes Land von mindestens einem Netzknotenpunkt repräsentiert, Italien und Deutschland sind durch je 2 Netzknotenpunkte abgebildet.')
    Itborder = ('\n\nZeitliche Genauigkeit: ' + str(res.index[1] - res.index[0]) + ' mit festem Zeitintervall.')
    file = open((ordner +'/Zusammenfassung Netzwerk.txt'),"w")
    file.write('ZUSAMMENFASSUNG NETZWERK (' + str(dt.datetime.now()) + ')\n\n' + Iscenario + Iloads + Itechs + Icaps + str(Ispeicher) + Igborder + Itborder)
    file.close()
###########    		

def plot_series(data, title, unit):
    data.plot(figsize = (20,5), color = 'orange')
    plt.ylabel(unit)
    plt.title(title, fontsize = 23)
    plt.savefig(ordner + '/Plots/Basic_Data/' + title + '.png',
                format = 'png',
                bbox_inches = 'tight')
    plt.close()
    data.to_csv(ordner + title + '.csv') 
    with pd.ExcelWriter(excel_path, mode = 'a') as writer: 
        data.to_excel(writer, 
                        sheet_name = 'BD_' + title, 
                        header = [title + '[' + unit + ']'])
                        
def plot_techs(data, title, unit):
    data.plot(kind = 'bar', figsize=(20,5), color = 'orange')
    plt.ylabel(unit)
    plt.title(title, fontsize = 23)
    plt.savefig(ordner + '/Plots/Basic_Data/' + title + '.png',
                format = 'png',
                bbox_inches = 'tight')
    plt.close()
    data.to_csv(ordner + title + '.csv')
    with pd.ExcelWriter(excel_path, mode = 'a') as writer: 
        data.to_excel(writer, 
                        sheet_name = 'BD_' + title, 
                        header = [title + '[' + unit + ']'])


## Residual load #################################################################################
def Residuallast_per_hour(jahr, res):
    #print('function Residuallast_per_hour:    Begin')
    counts = res
    counts.to_csv(ordner + 'RES_pro_Stunde.csv') 
    with pd.ExcelWriter(excel_path, mode = 'a') as writer: 
        counts.to_excel(writer, 
                        sheet_name = 'RES pro Stunde', 
                        header = ['Residuallast [MWh]'])
    plot_Residuallast(counts, 'Stunde', 'h') # Graphing
    fourier(counts, 'Residuallast') # Frequenzanalyse
    # ein csv mit der stündlichen Residuallast

def Residuallast_per_day (jahr, silvester, pres, einjahr): 
    datum = datetime.strptime(jahr+'/'+'01'+'/01 00:00:00', '%Y/%m/%d %H:%M:%S') 
    counts=[] 
    while datum <= silvester:
        begin = datum
        end = datum + timedelta(days=1)
        datum = end
        data = res.loc[begin:end]
        summe = 0
        for v in range(1, data.size):
            summe += data.iloc[v, 0]
        counts.append(summe)
    counts = pd.DataFrame(data = counts, index = einjahr)
    plot_Residuallast(counts, 'Tag', 'd') #Graphing
    counts.to_csv(ordner + 'RES_pro_Tag.csv') 
    with pd.ExcelWriter(excel_path, mode = 'a') as writer: 
        counts.to_excel(writer, 
                        sheet_name = 'RES pro Tag', 
                        header = ['Residuallast [MWh]'])
    # ein csv mit der täglich summierten Residuallast => Energie pro Tag

def Residuallast_per_month (jahr, pres):
    counts = res.groupby(pres.index.month).sum() 
    counts['Monat'] = months 
    counts.set_index('Monat', drop=True, inplace = True)
    plot_Residuallast(counts, 'Monat', 'm')
    counts.to_csv(ordner + 'RES_pro_Monat.csv') 
    with pd.ExcelWriter(excel_path, mode = 'a') as writer: 
        counts.to_excel(writer, 
                        sheet_name = 'RES pro Monat', 
                        header = ['Residuallast [MWh]'])
    # ein csv mit der monatlich summierten Residuallast => Energie pro Monat

#für die durchschnittlichen Leistungen ganzes res[i] übergeben und +=1 durch += res[i] ersetzen
def count_per_month(v, nach_monat_zeit):
    point = res.index[v]
    for i in range(0, len(nach_monat_zeit)):
        if (v >= monate[i] and v < monate[i+1]): ##warum auch immer das so ist!
            for ask_run in range(0, 24):
                if ask_run == point.hour:
                    nach_monat_zeit[i][ask_run] +=1
    return(nach_monat_zeit)
    # run automatically! zählt die Anzahl pro Stunde nach monaten geordnet

def ND_und_RES_per_month_per_hour():
    ND_nmz = [[0]*24,[0]*24,[0]*24,[0]*24,[0]*24,[0]*24,[0]*24,[0]*24,[0]*24,[0]*24,[0]*24,[0]*24]
    RES_nmz = [[0]*24,[0]*24,[0]*24,[0]*24,[0]*24,[0]*24,[0]*24,[0]*24,[0]*24,[0]*24,[0]*24,[0]*24]
    for v in range(1, res.size):
        if (res['AT0 0'].iloc[v-1] < 0 and res['AT0 0'].iloc[v] > 0):
            ND_nmz = count_per_month(v, ND_nmz)
        elif(res['AT0 0'].iloc[v] > 0):
            RES_nmz = count_per_month(v, RES_nmz)
    ND_frame = pd.DataFrame(ND_nmz, columns = range(0,24), index = months)
    RES_frame = pd.DataFrame(RES_nmz, columns = range(0,24), index = months)
    if not os.path.exists(ordner + 'Plots/Residuallast/Analyse_nach_Tageszeit'):
    	os.mkdir(ordner + 'Plots/Residuallast/Analyse_nach_Tageszeit')
    for monat in months:
    	plot_ND_und_RES_mh(monat, ND_frame, RES_frame)
    with pd.ExcelWriter(excel_path, mode = 'a') as writer:
        ND_frame.to_excel(writer, 
                       sheet_name = 'ND nach Tageszeit_pro Monat')
    
    # ein csv das die Tageszeiten pro Monat enthält, mit der Anzahl der ND und RES-Abhängigkeit

def plot_ND_und_RES_mh(monat, ND_frame, RES_frame):
    fig,ax = plt.subplots(figsize=(20,5))
    title2 = ' '
    if monat != 'nan':
    	title2 = (' im ' + monat)
    plt.title('Residuallast-Zeiten und Nulldurchgänge' + title2, fontsize = 23)
    ax.plot(range(0, 24),
    		RES_frame.loc[monat],
    		color = 'lavender')
    ax.set_ylabel('Anzal der Stunden mit RES > 0')
    plt.ylim([0,30])
    plt.fill_between(range(0,24), RES_frame.loc[monat],
    		color = 'lavender')
    ax2 = ax.twinx()
    ax2.bar(range(0,24), 
    		ND_frame.loc[monat],
    		color = 'darkblue')
    ax2.set_ylabel('Anzahl Nulldurchgänge', color = 'darkblue')
    plt.ylim([0, 13])
    plt.savefig(ordner + '/Plots/Residuallast/Analyse_nach_Tageszeit/' + 'ND_und_RL' + title2 + '.png',
                format = 'png',
                bbox_inches = 'tight')
    plt.close()
# run automatically! 
# erstellt alle ND und RL Plots im Ordner Analys nach Tageszeit


def Residuallast_yn_per_day_per_hour(res, jahr):
    nach_monat_zeit = [[0] * 24, [0] * 24, [0] * 24, [0] * 24, [0] * 24, [0] * 24, [0] * 24, [0] * 24, [0] * 24, [0] * 24, [0] * 24, [0] * 24]
    for v in range(1, res.size):
        if (res['AT0 0'].iloc[v] > 0):
            point=res.index[v]
            for i in range(0, len(nach_monat_zeit)):
                if (v >= monate[i] and v < monate[i+1]):
                    for ask_run in range(0,24):
                        if ask_run == point.hour:
                            nach_monat_zeit[i][ask_run] += 1
    frame = pd.DataFrame(nach_monat_zeit, columns= range(0,24), index = months)
    frame.to_csv(ordner + 'RL_nach_Tageszeit_pro_Monat.csv')
    with pd.ExcelWriter(excel_path, mode = 'a') as writer:
        frame.to_excel(writer, 
                       sheet_name = 'RL nach Tageszeit_pro Monat')
    # ein csv, das die Tage enthält, die in dem Monat zu der Uhrzeit eine Residuallast entstanden ist.
def plot_Residuallast(counts, zeitraum, einheit):
    fig, ax = plt.subplots(figsize = (20,5))
    counts.plot(kind = 'bar',
                figsize = (20,5),
                ylabel = 'Energie [MWh]',
                xlabel = 'Zeit [' + einheit +']',
                color = 'darkblue',
                ax = ax) #'darkblue')
    xt = []
    for i in range(0, 12):
    	xt.append(i/12 * len(counts))
    ax.set_xticks(xt)
    ax.set_xticklabels(months, minor = False, rotation = 45)
    ax.legend(['RES'], loc = 1)
    plt.title("Menge an Residuallast pro " + zeitraum, fontsize = 23)
    plt.savefig(ordner + '/Plots/Residuallast/RES pro ' + zeitraum + '.png',
                format = 'png',
                bbox_inches = 'tight')
    plt.close()
    #print('plot_Residuallast:    Ende \n')
    # Plot und speicher von Residuallast-Funktionen
    # wird von den Funktionen selbstständig aufgerufen!

## Nulldurchgang #################################################################################
def Nulldurchgang_per_day(jahr, res, silvester, einjahr):
    #print('function Nulldurchgang_per_day:    Begin')
    datum = datetime.strptime(jahr+'/'+'01'+'/01 00:00:00', '%Y/%m/%d %H:%M:%S') 
    counts=[]
    while datum <= silvester: 
        begin = datum 
        end = datum + timedelta(days=1) 
        datum = end 
        data = res.loc[begin:end] 
        n = 0 
        for v in range(1, data.size): 
            if (data['AT0 0'].iloc[v-1] < 0 and data['AT0 0'].iloc[v] >= 0): 
                n += 1 
        counts.append(n)
    frame = pd.DataFrame(data = counts, index = einjahr)
    frame.to_csv(ordner + 'ND_per_day.csv')
    with pd.ExcelWriter(excel_path, mode = 'a') as writer:
        frame.to_excel(writer,  
                       sheet_name = 'ND per day', 
                       header = ['# Nulldurchgänge'])
    plot_Nulldurchgang(counts, einjahr, 'Tag')
    # ein csv mit der Anzahl an Nulldurchgänge an einem Tag

def Nulldurchgang_per_month(jahr, res, months):
    datum = jahr+'/'+'01'+'/01 00:00:00'
    counts=[]
    for i in range(2,14):
        month_begin = datetime.strptime(datum, '%Y/%m/%d %H:%M:%S')
        if i == 13:
            datum = jahr+'/'+str(12)+'/31 23:00:00'
        else:
            datum = jahr+'/'+str(i)+'/01 00:00:00'
        month_end = datetime.strptime(datum, '%Y/%m/%d %H:%M:%S')
        m=0
        monthdata = res.loc[month_begin:month_end]
        for v in range(1, monthdata.size):
            if (monthdata['AT0 0'].iloc[v-1] < 0 and monthdata['AT0 0'].iloc[v] >= 0):
                m += 1
        nPerd = m/(monthdata.size/24)
        counts.append(nPerd)
    frame = pd.DataFrame(data = counts, index = months)
    #if os.path.exists(ordner + 'ND_per_day_nach_Monat.csv'):
    #    return frame # bei zweimaligem Durchlaufen der Funktion
    frame.to_csv(ordner + 'ND_per_day_nach_Monat.csv')
    with pd.ExcelWriter(excel_path, mode = 'a') as writer:
        frame.to_excel(writer, 
                       sheet_name = 'ND per day nach Monat',  
                       header = ['# Nulldurchgänge/d'])
    plot_Nulldurchgang(counts, months, 'Monat')
    # ein csv mit der durchschnittlichen Anzahl an Nulldurchgängen pro Tag nach Monat

def plot_Nulldurchgang(counts, index, zeitraum):
    series = pd.Series(counts, index = index)
    series.plot.bar(figsize=(20,5),
    			ylabel = 'Nulldurchgänge pro Tag',
    			color = 'blue')
    plt.title('Anzahl der Nulldurchgänge pro ' + zeitraum, fontsize=23)
    plt.savefig(ordner + '/Plots/Nulldurchgang/' + 'ND pro ' + zeitraum + '.png',
                   format = 'png',
                   bbox_inches = 'tight') #falscher Plot, der hier gespeichert wird! 
    plt.close()


#def plot_Nulldurchgang(counts, months, einjahr):
#    if len(counts) == 12:
#        zeitraum = 'Monat'
#        index = monate #months
#    if len(counts) > 300:
#        zeitraum = 'Tag'
#        index = einjahr
#    series = pd.Series(counts, index = index)
#    series.plot.bar(figsize=(20,5), 
#                        ylabel = 'Nulldurchgänge pro Tag', 
#                        color = 'blue')
#    plt.title('Anzahl der Nulldurchgänge pro ' + zeitraum, fontsize = 23)
#    plt.savefig(ordner + '/Plots/Nulldurchgang/' + 'ND pro ' + zeitraum + '.png',
#                   format = 'png',
#                   bbox_inches = 'tight') #falscher Plot, der hier gespeichert wird! 
#    plt.close()
## Jahresdauerlinie #############################################################################
def Jahresdauerlinie(frame, art):
    for i in range(0, res.size):
        if frame.iloc[i]['AT0 0'] < 0:
            art = 'Residuallast'
    sort = frame.sort_values(by = 'AT0 0',
                             ascending = False,
                             ignore_index = True)
    sort.plot.area(stacked = False, 
                   figsize = (20,5),
              ylabel = 'Leistung [MW]',
              xlabel = 'Zeit [h]',
              color = 'darkblue')
    plt.axhline(y = 0, c = 'blue')
    plt.title('Jahresdauerlinie der ' + art, fontsize = 23)
    plt.savefig(ordner + '/Plots/Jahresdauerlinie/' + 'JDL der ' + art + '.png',
                   format = 'png',
                   bbox_inches = 'tight')
    plt.close()
    sort.to_csv(ordner + 'JDL - ' + art + '.csv')
    with pd.ExcelWriter(excel_path, mode = 'a') as writer:
        frame.to_excel(writer, 
                       sheet_name = 'JDL - ' + art,  
                       header = ['Leistung [MW]'])
    # aufrufbar für res, pres und renload (bei renload auch load übergeben!)
    
## Zeitdauern ####################################################################################
def ZR_RES(res):
    dauer = np.zeros(300)
    end = 0
    begin = 0
    for v in range(0, res.size):
        if res['AT0 0'].iloc[v-1] < 0 and res['AT0 0'].iloc[v] >= 0:
            if v - end > 1:
                dauer[end-begin] +=1
                begin = v
            for w in range(v, res.size):
                if res['AT0 0'].iloc[w] < 0:
                    end = w
                    break
    build_dataframe(dauer, 'positiv') # macht ein der Groesse entsprechendes DataFrame aus Zeitdauern
    # ein csv mit der Häufigkeit, mit der eine Zeitdauer RES vorkommt
    
def ZR_EE(res):
    dauer = np.zeros(300)
    end = 0
    begin = 0
    for v in range(0, res.size-1):
    	if res['AT0 0'].iloc[v-1] > 0 and res['AT0 0'].iloc[v] <= 0:
            if v - end > 1:
                dauer[end-begin] +=1
                begin = v
            for w in range(v, res.size-1):
                if res['AT0 0'].iloc[w] > 0:
                    end = w
                    break
    build_dataframe(dauer, 'negativ')
    # ein csv mit der Häufigkeit, mit der eine Zeitdauer EE vorkommt
    
def build_dataframe(dauer, RES):
    summe = 0
    run = len(dauer)-1
    while run > 0:
    	summe += dauer[run]
    	if summe > 20:
    		break
    	run -= 1
    frame = pd.DataFrame(dauer[1:run+1])
    frame.index = np.arange(1, len(frame)+1)
    frame.plot(kind = 'bar', figsize = (20,5),
    			ylabel = 'Anzahl',
    			xlabel = 'Zeitdauer [h]',
    			color = 'darkorange')
    plt.title('Zeitdauern einer Phase ' + RES + 'er Residuallast', fontsize = 23)
    plt.savefig(ordner + '/Plots/Zeitdauern/' + 'ZD ' + RES + '. RES.png',
                   format = 'png',
                   bbox_inches = 'tight')
    plt.close()
    frame.to_csv(ordner + 'ZD - RES ' + RES + '.csv')
    with pd.ExcelWriter(excel_path, mode = 'a') as writer:
        frame.to_excel(writer, 
                       sheet_name = 'ZD - RES ' + RES,  
                       header = ['Anzahl der Zeitdauer [# h]'])
    # speichert und plottet die Zeitdauer (wird von den funktionen aufgerufen!)

## Lastverschiebungssimulation ###################################################################
def count_power(begin, end):
    sum = 0
    for i in range(begin, end):
        sum += res['AT0 0'].iloc[i]
    return sum

def Lastverschiebung_einfach(res):
    v=0
    lastv = 0
    w=0
    RES=0
    EE=0    
    cols = ['Anfang', 'Ende', 'RES', 'EE', 'Wechsel_EE-RES']
    allframe = pd.DataFrame(columns=cols)
    for v in range(1, res.size):
    	if (res['AT0 0'].iloc[v-1] > 0 and res['AT0 0'].iloc[v] <= 0):
    		RES = count_power(w,v)
    		df2 = pd.DataFrame([[lastv, v, RES, EE, w]], columns = cols)
    		for w in range(v, res.size):
    			if res['AT0 0'].iloc[w-1] <= 0 and res['AT0 0'].iloc[w] > 0:
    				EE = count_power(v,w)
    				allframe = allframe.append(df2, ignore_index = True)
    				lastv = v
    				break
    allframe['Delta_t'] = allframe ['Ende'] - allframe['Anfang']
    allframe['Delta_E'] = allframe['RES'] + allframe['EE']
    LVp = [0] * allframe.index.size # variablen erst jetzt definieren, wo klar ist, wie lange das df ist
    LVl = [0] * allframe.index.size
    P_kurz = [0] * allframe.index.size
    P_lang = [0] * allframe.index.size
    for i in range(0,allframe.index.size):
    	res = allframe['RES'].iloc[i]
    	ee = allframe['EE'].iloc[i]
    	t = allframe['Delta_t'].iloc[i]
    	LVp[i] = min([res,ee], key = abs)
    	LVl[i] = max([res, ee], key = abs) - LVp[i]
    	LVp[i] = abs(LVp[i])
    	P_kurz[i] = LVp[i] / t
    	P_lang[i] = LVl[i] / t
    allframe['LV_kurz'] = LVp
    allframe['LV_lang'] = LVl
    allframe['P_kurz'] = P_kurz
    allframe['P_lang'] = P_lang
    allframe['Anteil_kurz'] = abs(allframe['LV_kurz'] / (allframe['LV_kurz'] + abs(allframe['LV_lang'])))
    allframe['Anteil_lang'] = abs(allframe['LV_lang'] / (allframe['LV_kurz'] + abs(allframe['LV_lang'])))
    allframe.to_csv(ordner + 'Lastverschiebung_einfach.csv')
    with pd.ExcelWriter(excel_path, mode = 'a') as writer:
        allframe.to_excel(writer, 
                       sheet_name = 'LV - Residuallast')
    plot_Lastverschiebung('Kurzzeitige Verschiebung (innerhalb eines Zyklus)', allframe['Delta_t'], allframe['P_kurz'], allframe['Anfang'], allframe['Ende'])
    plot_Lastverschiebung('Residuallast bei Lastverschiebung innerhalb eines Zyklus', allframe['Delta_t'], allframe['P_lang'], allframe['Anfang'], allframe['Ende'])
   # einfache Lastverschiebung innerhalb eines Zyklus: ein csv das alle dafür relevanten berechneten Größen enthält

def plot_Lastverschiebung(beschreibung, dauer, power, start, end):
    color = 'orange' if beschreibung == 'Kurzzeitige Verschiebung (innerhalb eines Zyklus)' else 'red'
    	
    fig,ax = plt.subplots(figsize=(20,5))
    ax.plot([0,100],[0,100], linewidth = 0.0003)
    ax.set_yscale('symlog')
    for v in range(0, power.size):
    	if color != 'orange':
            color = 'green' if power[v] < 0 else 'red'
    	ax.add_patch(Rectangle((start[v],0),
    			width = dauer[v],
    			height = power[v],
    			color = color))
    plt.title(beschreibung, fontsize = 23)
    xt = []
    for i in range(0, 12):
        xt.append(int(i/12 * end[end.size-1]))
    ax.set_xticks(xt)
    ax.set_xticklabels(months, minor = False, rotation = 45)
    plt.xlabel('Zeit [h]')
    plt.ylabel('Leistung [MW]')
    plt.savefig(ordner + '/Plots/Lastverschiebung/' + beschreibung +'.png',
                   format = 'png',
                   bbox_inches = 'tight')
    plt.close()
    # plots lastverschiebungssimulation => runs automatically! 
# es gäbe noch eine zweite gute Plotfunktion zum Einbauen (anteile)
## Frequenzanalyse ###############################################################################
def fourier(data, data_type):
    N = data.index.size
    T = 1/8760 #restriction
    x = np.linspace(0.0, N*T, N, endpoint = False)
    y = data['AT0 0']
    yf = np.fft.fft(y) #fourier transform
    xf = fftfreq(N, T)[:N//2] #sample frequencies
    df = pd.DataFrame(data = (2.0/N * np.abs(yf[0:N//2])), index = xf, columns = ['f'])
    df = find_maxima(df)
    plot_fourier(df, data_type)
    df.to_csv(ordner + 'Frequenzen_' + data_type + '.csv')
    with pd.ExcelWriter(excel_path, mode = 'a') as writer: 
        df.to_excel(writer, 
                        sheet_name = 'f(' + data_type + ')')
    # one csv with leading frequencies of data
    # callable for all data in hourly resolution

def find_maxima(df):
    ## variable parameters for finding maxima
    delT = 2
    scalV = 0.5
    break_point = (df['f'][10:365].max() * 3.3) / (df['f'][10:365].mean() * 1)
    df['max'] = np.nan
    for v in range(10, 365):
        if df['f'][v] > break_point:
            for w in range(v, df['f'].size):
                if df['f'][w] < break_point:
                    maxindex = df['f'][v:w].idxmax()
                    maxvalue = df['f'][v:w].max()
                    if ((df['f'][maxindex-delT] * scalV < maxvalue) & (df['f'][maxindex+delT] * scalV < maxvalue)):
                        df['max'][maxindex] = maxvalue
                    break
    return(df)
    # function to find maximum of column 'f' of pd.DataFrame (runs automatically from fourier(data, type))

def plot_fourier(df, name):
    df['f'].plot(linewidth = 5,
      figsize = (20,5),
      color = 'grey')
    for v in range(0, 366):
        if not math.isnan(df['max'][v]):
            plt.text(x = v+10, y = df['max'][v], s = ('f =' + str(v)), fontsize = 12)
    plt.scatter(x = df.index, y = df['max'], color = 'r', s = 200)
    plt.xlim(1, 400)
    plt.ylim(0, df['f'][1:365].max() + 5e+2)
    plt.xlabel('Frequenz [1/a]')
    plt.ylabel('Spektrale Dichte [MWh]')
    plt.title('Vorherrschende Frequenzen der ' + name, fontsize = 23)
    plt.savefig(ordner + '/Plots/Frequenzanalyse/FREQ_' + name + '.png', 
                format = 'png',
                bbox_inches = 'tight')
    plt.close()
    #Plot for fourier Analyse (NUR von fourier(data, data_type) aufrufen!!)

## Schaltsignale ##############################################################################
def compare(mean, res, i, comparison): 
    n = 2 # Anzahl an Stunden, die zusammen betrachtet werden sollen
    found = False
    ops = {'>' : operator.gt, '<' : operator.lt, '==' : operator.eq}
    if i+n >= signaldf.index.size:
        i = signaldf.index.size-n
    for j in range(i, i+n):
        if not ops[comparison](mean[j], res[j]):
            found = True
        return not found
# compares the entries given at point i for the next n points

def mean_calcs(signaldf):
    n = 5
    signaldf['mean'] = signaldf['Residuallast'].rolling(n).mean().shift(-int(n/2))
    m = 24
    signaldf['mean_24h'] = signaldf['Residuallast'].rolling(m).mean().shift(-int(m/2))
    signaldf['max'] = signaldf['mean'].abs().rolling(n).max(std = n)
    signaldf['normed_mean'] = signaldf['mean'] / signaldf['max']
    return signaldf


def ressignal1 (signaldf):
    signaldf = mean_calcs(signaldf)
    signalseries = pd.Series(np.nan * signaldf.index.size, index = signaldf.index)
    for i in range(0, signaldf.index.size):
        if compare(signaldf['normed_mean'], [[0]]*signaldf.index.size, i, '<'):
            signalseries[i] = 1
        if compare(signaldf['normed_mean'], [[0]]*signaldf.index.size, i, '>') or compare(signaldf['normed_mean'], [[0]]*signaldf.index.size, i, '=='):
            signalseries[i] = 0
    signaldf['RES1'] = signalseries
    signaldf['RES1'].to_csv(ordner + 'Schaltsignal_RES1.csv')
    with pd.ExcelWriter(excel_path, mode = 'a') as writer: 
        signaldf.to_excel(writer, 
                        sheet_name = 'Schalts.RES1')
    plot_Schaltsignale('Residuallast_1', signaldf, signaldf['RES1'])
# Signal entsteht NUR durch den running avg der Residuallast
    
def ressignal2 (signaldf):
    signaldf = mean_calcs(signaldf)
    signalseries = pd.Series(np.nan * signaldf.index.size, index = signaldf.index)
    for i in range(0, signaldf.index.size):
        if compare(signaldf['mean_24h'], signaldf['Residuallast'], i, '>'):
            signalseries[i] = 1
        elif compare(signaldf['mean_24h'], signaldf['Residuallast'], i, '<') or compare(signaldf['mean_24h'], signaldf['Residuallast'], i, '=='):
            signalseries[i] = 0
    signaldf['RES2'] = signalseries
    signaldf['RES2'].to_csv(ordner + 'Schaltsignal_RES2.csv')
    with pd.ExcelWriter(excel_path, mode = 'a') as writer: 
        signaldf.to_excel(writer, 
                        sheet_name = 'Schalts.RES2')
    plot_Schaltsignale('Residuallast_2', signaldf, signaldf['RES2'])
# Signal entsteht NUR durch das Verhältnis von RES zum running avg von RES

def ressignal3 (signaldf):
    res_load = res['AT0 0']
    signalseries = pd.Series(np.nan * signaldf.index.size, index = signaldf.index)
    def signal1(res_load, i): #True, wenn der Mittelwert kleiner 0 ist
    	if (running_avg(res_load, i, i+2) > 0):
    		return False
    	return True
    	
    def signal2(res_load, i): #True wenn der Wert kleiner als der running avg ist und RES in den nächsten Stunden nicht stark negativ wird
    	if (res_load[i] > running_avg(res_load, i-3, i+6)): #Begründung: längerfristig sinnvoller (?)
    		return False
    	if (running_avg(res_load, i+6, i+12) < 0):
    		return False
    	return True

    def running_avg(res_load, a, b): # returns running avg
    	if a > (res_load.size - 12) or b > (res_load.size - 12):
    		return 0
    	sum = 0
    	for i in range(a, b):
    		sum += res_load[i]
    	return(sum/(b-a))

    for i in range(0, res.index.size):
    	if res_load[i] < 0: #function RES1
    		if signal1(res_load, i):
    			signalseries[i] = 1
    		else:
    			signalseries[i] = 0
    	elif res_load[i] >= 0:
    		if running_avg(res_load, i+2, i+5) < 0:
    			signalseries[i] =  0 #kurze RES-Phase => 0
    		else:
    			signalseries[i] = 1 if signal2(res_load, i) else 0
    	else:
    		print('### FUNCTION SIGNAL3: NOT RECOGNIZED ', i, ' ###')
    signaldf['RES3'] = signalseries
    signaldf['RES3'].to_csv(ordner + 'Schaltsignal_RES3.csv')
    with pd.ExcelWriter(excel_path, mode = 'a') as writer: 
        signaldf.to_excel(writer, 
                        sheet_name = 'Schalts.RES3')
    plot_Schaltsignale('Residuallast_3', signaldf, signaldf['RES3'])	

def plot_Schaltsignale(name, signaldf, signal):
    plotting = pd.DataFrame(columns = ['Zeiten', 'a', 'b'])
    plotting['Zeiten'] = pd.Series(['ein_Sommermonat', 'ein_Wintermonat', '2_Apriltage', '2_Wintertage', '2_Sommertage'])
    plotting['a'] = pd.Series([180, 30, 90, 58, 208])
    plotting['b'] = pd.Series([210, 60, 92, 60, 210])
    for t in range(0, plotting.index.size):
        a = plotting['a'][t]*24
        b = plotting['b'][t]*24
        plt.figure(figsize=(40,10))
        ax = plt.subplot(111)
        plt.title('Signal ' + name + ' für ' + plotting['Zeiten'][t], fontsize=23)
        signaldf['Residuallast'][a:b].plot(ax = ax)
        signaldf['mean_24h'][a:b].plot(ax = ax)
        (signal[a:b]*10000).plot(alpha = 0.8, ax = ax)
        plt.axhline(y = 0, c = 'r', alpha = 0.5, linewidth = 0.5)
        plt.legend(loc = 1)
        if not os.path.exists(ordner + 'Plots/Schaltsignale/' + name):
        	os.mkdir(ordner + 'Plots/Schaltsignale/' + name)
        plt.savefig(ordner + 'Plots/Schaltsignale/' + name + '/' + plotting['Zeiten'][t] + '.png', 
                format = 'png',
                bbox_inches = 'tight')
        plt.close()
        
run = config.get('evaluation')
basic_data()

if run.get('residual_load_ordered') == True:
	Residuallast_per_hour(jahr, res)
	Residuallast_per_day(jahr, silvester, pres, einjahr)
	Residuallast_per_month(jahr, pres)
	Residuallast_yn_per_day_per_hour(res, jahr)
if run.get('zero_crossings') == True:
	Nulldurchgang_per_day(jahr, res, silvester, einjahr)
	Nulldurchgang_per_month(jahr, res, months)
	ND_und_RES_per_month_per_hour()
if run.get('duration_line') == True:
	Jahresdauerlinie(res, 'Residuallast')
	Jahresdauerlinie(pres, 'positiven Residuallast')
	Jahresdauerlinie(nres.abs(), 'negativen Residuallast')
	Jahresdauerlinie(renload, 'erzeugung erneuerbarer Energie')
if run.get('duration_statistics') == True:
	ZR_EE(res)
	ZR_RES(res)
if run.get('load_shifting_simulation') == True:
	Lastverschiebung_einfach(res)
if run.get('signals') == True:
	ressignal1(signaldf)
	ressignal2(signaldf)
	ressignal3(signaldf)	
	

