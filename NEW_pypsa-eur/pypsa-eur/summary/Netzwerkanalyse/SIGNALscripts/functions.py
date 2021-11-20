import pandas as pd
import numpy as np

name = "functions"

def import_func(dateien, names):
    signals = []
    for d in dateien: 
        s = pd.read_csv('result_files/' + d, header = 1, names = ['snapshots', 'signal'])
        signals.append(s)
    return [names, signals]


def check_input(signals):
    for s in signals:
        if s['signal'].sum() < 1 or s['signal'].sum() > s.index.size:
            return False
        if s['signal'].max() > 1 or s ['signal'].min() < 0:
            return False
        return True

def length_stats(signal):
    pb = 0
    pe = 0
    neg_phase = np.array(0)
    pos_phase = np.array(0)
    for v in range(1, signal.size):
        if signal[v-1] == 0 and signal[v] > 0:
            pb = v
            neg_phase = np.append(neg_phase, v-pe) # von pe bis pb jetzt
        if signal[v-1] > 0 and signal[v] == 0:
            pe = v
            pos_phase = np.append(pos_phase,v-pb)#von pb bis pe jetzt
    x = {'Längste Sperre [h]' : neg_phase.max(),
         'Mean  Sperre[h]' : round(neg_phase.mean(), 2),
         'Längste Freigabe [h]' : pos_phase.max(),
         'Mean Freigabe [h]' : round(pos_phase.mean(), 2)}
    return x

def statistics(sdf):
    stats = pd.DataFrame(columns = ['Anteil Freigabe',
                                    'Anteil Sperre',
                                    'Längste Sperre [h]',
                                    'Mean Sperre [h]',
                                    'Längste Freigabe [h]',
                                    'Mean Freigabe [h]'])
    for s in sdf[1]:
        Pp = round(s['signal'].sum()/s.index.size, 2) #positive part
        Np = round((s.index.size-s['signal'].sum())/s.index.size, 2) #negative part
        #dic = {**{'Anteil Sperre' : Np, 'Anteil Freigabe' : Pp}, **length_stats(s['signal'])}
        dic = {**{'Anteil Freigabe' : Pp, 'Anteil Sperre' : Np}, **length_stats(s['signal'])}
        stats = stats.append(dic, ignore_index = True)
    stats.insert(loc=0, column = 'Name', value = sdf[0])
    return(stats)

def all_lengths(signal):
    pb = 0
    pe = 0
    neg_phase = np.zeros(400)
    pos_phase = np.zeros(400)
    for v in range(1, signal.size):
        if signal[v-1] == 0 and signal[v] > 0:
            pb = v
            neg_phase[v-pe] += 1 # von pe bis pb jetzt
        if signal[v-1] > 0 and signal[v] == 0:
            pe = v
            pos_phase[v-pb] += 1 #von pb bis pe jetzt
    return pd.DataFrame({'Freigabe' : pos_phase, 'Sperre' : neg_phase})

def signal_res(sdf, resload):
    PosSNegR = 0
    PosSPosR = 0
    NegSNegR = 0
    NegSPosR = 0
    NegRes = 0
    PosRes = 0
    stats = pd.DataFrame(columns = ['Freigabe RES > 0',
                                'Freigabe RES < 0',
                                'Sperre RES > 0',
                                'Sperre RES < 0',
                                'Anteil Sperre RES < 0',
                                'Anteil Freigabe RES > 0'])
    for s in sdf[1]: 
        for i in range (0,len(s['signal'])):
            if resload['signal'][i] < 0:
                NegRes += 1
                if s['signal'][i]:
                    PosSNegR += 1
                else:
                    NegSNegR += 1
            else:
                PosRes += 1
                if s['signal'][i]:
                    PosSPosR += 1
                else:
                    NegSPosR += 1
        x = {'Freigabe RES > 0' : PosSPosR,
            'Freigabe RES < 0' : PosSNegR,
            'Sperre RES > 0' : NegSPosR,
            'Sperre RES < 0' : NegSNegR,
            'Anteil Sperre RES < 0' : round(NegSNegR/NegRes, 3),
            'Anteil Freigabe RES > 0' : round(PosSPosR/PosRes, 3)}
        stats = stats.append(x, ignore_index = True)
    stats.insert(loc = 0, column = 'Name', value = sdf[0])
    return(stats)

