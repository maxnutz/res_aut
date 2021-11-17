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
    x = {'Längstes neg. Signal [h]' : neg_phase.max(),
         'Mean neg. Signal [h]' : round(neg_phase.mean(), 2),
         'Längstes pos. Signal [h]' : pos_phase.max(),
         'Mean pos. Signal [h]' : round(pos_phase.mean(), 2)}
    return x

def statistics(sdf):
    stats = pd.DataFrame(columns = ['Anteil pos. Signal',
                                    'Anteil neg. Signal',
                                    'Längstes neg. Signal [h]',
                                    'Mean neg. Signal [h]',
                                    'Längstes pos. Signal [h]',
                                    'Mean pos. Signal [h]'])
    for s in sdf[1]:
        Pp = round(s['signal'].sum()/s.index.size, 2) #positive part
        Np = round((s.index.size-s['signal'].sum())/s.index.size, 2) #negative part
        dic = {**{'Anteil neg. Signal' : Np, 'Anteil pos. Signal' : Pp}, **length_stats(s['signal'])}
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
    return pd.DataFrame({'pos Signal' : pos_phase, 'neg Signal' : neg_phase})

def signal_res(sdf, resload):
    PosSNegR = 0
    PosSPosR = 0
    NegSNegR = 0
    NegSPosR = 0
    NegRes = 0
    PosRes = 0
    stats = pd.DataFrame(columns = ['Pos. Signal RES > 0',
                                'Pos. Signal RES < 0',
                                'Neg. Signal RES > 0',
                                'Neg. Signal RES < 0',
                                'Anteil neg. Signal RES < 0',
                                'Anteil pos. Signal RES > 0'])
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
        x = {'Pos. Signal RES > 0' : PosSPosR,
            'Pos. Signal RES < 0' : PosSNegR,
            'Neg. Signal RES > 0' : NegSPosR,
            'Neg. Signal RES < 0' : NegSNegR,
            'Anteil neg. Signal RES < 0' : round(NegSNegR/NegRes, 3),
            'Anteil pos. Signal RES > 0' : round(PosSPosR/PosRes, 3)}
        stats = stats.append(x, ignore_index = True)
    stats.insert(loc = 0, column = 'Name', value = sdf[0])
    return(stats)

