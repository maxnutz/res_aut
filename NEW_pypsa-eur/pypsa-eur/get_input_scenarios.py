import pandas as pd

def mwh_to_mwh(data, exportdf, v):
    for t in ['wind', 'solar', 'ror']:
    	if type(data[t].loc[v]) == str:
    		print('ERROR: bad input format: this is often caused by bad excel formatting: open input-file with text-editor and remove  -> " <- ')
    		break
    	exportdf[t].loc[v] = data[t].loc[v]
    return exportdf

def twh_to_mwh(data, exportdf, v):
    for t in ['wind', 'solar', 'ror']:
    	if type(data[t].loc[v]) == str:
    		print('ERROR: bad input format: this is often caused by bad excel formatting: open input-file with text-editor and remove  -> " <- ')
    		break
    	exportdf[t].loc[v] = data[t].loc[v] * 1.0e+6
    return exportdf

def pj_to_mwh(data, exportdf, v):
    for t in ['wind', 'solar', 'ror']:
    	if type(data[t].loc[v]) == str:
    		print('ERROR: bad input format: this is often caused by bad excel formatting: open input-file with text-editor and remove  -> " <- ')
    		break
    	exportdf[t].loc[v] = data[t].loc[v] * 277778
    return exportdf

def gwh_to_mwh(data, exportdf, v):
    for t in ['wind', 'solar', 'ror']:
    	if type(data[t].loc[v]) == str:
    		print('ERROR: bad input format: this is often caused by bad excel formatting: open input-file with text-editor and remove  -> " <- ')
    		break
    	exportdf[t].loc[v] = data[t].loc[v] *1.0e+3
    return exportdf



data = pd.read_csv('Scenarios.csv', comment = '#',
			skip_blank_lines = True,
			skiprows = 2).set_index('Name')
exportdf = pd.DataFrame(index = data.index, columns = ['consumption[per]', 'wind', 'solar', 'ror'])
print(data, '\n')

for v in data.index:
	unit = data['unit'].loc[v]
	if unit == 'PJ' or unit == 'pj' or unit == 'Pj':
		exportdf = twh_to_mwh(data, exportdf, v)
	elif unit == 'TWh' or unit == 'twh' or unit == 'TWH' or unit == 'Twh':
		exportdf = twh_to_mwh(data, exportdf, v)
	elif unit == 'MWh' or unit == 'mwh' or unit == 'MWH' or unit == 'Mwh':
		exportdf = mwh_to_mwh(data, exportdf, v)
	elif unit == 'GWh' or unit == 'gwh' or unit == 'GWH' or unit == 'Gwh':
		exportdf = gwh_to_mwh(data, exportdf, v)
	else:
		print('ERROR, cant read unit!')
exportdf['consumption[per]'] = data['consumption[per]']
exportdf['cap_ror[MW]'] = exportdf['ror']/3.4e+3
exportdf.to_csv('scenarios_input.csv', sep = ',', na_rep = 'nan')
