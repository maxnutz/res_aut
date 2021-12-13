import pandas as pd
import matplotlib.pyplot as plt
from FLUCCOplus import *

df=pd.DataFrame(index = [0,2,3,4,5,6,7,8,9,10,11,12,13,25,25],
	data = [0.000,0.000,0.005,0.150,0.300,0.525,0.905,1.375,1.950,2.580,2.960,3.050,3.060,3.060,0.000],
	columns = ['V112 3MW'])
df.plot()
plt.xlabel('velocity [m/s]')
plt.ylabel('power [MW]')
plt.title('Verwendete Leistungskurve der Windkraftanlage')
plt.show()
