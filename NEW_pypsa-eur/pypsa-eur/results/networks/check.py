import pypsa
import pandas as pd
import matplotlib.pyplot as plt
n = pypsa.Network('elec_uba2050_2lines.nc')

#new_generators = n.generators.replace(
#	to_replace = n.generators.p_nom#.filter(like = 'AT0 0 ror').values,
#	value = 5.0e+03)
#n = n.replace(to_replace = n.generators,
#	value = new_generators)
#print(n.generators.p_nom.filter(like = 'AT0 0 ror'), '\n', n.loads_t.p.filter(like = 'AT'))
#n.generators.at['AT0 0 ror', 'p_nom'] = 5000
plt.figure()
n.generators_t.p.filter(like = 'AT').plot.area(figsize=(20,5))
plt.show()
