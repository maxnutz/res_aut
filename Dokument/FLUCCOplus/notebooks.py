from FLUCCOplus.utils import *
from FLUCCOplus.config import *

# plt.style.use('fivethirtyeight')


sns.set_style("whitegrid")

sns.set_context("paper",
                font_scale=1.,
                rc={"lines.linewidth": 0.7})

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

if VERBOSE:
    print(__name__, "loaded!")