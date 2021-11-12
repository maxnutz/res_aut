

from sklearn import preprocessing

from FLUCCOplus.utils import *

from matplotlib.figure import Figure


def transform(support_points:list, hour_scale:int=8760):
    """
    returns the trigonometric polynomial fitting
    a given set of support points

    output dimension is always 8760 hours
    """

    h = hour_scale #hours of periodic timeframe
    points = np.array(support_points)
    N = len(points) #Number of support points
    x = np.arange(0, h, h/N, dtype=np.float64) #position of support points
    y = points  # support points of the scaler


    sum = np.fft.rfft(y) # calculate trigonometric polynomial of given support points (discrete fourier transform (dft))
    added_zeros = np.zeros(int((h-N)/2)) #fill the higher frequency fourier coefficients
    padded_sum = np.concatenate([sum, added_zeros])

    scaler = np.fft.irfft(padded_sum) * h / N #for given hour timescale
    reps = int(np.ceil(9000 / h)) #how often does the signal repeat in a year(9000 instead of 8760 to account for rounding

    return np.concatenate([scaler] * reps)[:8760] # copies the signal reps times and returns only the full year


def plot(scaler, support_points, hour_scale=8760, view_scale=1.5):
    p, h = support_points, hour_scale

    x = np.arange(0, h, h/len(p))
    xh = np.arange(0,8760,1)

    fig = Figure()
    ax = fig.add_subplot()
    a, b = 0, int(h*view_scale)
    ax.plot(x[a:b], p[a:b], "bo")
    ax.plot(xh[a:b], scaler[a:b], "r")
    return fig



@logg
def normalize(self, dataframe=None, columns=None):
    """
    :return dataframe (normalized):
    """
    df = self.df if dataframe is None else dataframe

    if columns is None:
        columns = self.vars + self.pps + self.pcs + self.cops
    print(columns)

    x = df[columns].values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    ndf = pd.DataFrame(x_scaled, columns=df[columns].columns, index=df.index)
    return ndf


#%%
