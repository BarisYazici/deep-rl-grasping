import argparse

# import pandas as pd
import pandas
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
import seaborn
from matplotlib.ticker import FuncFormatter
import json
# import matplotlib.cm as cm


def check_positive(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def millions(x, pos):
    """
    Formatter for matplotlib
    The two args are the value and tick position

    :param x: (float)
    :param pos: (int) tick position (not used here
    :return: (str)
    """
    return '{:.1f}M'.format(x * 1e-6)


def closestNumber(n, m) : 
    # Find the quotient 
    q = int(n / m) 
      
    # 1st possible closest number 
    n1 = m * q 
      
    # 2nd possible closest number 
    if((n * m) > 0): 
        n2 = (m * (q + 1))  
    else: 
        n2 = (m * (q - 1)) 
  
    # if true, then n1 is the required closest number 
    if (abs(n - n1) < abs(n - n2)) : 
        return n1 
      
    # else n2 is the required closest number  
    return n2

class PlotShaded:
    def __init__(self):
        self.algos = []
        self.monitor = []

    def plot(self, datas, title, pad=100):
        # colors = cm.rainbow(np.linspace(0, 1, 20))
        colors = ["lightcoral", "cornflowerblue", "orange", "darkseagreen", "brown", "darkgreen"]
        for i, (data, monitor) in enumerate(datas):
            if monitor:
                x = data.l
                y = data.s
                print(y.shape)
                x = np.cumsum(x)
                # x, y= self.smooth((x, y), window=1000)
                x, y = self.smooth((x, y), window=1000)
            else:
                x = "total_timesteps"
                y = "success_rate"
                x = data.total_timesteps
                y = data.success_rate
                # y = data.mean_rew
                x, y = self.smooth((x, y), window=50)
            fig = plt.figure(title)
            plt.title(args.title)
            formatter = FuncFormatter(millions)
            plt.xlabel('Number of Timesteps')
            plt.ylabel("Success Rate")
            if pad is not 0:
                plt.plot(x, y, label=self.algos[i], color= colors[i])
                print("PAD VARIANCE")
  
                print(closestNumber(len(y), pad))
                print("length", len(y))
                divider = closestNumber(len(y), pad)
                # y = y.to_numpy()
                if divider > len(y):
                    divider = closestNumber(len(y), pad) - pad
                y = y[:closestNumber(len(y), divider)]
                print("y length", len(y))
                print("X TYPE", type(x))
                x = x.to_numpy()
                
                print("X TYPE", type(x))
                x = x[:closestNumber(len(y), pad)]
                print("length", len(y))
                y = y.reshape(pad, -1)
                x = x.reshape(pad, -1)
                print("first x", x[0])
                print("y shape", y.shape)
                success_mean = np.mean(y, axis=1)
                vec_std_vars = np.std(y, axis=1)
                # vec_std_vars = np.transpose(vec_std_vars)
                # timestep_array = np.linspace(1, len(x)+ 1, len(x))
                # plt.plot(x, success_mean, label=self.algos[i])

                # print(algos[i], vec_std_vars)
                for a in range(pad):
                    plt.fill_between(x[a], y[a] - vec_std_vars[a], y[a] + vec_std_vars[a], alpha=0.5,color=colors[i])
                
                # plt.fill_between(y - vec_std_vars, y + vec_std_vars, alpha=0.5)#, color=colors)
            else:
                plt.plot(x, y, label=self.algos[i])# color= colors[i])

                # std_success = np.std(y)

                # plt.fill_between(x, y - std_success, y + std_success, alpha=0.5)#,color=colors[i])

            fig.axes[0].xaxis.set_major_formatter(formatter)
            # plt.xlim(1000000)

        plt.legend()
        plt.show()

    def read_file(self, path: str) -> pandas.DataFrame:
        """
        Load all Monitor logs from a given directory path matching ``*monitor.csv`` and ``*monitor.json``

        :param path: (str) the directory path containing the log file(s)
        :return: (pandas.DataFrame) the logged data
        """
        data_frames = []
        str_path = path.split('/')
        print(str_path)
        algo = str_path[len(str_path)-2]
        print(algo)
        self.algos.append(algo)
        monitor = False
        with open(path, 'rt') as file_handler:
            first_line = file_handler.readline()
            print("FIRST LINE ", first_line)
            if first_line[0] == '#':
                monitor = True
                header = json.loads(first_line[1:])
                # file_handler.seek(2)
                data_frame = pandas.read_csv(file_handler, index_col=None)
            else:
                file_handler.seek(0)
                header = first_line
                data_frame = pandas.read_csv(file_handler, index_col=None)
            # print(header)
            # headers.append(header)
            print(data_frame)
    
        data_frames.append(data_frame)
        data_frame = pandas.concat(data_frames)
        # data_frame.sort_values('t', inplace=True)
        # data_frame.reset_index(inplace=True)
        # data_frame['t'] -= min(header['t_start'] for header in headers)
        # data_frame.headers = headers  # HACK to preserve backwards compatibility
        return data_frame, monitor

    def moving_average(self, values, window):
        """
        Smooth values by doing a moving average

        :param values: (numpy array)
        :param window: (int)
        :return: (numpy array)
        """
        weights = np.repeat(1.0, window) / window
        return np.convolve(values, weights, 'valid')


    def smooth(self, xy, window=50):
        print("smooth")
        x, y = xy
        if y.shape[0] < window:
            return x, y

        original_y = y.copy()
        y = self.moving_average(y, window)

        if len(y) == 0:
            return x, original_y

        # Truncate x
        x = x[len(x) - len(y):]
        return x, y

# Init seaborn
seaborn.set()

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--log-dirs', help='Log folder(s)', nargs='+', required=True, type=str)
parser.add_argument('--title', help='Plot title', default='Learning Curve', type=str)
parser.add_argument('--smooth', action='store_true', default=False,
                    help='Smooth Learning Curve')

parser.add_argument('-p', '--pad', type=check_positive)
args = parser.parse_args()

results = []
algos = []
plt_shade= PlotShaded()

for folder in args.log_dirs:
    timesteps, monitor = plt_shade.read_file(folder)
    results.append((timesteps, monitor))
    if folder.endswith('/'):
        folder = folder[:-1]
    algos.append(folder.split('/')[-2])

pad = int(args.pad)
plt_shade.plot(results, args.title, pad)
