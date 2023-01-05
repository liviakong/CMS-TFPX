import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
from openpyxl.utils.cell import coordinate_to_tuple


#create thermal runaway plots for any number of datasets, plot the average runaway
#curve, plot one standard deviation above and below, and determine the uncertainty
#in the average runaway temperature.


class Curve:
    '''
    attributes:
        bath (list): contains bath temperatures (float)
        sample (list): contains sample temperatures (float)
    '''
    def __init__(self):
        self.bath = []
        self.sample = []


class Dataset(Curve):
    '''
    attributes:
        address (str): folder location
        label (str): curve label
        comb_sample (list): contains actual and extrapolated sample temperatures
                            (float) combined together
        all Curve attributes
    '''
    def __init__(self, directory, label):
        super().__init__()
        self.address = directory
        self.label = label
        self.comb_sample = []


def choose_folders(n):
    '''
    inputs:
        n (int): number of input folders
    
    returns:
        directory_list (list): locations of input folders (str)
        label_list (list): user-inputted labels for folders/curves (str)
    
    asks user to choose n folders and give labels for each curve. directory_list
    contains folder locations and label_list contains curve labels.
    '''
    directory_list = []
    label_list = []
    for i in range(n):
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory()
        directory_list.append(folder)
        label = input('Curve label (ex. HD-7): ')
        label_list.append(label)
        print('\''+label+'\' '+folder)
    return directory_list, label_list


def bath_temps(d):
    '''
    inputs:
        d (Dataset)
    
    returns:
        csv_list (list): addresses of final csv images in each bath temperature
                         folder (str)
    
    modifies d.bath to contain all bath temperatures (float). creates csv_list
    to store csv files corresponding to each bath temperature.
    '''
    bath_folder_list = os.listdir(d.address)
    csv_list = []
    for folder in bath_folder_list:
        d.bath.append(float(folder.split('_')[-1][:-1]))
        csv_file = d.address+'\\'+folder+'\\'+os.listdir(d.address+'\\'+folder)[-2]
        csv_list.append(csv_file)
    return csv_list


def sample_temps(d, csv_list, r1, r2, c1, c2):
    '''
    inputs:
        d (Dataset)
        csv_list (list): contains csv file locations (str)
        r1, r2, c1, c2 (int): row and col numbers of corner cells
    
    modifies d.sample to contain sample temp values (float) determined
    from the average value of the selected region in the csv file.
    '''
    for csv_file in csv_list:
        df = pd.read_csv(csv_file, sep=',', header=None)
        points = []
        i = r1
        while i < r2:
            j = c1
            while j < c2:
                points.append(df.values[i][j])
                j += 1
            i += 1
        d.sample.append(sum(points)/len(points))


def explt(comb_bath, d):
    '''
    inputs:
        comb_bath (list): contains all possible bath temperatures (float) from
                          all datasets
        d (Dataset)
    
    if d.bath contains all possible bath temperatures, d.comb_sample is the same
    as d.sample.
    if d.bath does not contain all the bath temperatures, d.comb_sample has all
    the temperatures from d.sample as well as extrapolated temperatures.
    
    the extrapolating function has the form
        sample = f*e^(g*bath)+h
    where f, g, and h are determined from a least-squares fit. this function
    obeys the roughly exponential relationship between the sample and bath
    temperatures.
    '''
    b_explt = []
    s_explt = []
    if d.bath != comb_bath:
        for temp in comb_bath:
            if temp not in d.bath:
                b_explt.append(temp)

        popt, pcov = curve_fit(lambda t, f, g, h: f*np.exp(g*t)+h, d.bath, d.sample, maxfev=10000)
        f = popt[0]
        g = popt[1]
        h = popt[2]

        for temp in b_explt:
            s_explt.append(f*np.exp(g*temp)+h)
    d.comb_sample = sorted(d.sample+s_explt)


def find_roots(x, y):
    '''
    inputs:
        x (np array): x coordinates
        y (np array): y coordinates
    
    finds roots of a function and returns leftmost root. roots are not
    necessarily in the given list of x coordinates.

    solution from: https://stackoverflow.com/questions/46909373/how-to-find-the-exact-intersection-of-a-curve-as-np-array-with-y-0/46911822#46911822
    '''
    s = np.abs(np.diff(np.sign(y))).astype(bool)
    roots = x[:-1][s] + np.diff(x)[s]/(np.abs(y[1:][s]/y[:-1][s])+1)
    return roots[0]


def p(list):
    '''
    returns list in array format (for plotting).
    '''
    return np.array(list)


def r(num):
    '''
    returns number rounded to 3 decimal places in string format.
    '''
    return str(round(num,3))


def data_plot(directory_list, label_list):
    '''
    inputs:
        directory_list (list): input data folder locations (str)
        label_list (list): user-inputted labels for curves (str)
    
    returns:
        datasets (list): contains a Dataset for each selected folder

    graphs runaway curves for all datasets in a single plot.
    '''
    fig, ax = plt.subplots()
    
    datasets = []
    for i in range(len(directory_list)):
        d = Dataset(directory_list[i], label_list[i])
        datasets.append(d)

        csv_list = bath_temps(d)

        #this is where csv image recognition can come in
        tl = input('Top left cell for \''+d.label+'\' (ex. A1): ').lower()
        br = input('Bottom right cell for \''+d.label+'\' (ex. D4): ').lower()
        r1, c1 = coordinate_to_tuple(tl)
        r2, c2 = coordinate_to_tuple(br)
        sample_temps(d, csv_list, r1, r2, c1, c2)

        ax.plot(p(d.bath), p(d.sample)-p(d.bath), label=d.label)
    
    axes = plt.gca()
    y_min, y_max = axes.get_ylim()
    ax.legend()
    ax.set(xlabel='T$_{cooling}$ (C)', ylabel='T$_{max}$ - T$_{cooling}$ (C)',
        title='Thermal Runaway')
    plt.savefig('runaway_data_plot.png', format='png')

    return datasets, y_min, y_max


def avg_plot(datasets, y_min, y_max):
    '''
    inputs:
        datasets (list): contains a Dataset for each selected folder
    
    graphs average runaway curve, shows 1 standard deviation above and below
    the average, and calculates the error at the point of thermal runaway.
    '''
    fig, ax = plt.subplots()
    c_avg = Curve()

    comb_bath = []
    for d in datasets:
        comb_bath += d.bath
    c_avg.bath = sorted(set(comb_bath))

    for d in datasets:
        explt(c_avg.bath, d)
    
    for i in range(len(c_avg.bath)):
        sample_sum = 0
        for d in datasets:
            sample_sum += d.comb_sample[i]
        c_avg.sample.append(sample_sum/len(datasets))
    
    c_above = Curve()
    c_below = Curve()
    for i in range(len(c_avg.bath)):
        sq_error_sum = 0
        for d in datasets:
            sq_error_sum += (d.comb_sample[i]-c_avg.sample[i])**2
        stdev = np.sqrt(sq_error_sum/len(datasets))
        c_above.sample.append(c_avg.sample[i]+stdev)
        c_below.sample.append(c_avg.sample[i]-stdev)
    
    dydx = Curve()
    for i in range(len(c_avg.bath)-1):
        temp_midpt = (c_avg.bath[i]+c_avg.bath[i+1])/2
        dydx.bath.append(temp_midpt)
    dydx.sample = np.diff(c_avg.sample)/np.diff(c_avg.bath)

    i_tra = next(x for x, slope in enumerate(dydx.sample) if slope >= 2)
    tra_bath = dydx.bath[i_tra]

    tra_sample = np.interp(tra_bath, c_avg.bath, p(c_avg.sample)-p(c_avg.bath))
    l_int = np.interp(tra_sample, p(c_above.sample)-p(c_avg.bath), c_avg.bath)
    r_int = find_roots(p(c_avg.bath), p(c_below.sample)-p(c_avg.bath)-tra_sample)
    l_error = tra_bath-l_int
    r_error = r_int-tra_bath
    print('Runaway temperature (C): ['+r(tra_bath)+'-'+r(l_error)+','+r(tra_bath)
        +'+'+r(r_error)+']')
    
    ax.plot(p(c_avg.bath), p(c_avg.sample)-p(c_avg.bath),
        label='Average')
    plt.fill_between(c_avg.bath, p(c_above.sample)-p(c_avg.bath),
        p(c_below.sample)-p(c_avg.bath), color='lightskyblue')
    plt.text(tra_bath, tra_sample+3, r'${tra}_{{-{left}}}^{{+{right}}}$'
        .format(tra=r(tra_bath),left=r(l_error),right=r(r_error)),
        horizontalalignment='center', verticalalignment='center')
    
    vmin = np.interp(tra_bath, c_avg.bath, p(c_above.sample)-p(c_avg.bath))
    vmax = np.interp(tra_bath, c_avg.bath, p(c_below.sample)-p(c_avg.bath))
    plt.vlines(tra_bath, vmin, vmax)
    plt.hlines(tra_sample, l_int, r_int)
    
    ax.set_ylim([y_min, y_max])
    ax.legend()
    ax.set(xlabel='T$_{cooling}$ (C)', ylabel='T$_{max}$ - T$_{cooling}$ (C)',
        title='Average Thermal Runaway')
    plt.savefig('runaway_avg_plot.png', format='png')


n = int(input('Number of folders to select: '))
directory_list, label_list = choose_folders(n)
datasets, y_min, y_max = data_plot(directory_list, label_list)
avg_plot(datasets, y_min, y_max)
plt.show()
