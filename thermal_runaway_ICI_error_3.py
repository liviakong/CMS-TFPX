import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from openpyxl.utils.cell import coordinate_to_tuple


#create thermal runaway plots for any number of datasets, plot the average runaway
#curve, and determine the uncertainty in the average runaway temperature.


class Dataset():
    '''
    attributes:
        address (str): folder location
        label (str): curve label
        bath (list): contains bath temperatures (float)
        sample (list): contains sample temperatures (float)
        lt (list): for actual datasets, contains actual measurements and
                   interpolated values (float). for c_avg, used for counting
                   number of datasets that have actual or interpolated values
                   at each bath temperature.
    '''
    def __init__(self, directory='', label=''):
        self.address = directory
        self.label = label
        self.bath = []
        self.sample = []
        self.lt = []


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


def combine_intplt(b_real, b_intplt, s_real, s_intplt):
    '''
    returns actual data combined with interpolated data in order so that bath
    temps correspond with sample temps. even allows for non-monotonic data
    (unlikely but occasionally possible).
    '''
    s_combined = []
    i = 0
    j = 0
    while i < len(b_real):
        if j < len(b_intplt):
            if b_real[i] < b_intplt[j]:
                s_combined.append(s_real[i])
                i += 1
            else:
                s_combined.append(s_intplt[j])
                j += 1
        else:
            s_combined.append(s_real[i])
            i += 1
    while j < len(b_intplt):
        s_combined.append(s_intplt[j])
        j += 1
    return s_combined


def intplt(comb_bath, d):
    '''
    inputs:
        comb_bath (list): contains all possible bath temperatures (float) from
                          all datasets
        d (Dataset)
    
    if d.bath contains all possible bath temperatures, d.lt is the same as
    d.sample.
    if d.bath does not contain all the bath temperatures, d.lt has all the
    temperatures from d.sample as well as linearly interpolated temperatures
    between actual measurements.
    '''
    b_intplt = []
    s_intplt = []
    if d.bath != comb_bath:
        i = comb_bath.index(d.bath[0])
        i_end = comb_bath.index(d.bath[-1])
        while i <= i_end:
            if comb_bath[i] not in d.bath:
                b_intplt.append(comb_bath[i])
            i += 1
        s_intplt = np.interp(b_intplt, d.bath, d.sample).tolist()
    d.lt = combine_intplt(d.bath, b_intplt, d.sample, s_intplt)


def avg_sample(c_avg, datasets):
    '''
    inputs:
        c_avg (Dataset): c_avg.bath contains all bath temperatures (float)
        datasets (list): contains actual Datasets
    
    modifies c_avg.sample to contain average sample temperatures for all bath
    temperatures.
    '''
    for i in range(len(c_avg.bath)):
        c_avg.sample.append(0)
    c_avg.lt = c_avg.sample.copy()

    for d in datasets:
        i = c_avg.bath.index(d.bath[0])
        j = 0
        while j < len(d.lt):
            c_avg.sample[i] += d.lt[j]
            c_avg.lt[i] += 1
            i += 1
            j += 1
    
    for i in range(len(c_avg.bath)):
        c_avg.sample[i] = c_avg.sample[i]/c_avg.lt[i]


def get_tra_baths(datasets):
    '''
    inputs:
        datasets (list): contains Datasets
    
    returns:
        tra_bath_lt (list): contains list of bath temperatures (float) at which
                            each Dataset reaches runaway. point of runaway is
                            where the slope of the runaway curve is first >= 2.
    '''
    tra_bath_lt = []
    for d in datasets:
        dydx = Dataset()
        for i in range(len(d.bath)-1):
            temp_midpt = (d.bath[i]+d.bath[i+1])/2
            dydx.bath.append(temp_midpt)
        dydx.sample = np.diff(d.sample)/np.diff(d.bath)

        i_tra = next(x for x, slope in enumerate(dydx.sample) if slope >= 2)
        tra_bath_lt.append(dydx.bath[i_tra])
    return tra_bath_lt


def a(list):
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
        y_min, y_max (float): y-axis limits in graphical output

    graphs runaway curves for all datasets in a single plot.
    '''
    fig, ax = plt.subplots()
    
    datasets = []
    for i in range(len(directory_list)):
        #create a Dataset object for each set of data
        d = Dataset(directory_list[i], label_list[i])
        datasets.append(d)

        #d.bath contains all bath temperatures, and csv_list has locations for
        #all relevant sample images (last csv file in each bath increment folder)
        csv_list = bath_temps(d)

        ############this is where csv image recognition can come in############
        #converts user inputs for top left and bottom right corner cells into
        #coordinates. d.sample contains the average sample temperatures
        #calculated from the csv files
        tl = input('Top left cell for \''+d.label+'\' (ex. A1): ').lower()
        br = input('Bottom right cell for \''+d.label+'\' (ex. D4): ').lower()
        r1, c1 = coordinate_to_tuple(tl)
        r2, c2 = coordinate_to_tuple(br)
        sample_temps(d, csv_list, r1, r2, c1, c2)
        
        #plots runaway curves for each Dataset
        ax.plot(a(d.bath), a(d.sample)-a(d.bath), label=d.label)
    
    #edit axes and labels
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
    
    finds average runaway curve, average runaway temperature (runaway
    temperature for each Dataset is where the runaway curve's slope is first
    >= 2), and uncertainty (a standard deviation). in a second plot, shows
    average curve, average point of runaway, and uncertainty.
    '''
    fig, ax = plt.subplots()
    c_avg = Dataset()

    #c_avg.bath contains all possible bath temperatures from all Datasets
    comb_bath = []
    for d in datasets:
        comb_bath += d.bath
    c_avg.bath = sorted(set(comb_bath))
    
    #c_avg.sample contains the average sample temperatures at each bath
    #temperature (so it is the average of all the runaway curves)
    for d in datasets:
        intplt(c_avg.bath, d)
    avg_sample(c_avg, datasets)
    
    #finds runaway temperature for each Dataset, their average, and standard
    #deviation
    tra_bath_lt = get_tra_baths(datasets)
    tra_bath = sum(tra_bath_lt)/len(tra_bath_lt)
    stdev = (sum((((temp-tra_bath)**2) for temp in tra_bath_lt))/
        len(tra_bath_lt))**0.5

    #prints runaway temperature to console. plots average curve, runaway
    #temperature, and uncertainty
    print('Runaway temperature: '+r(tra_bath)+' +/- '+r(stdev)+' C')
    ax.plot(a(c_avg.bath), a(c_avg.sample)-a(c_avg.bath), label='Average')
    plt.vlines(tra_bath, y_min, y_max)
    ax.axvspan(tra_bath-stdev, tra_bath+stdev, alpha=0.33, color='red',
        label='68% CI')
    plt.text(tra_bath-0.15, y_max-3, r'${tra} \pm {stdev}$ C'
        .format(tra=r(tra_bath), stdev=r(stdev)),
        horizontalalignment='right', verticalalignment='center')
    
    #edit axes and labels
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