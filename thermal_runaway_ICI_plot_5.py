import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from openpyxl.utils.cell import coordinate_to_tuple
import thermal_runaway_ICI_plot_gui_2


#create thermal runaway plots for any number of samples, plot the average runaway
#curves, and determine the uncertainties in the threshold temperatures.


class Dataset():
    '''
    attributes:
        address (str): folder location
        label (str): curve label
        coords (list): top left and bottom right coordinates (str)
        increment (bool): True if large increments, False if small increments
        bath (list): contains bath temperatures (float)
        sample, sample_2 (list): contains sample temperatures (float) from
                                 last and second-to-last data files in each
                                 temperature increment folder
        lt (list): for actual datasets, contains actual measurements and
                   interpolated values (float). for c_avg, used for counting
                   number of datasets that have actual or interpolated values
                   at each bath temperature.
    '''
    def __init__(self, directory='', label='', coords=[], increment=True):
        self.address = directory
        self.label = label
        self.coords = coords
        self.increment = increment
        self.bath = []
        self.sample = []
        self.sample_2 = []
        self.lt = []


class Sample():
    '''
    attributes:
        name (str): name of sample
        large (list): contains Datasets with large temperature increments
        fine (list): contains Datasets with fine temperature increments
        avg (Dataset): average Dataset obtained through piecewise combination
                       of averages of large and fine increment Datasets
    '''
    def __init__(self, name=''):
        self.name = name
        self.large = []
        self.fine = []
        self.avg = Dataset(label='Average' if name=='' else name+' Average')


def a(list):
    '''
    returns list in NumPy array format (for plotting).
    '''
    return np.array(list)


def r(num):
    '''
    returns number rounded to 3 decimal places in string format.
    '''
    return str(round(num,3))


def create_samples(values):
    '''
    inputs:
        values (dict): contains user-inputted information for Datasets from GUI
    
    returns:
        samples (list): contains all created Samples, which each contain Datasets
    '''
    samples = []
    i = 0
    for key in values:
        if 'samp_' in key:
            i = int(key[5])
            samples.append(Sample(name=values[key]))
        elif 'dir_' in key and values[key] != '':
            n = key[-1]
            dataset = Dataset(directory=values[f'dir_{i}'+n], label=values[f'label_{i}'+n],
                coords=[values[f'tl_{i}'+n], values[f'br_{i}'+n]], increment=values[f'inc_{i}'+n])
            if values[f'inc_{i}'+n]:
                samples[i].large.append(dataset)
            else:
                samples[i].fine.append(dataset)

    for i in range(len(samples)):
        if len(samples[i].large+samples[i].fine) == 0:
            samples[i] = 0
    samples = [i for i in samples if i != 0]

    return samples


def bath_temps(d):
    '''
    inputs:
        d (Dataset)
    
    returns:
        csv_list, csv_list_2 (list): addresses (str) of final and second-to-last
                                     csv images in each bath temperature folder
    
    modifies d.bath to contain all bath temperatures (float). creates csv_list,
    csv_list_2 to store csv files corresponding to each bath temperature.
    '''
    bath_folder_list = os.listdir(d.address)
    csv_list = []
    csv_list_2 = []
    for folder in bath_folder_list:
        d.bath.append(float(folder.split('_')[-1][:-1]))
        csv_file = d.address+'\\'+folder+'\\'+os.listdir(d.address+'\\'+folder)[-2]
        csv_file_2 = d.address+'\\'+folder+'\\'+os.listdir(d.address+'\\'+folder)[-4]
        csv_list.append(csv_file)
        csv_list_2.append(csv_file_2)
    return csv_list, csv_list_2


def sample_temps(csv_list, r1, r2, c1, c2):
    '''
    inputs:
        csv_list (list): contains csv file locations (str)
        r1, r2, c1, c2 (int): row and col numbers of corner cells
    
    returns:
        sample_list (list): contains sample temperatures (float) from csv files
    
    extracts average values (float) of the selected region in the csv files.
    '''
    sample_list = []
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
        sample_list.append(sum(points)/len(points))
    return sample_list


def combine_intplt(b_real, b_intplt, s_real, s_intplt):
    '''
    inputs:
        b_real, s_real (list): actual bath and sample temperatures
        b_intplt, s_intplt (list): intermediate bath and sample temperatures
                                   from interpolation

    returns:
        s_combined (list): contains actual and interpolated sample temperatures
                           in order

    returns actual sample temps combined with interpolated sample temps (even
    allows for non-monotonic data, which is unlikely but possible). ensures that
    sample temps are ordered according to their corresponding bath temp.
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
                          all Datasets up until final fine increment bath temperature
        d (Dataset)
    
    linearly interpolates d for values between actual measurements. note: no
    extrapolation or fitting.
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


def find_fine_start(s):
    '''
    inputs:
        s (Sample)

    returns:
        fine_start (float): lowest fine increment starting bath temp (float).
                            if there are no fine Datasets in s,fine_start = 100.
    '''
    fine_start = 100 #arbitrary, but we never increment the bath to 100 C
    if s.fine != []:
        for d in s.fine:
            if d.bath[0] < fine_start:
                fine_start = d.bath[0]
    return fine_start


def limit_to_fine(s, fine_start):
    '''
    inputs:
        s (Sample)
        fine_start (float): starting fine increment bath temperature
    
    modifies s.avg.bath to contain bath temperatures only until final fine
    increment bath temperature.
    '''
    comb_bath_fine = []
    for d in s.fine:
        comb_bath_fine += d.bath
    
    for temp in s.avg.bath:
        if temp > fine_start and temp not in comb_bath_fine:
            s.avg.bath[s.avg.bath.index(temp)] = 0
    s.avg.bath = [i for i in s.avg.bath if i != 0]


def avg_sample(c_avg, large, fine, fine_start):
    '''
    inputs:
        c_avg (Dataset): c_avg.bath contains all bath temperatures (float) until
                         final fine increment temperature
        large, fine (list): contain large and fine increment Datasets
        fine_start (float): bath temperature at which fine increment measurements begin
    
    modifies c_avg.sample to contain average sample temperatures for all bath
    temperatures. large and fine increment Datasets are averaged separately and
    combined piecewise such that the large Dataset data are only used up until
    the first measurement from a fine Dataset, from which point onwards only the
    fine increment data are used.
    '''
    for i in range(len(c_avg.bath)):
        c_avg.sample.append(0)
    c_avg.lt = c_avg.sample.copy()

    for d in large:
        i = c_avg.bath.index(d.bath[0])
        j = 0
        while j < len(d.lt) and c_avg.bath[i] < fine_start:
            c_avg.sample[i] += d.lt[j]
            c_avg.lt[i] += 1
            i += 1
            j += 1
    
    for d in fine:
        i = c_avg.bath.index(d.bath[0])
        j = 0
        while j < len(d.lt):
            c_avg.sample[i] += d.lt[j]
            c_avg.lt[i] += 1
            i += 1
            j += 1
    
    for i in range(len(c_avg.bath)):
        c_avg.sample[i] = c_avg.sample[i]/c_avg.lt[i]


def thresholds(s):
    '''
    inputs:
        s (Sample)
    
    returns:
        tra_bath_lt (list): contains bath temperatures (float) at which each Dataset
                            reaches the threshold temperature.
    
    finds threshold temperatures (where the slope of the sample temp vs. number
    of measurements curve is first >= 2) for all Datasets. returns their average
    and standard deviation.
    '''
    datasets = s.fine
    if s.fine == []:
        datasets = s.large

    threshold_lt = []
    for d in datasets:
        d.sample_2 = np.subtract(a(d.sample), a(d.sample_2)).tolist()
        i_threshold = next(i for i, slope in enumerate(d.sample_2) if slope >= 2)
        threshold_lt.append(d.bath[i_threshold])
    
    threshold = sum(threshold_lt)/len(threshold_lt)
    stdev = (sum((((temp-threshold)**2) for temp in threshold_lt))/
        len(threshold_lt))**0.5
    return threshold, stdev


def data_plot(samples):
    '''
    inputs:
        samples (list): Samples containing Datasets created from user inputs
    
    generates a new plot for each Sample. each plot contains runaway curves for
    all Datasets within the Sample.
    '''
    for s in samples:
        fig, ax = plt.subplots()

        for d in s.large+s.fine:
            csv_list, csv_list_2 = bath_temps(d)

            r1, c1 = coordinate_to_tuple(d.coords[0].lower())
            r2, c2 = coordinate_to_tuple(d.coords[1].lower())
            d.sample = sample_temps(csv_list, r1, r2, c1, c2)
            d.sample_2 = sample_temps(csv_list_2, r1, r2, c1, c2)
            
            ax.plot(a(d.bath), a(d.sample)-a(d.bath), label=d.label)
    
            h, legend = ax.get_legend_handles_labels()
            if legend != []:
                ax.legend()
            ax.set(xlabel='T$_{cooling}$ (C)', ylabel='T$_{max}$ - T$_{cooling}$ (C)',
                title='Thermal Runaway' if s.name=='' else s.name+' Thermal Runaway')
            plt.savefig(f'runaway_data_plot_{samples.index(s)}.png', format='png')


def avg_plot(samples, gui_values):
    '''
    inputs:
        samples (list): Samples containing Datasets created from user inputs
        gui_values (dict): contains user inputs from GUI
    
    for each Sample, finds average runaway curve, average threshold temperature
    (where the temperature measurement curve's slope is first >= 2), and threshold
    uncertainty (a standard deviation). plots average curves and uncertainties
    for all Samples together in a single plot.
    '''
    fig, ax = plt.subplots()
    
    for s in samples:
        for d in s.large+s.fine:
            s.avg.bath += d.bath
        s.avg.bath = sorted(set(s.avg.bath))
        
        for d in s.large+s.fine:
            intplt(s.avg.bath, d)
        
        fine_start = find_fine_start(s)
        limit_to_fine(s, fine_start)
        avg_sample(s.avg, s.large, s.fine, fine_start)

        threshold, stdev = thresholds(s)

        curve = ax.plot(a(s.avg.bath), a(s.avg.sample)-a(s.avg.bath), label='Average' if
            s.name=='' else s.name+' Average')
        print(f'Threshold temperature: ' + r(threshold) + ' +/- ' + r(stdev) + ' C')

        if gui_values['threshold']:
            plt.axvline(x=threshold, color=curve[0].get_color())
            ax.axvspan(threshold-stdev, threshold+stdev, alpha=0.33, color=curve[0].get_color(),
                label=str(r(threshold))+r' $\pm$ '+str(r(stdev)) if s.name==''
                else s.name+' '+str(r(threshold))+r' $\pm$ '+str(r(stdev)))
    
    ax.legend()
    ax.set(xlabel='T$_{cooling}$ (C)', ylabel='T$_{max}$ - T$_{cooling}$ (C)',
        title='Average Thermal Runaway')
    plt.savefig('runaway_avg_plot.png', format='png')


gui_values = thermal_runaway_ICI_plot_gui_2.values
samples = create_samples(gui_values)
data_plot(samples)
avg_plot(samples, gui_values)
plt.show()