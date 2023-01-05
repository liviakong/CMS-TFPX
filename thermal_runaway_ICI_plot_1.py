import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import os
from datetime import datetime
import time
import csv
import sys
from scipy.optimize import curve_fit
import math
from openpyxl.utils.cell import coordinate_to_tuple


#create thermal runaway plots for two datasets and determine error in runaway temperature
#between them. replace INPUT DIRECTORY with path to data folders that contain
#time_temperature folders (ex. C:\\Users\\...\\Gantry_moresco70_diamondglue_HD7_ICI_24_v2).


#row and column numbers from cell coordinate
def cell_coords(top_left, bottom_right):
    r1, c1 = coordinate_to_tuple(top_left)
    r2, c2 = coordinate_to_tuple(bottom_right)
    return r1, r2, c1, c2


#calculate average value for a region in a csv file
def temp_avg(input_file, row1, row2, col1, col2):
    df=pd.read_csv(input_file, sep=',',header=None)
    points = []
    i = row1
    while i < row2:
        j = col1
        while j < col2:
            points.append(df.values[i][j])
            j +=1
        i += 1
    avg_temp = sum(points)/len(points)
    return avg_temp


#create list of bath temp values and sample temp values
def get_temp_values(file_list, r1, r2, c1, c2):
    bath = []
    sample = []
    for i in file_list:
        bath_tem = i[0]
        csv_file = i[1]
        bath.append(bath_tem)
        sample.append(temp_avg(csv_file, r1, r2, c1, c2))
    return bath, sample


#create a list for input
def bath_temp(dir_input):
    folder_list = os.listdir(dir_input)
    bath_temp_list = []
    for folder in folder_list:
        bath_tem = float(folder.split('_')[-1][:-1])
        file_name = dir_input+'\\'+folder+'\\'+os.listdir(dir_input+'\\'+folder)[-2]
        temp_file = [bath_tem, file_name]
        bath_temp_list.append(temp_file)
    return bath_temp_list


#create complete list of bath temps
def all_bath_temps(b1, b2):
    b_all = sorted(set(b1+b2))
    return b_all


#extrapolate sample temps across complete bath temp range (if necessary)
def explt(b_all, bn, sn):
    b_explt = []
    if bn != b_all:
        for temp in b_all:
            if temp not in bn:
                b_explt.append(temp)
        popt, pcov = curve_fit(lambda t, f, g, h: f*np.exp(g*t)+h, bn, sn)
        f = popt[0]
        g = popt[1]
        h = popt[2]
    s_explt = []
    if b_explt != []:
        for temp in b_explt:
            s_explt.append(f*np.exp(g*temp)+g)
    return b_explt, s_explt


#combine actual data with extrapolated data in order (so that bath temps correspond with sample temps)
def combine_explt(b_real, b_explt, s_real, s_explt):
    s_combined = []
    i = 0
    j = 0
    while i < len(b_real):
        if j < len(b_explt):
            if b_real[i] < b_explt[j]:
                s_combined.append(s_real[i])
                i += 1
            else:
                s_combined.append(s_explt[j])
                j += 1
        else:
            s_combined.append(s_real[i])
            i += 1

    while j < len(b_explt):
        s_combined.append(s_explt[j])
        j += 1
    return s_combined


#define average curve
def avg_curve(s_combined1, s_combined2):
    s_avg = []
    for i in range(len(s_combined1)):
        s_avg.append((s_combined1[i]+s_combined2[i])/2)
    return s_avg


#define error boundary curves
def error_curve(s_avg, s1, s2):
    error_above = []
    error_below = []
    for i in range(len(s_avg)):
        sq_error1 = (s1[i]-s_avg[i])**2
        sq_error2 = (s2[i]-s_avg[i])**2
        stdev = math.sqrt((sq_error1+sq_error2)/2)
        error_above.append(s_avg[i]+stdev)
        error_below.append(s_avg[i]-stdev)
    return error_above, error_below


#return Tmax - Tcooling for plotting
def diff_t(Tmax, Tcooling):
    return sorted(np.array(Tmax)-np.array(Tcooling))


#save image after closing window
def handle_close(event):
    pic_name = 'runaway.png'
    plt.savefig(pic_name,format='png')




file_list_1 = bath_temp('INPUT DIRECTORY HERE*********************************')
file_list_2 = bath_temp('INPUT DIRECTORY HERE*********************************')


top_left_1 = input('Top left cell for first data set (ex. A1): ').lower()
bottom_right_1 = input('Bottom right cell for first data set (ex. D4): ').lower()
r11, r12, c11, c12 = cell_coords(top_left_1, bottom_right_1)
top_left_2 = input('Top left cell for second data set (ex. A1): ').lower()
bottom_right_2 = input('Bottom right cell for second data set (ex. D4): ').lower()
r21, r22, c21, c22 = cell_coords(top_left_2, bottom_right_2)


b1, s1 = get_temp_values(file_list_1, r11, r12, c11, c12)
b2, s2 = get_temp_values(file_list_2, r21, r22, c21, c22)


#for Tmax - T cooling
diff_t1 = diff_t(s1,b1)
bath1 = sorted(np.array(b1))

diff_t2 = diff_t(s2,b2)
bath2 = sorted(np.array(b2))

b_all = all_bath_temps(b1,b2)
b_explt1, s_explt1 = explt(b_all,b1,s1)
b_explt2, s_explt2 = explt(b_all,b2,s2)

s_combined1 = combine_explt(b1,b_explt1,s1,s_explt1)
diff_t_explt1 = diff_t(s_combined1,b_all)
s_combined2 = combine_explt(b2,b_explt2,s2,s_explt2)
diff_t_explt2 = diff_t(s_combined2,b_all)

s_avg = avg_curve(s_combined1,s_combined2)
diff_t_avg = diff_t(s_avg,b_all)
error_above, error_below = error_curve(s_avg, s_combined1, s_combined2)
diff_error_above = diff_t(error_above,b_all)
diff_error_below = diff_t(error_below,b_all)


#define plot
fig, ax = plt.subplots()

ax.plot(b_all, diff_t_explt1, color='blue', linestyle='dashed', label='_nolegend_')
ax.plot(bath1, diff_t1, color='blue', label='24')
ax.plot(b_all, diff_t_explt2, color='red', linestyle='dashed', label='_nolegend_')
ax.plot(bath2, diff_t2, color='red', label='24_v2')
ax.plot(b_all, diff_t_avg, color='dodgerblue', linestyle='dashed', label='average')

plt.fill_between(b_all, diff_error_above, diff_error_below, color='lightcyan')

ax.legend()
ax.set(xlabel='$T_{cooling}$ (C)', ylabel='$T_{max}$ - $T_{cooling}$ (C)',
       title='Plaquette')

ax = plt.gca()
ax.set_ylim([None,100])

fig.canvas.mpl_connect('close_event', handle_close)

plt.show(block=False)


#calculate error in runaway temperature based on input T_cooling temperature
tra_cool_avg = input('Input runaway T_cooling temperature (C): ')
tra_diff_avg = np.interp(tra_cool_avg, b_all, diff_t_avg)
error_left = np.interp(tra_diff_avg, diff_error_above, b_all)
error_right = np.interp(tra_diff_avg, diff_error_below, b_all)
print('The runaway temperature is between '+str(round(error_left,3))+' C and '+
        str(round(error_right,3))+' C, giving a temperature range of '+
        str(round(error_right-error_left,3))+' C.')


plt.show()
