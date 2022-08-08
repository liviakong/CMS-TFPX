import tkinter as tk
from tkinter import filedialog
import pandas as pd
from numpy import diff
from scipy.signal import find_peaks
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

#find y values where there are flatter regions (plateaus) in a graph. input is a csv
#file containing data with column headers in the first row (ex. RTD measurements over time)

#returns list with column name and values
def col_from_file(file, col_num):
    df = pd.read_csv(file, sep=',', header=None)
    col_name = df.values[0][col_num]
    col_values = []
    for row in df.values[1:]:
        col_values.append(float(row[col_num]))
    return col_name, col_values


#smoothens values out. higher smoothness means there is a stronger smoothing effect
def smooth(x, y, smoothness):
    smooth_list = sm.nonparametric.lowess(y, x[:len(y)], frac=smoothness)[:,1]
    return smooth_list


#returns list of negated absolute values of derivatives between consecutive points.
#dy values are "smoothed" out to account for noise/small fluctuations
#notes: negated since find_peaks() finds local maxima, absolute value taken so
#that plateaus can be found regardless of whether data values are increasing or
#decreasing
def derivative(x, y, smoothness):
    dydx = -abs(diff(y)/diff(x))
    dydx = smooth(x, dydx, smoothness)
    return dydx


#returns list of corresponding (the left point) coordinates for all local maxima in
#list of negated derivatives (defined here as the slope between two adjacent points)
def y_peak(x, y, dydx):
    peaks = find_peaks(dydx,plateau_size=1)[1]['right_edges']

    x_plat = []
    y_plat = []
    dydx_plat = []
    for index in peaks:
        if (round(y[index], 1)) not in y_plat: #avoid duplicates
            x_plat.append(x[index])
            y_plat.append((round(y[index], 1)))
            dydx_plat.append(dydx[index])

    return x_plat, y_plat, dydx_plat


#makes plots of y vs x with identified plateau points marked
def make_plot(file, plot_num, x_label, y_label, x_list, y_list, x_plat, y_plat):
    plt.subplot(1, 2, plot_num)
    line, = plt.plot(x_list[:len(y_list)], y_list)
    highlights, = plt.plot(x_plat, y_plat, 'ro')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    return line, highlights


#adds sliders, counter, and reset button to plot
def add_features():
    #creates sliders for y and dydx to adjust signal smoothness
    ax_y = plt.axes([0.25, 0.09, 0.65, 0.03])
    y_slider = Slider(ax_y, 'y smoothness', 0, 0.1, 0.015)
    ax_dydx = plt.axes([0.25, 0.05, 0.65, 0.03])
    dydx_slider = Slider(ax_dydx, 'dydx smoothness', 0, 0.2, 0.015)

    #displays number of points identified
    num_pts = plt.text(0, -1.25, 'points identified: ' + str(len(y_plat)))

    #creates reset button to reset values back to default
    ax_reset = plt.axes([0.8, 0.01, 0.1, 0.03])
    button = Button(ax_reset, 'reset', hovercolor='0.975')

    return y_slider, dydx_slider, num_pts, button


#dynamically updates plot based on slider values
def update(val):
    new_y_list = smooth(x_list, y_list, y_slider.val)
    new_dydx_list = derivative(x_list, new_y_list, dydx_slider.val)
    global y_plat
    x_plat, y_plat, dydx_plat = y_peak(x_list, new_y_list, new_dydx_list)

    y_line.set_ydata(new_y_list)
    y_highlight.set_data(x_plat, y_plat)
    dydx_line.set_ydata(new_dydx_list)
    dydx_highlight.set_data(x_plat, dydx_plat)

    num_pts.set_text('points identified: ' + str(len(y_plat)))


#resets values to default
def reset(event):
    y_slider.reset()
    dydx_slider.reset()


#prints y values to command line rounded to one decimal place after figure is closed
def handle_close(event):
    print('plateau values:')
    for value in y_plat:
        print(round(value, 1))





#prompts user to choose a csv file
root = tk.Tk()
root.withdraw()
file = filedialog.askopenfilename()


#determines which columns are used for data
print("enter x column letter (ex. a, b, etc.)")
x_col_num = ord(input()) - 97
print("enter y column letter (ex. a, b, etc.)")
y_col_num = ord(input()) - 97


#creates list of x and y values from csv file
x_label, x_list = col_from_file(file, x_col_num)
y_label, y_list = col_from_file(file, y_col_num)
#smoothens y values out to reduce effect of noise/small fluctuations
y_list = smooth(x_list, y_list, 0.015)


#finds negated absolute values of derivatives
dydx_list = derivative(x_list, y_list, 0.015)


#finds coordinates of plateaus in data
x_plat, y_plat, dydx_plat = y_peak(x_list, y_list, dydx_list)


#makes plots of y vs x and dydx vs x with identified plateaus marked
fig = plt.figure()
y_line, y_highlight = make_plot(file, 1, x_label, y_label, x_list, y_list, x_plat, y_plat)
dydx_line, dydx_highlight = make_plot(file, 2, x_label, y_label + '/' + x_label, x_list, dydx_list, x_plat, dydx_plat)
plt.tight_layout(rect=[0, 0.1, 1, 1])


#adds sliders, identified points counter, and reset button and dynamically updates plot
y_slider, dydx_slider, num_pts, button = add_features()
y_slider.on_changed(update)
dydx_slider.on_changed(update)
button.on_clicked(reset)


#prints identified y values to command line rounded to one decimal place after figure is closed
fig.canvas.mpl_connect('close_event', handle_close)


plt.show()
