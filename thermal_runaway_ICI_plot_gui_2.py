import PySimpleGUI as sg

rows = 15
samples = 5

instructions = [[sg.Text('Input the information for plotting.')],
                [sg.Text('Check True for large and False for fine bath increment data sets. Each data set must have a data folder. Note: You can use the same data folder multiple times (ex. multiple heaters in one image).')],
                [sg.Text('Curves within each sample tab will be combined and averaged. For different samples, enter the information in a separate tab.')]]

def tab(i):
    name = [sg.Text('Sample name:'), sg.Input(size=20, key='samp_'+str(i))]

    headings = [('Large inc.',8,'inc_'), ('Curve label (ex. HD-7)',20,'label_'),
                ('Top left cell (ex. A1)',20,'tl_'), ('Bottom right cell (ex. D4)',20,'br_'),
                ('Data folder',70,'dir_')]
    header = [(sg.Text(h[0], size=(h[1],1), justification='left', pad=(0,0))) for h in headings]

    input_rows = []
    for row in range(rows):
        increment = [sg.Checkbox('', default=True, size=4, key=headings[0][2]+str(i)+str(row))]
        inputs = [sg.Input(size=(round(1.13*h[1]),1), key=h[2]+str(i)+str(row), pad=(0,0)) for h in headings[1:]]
        browse = [sg.FolderBrowse()]
        input_rows.append(increment+inputs+browse)

    tab_layout = [name, header, *input_rows]
    return tab_layout

tabgroup = [[sg.Tab(f'Sample {i}', tab(i)) for i in range(samples)]]
layout = [  *instructions,
            [sg.TabGroup(tabgroup)],
            [sg.Checkbox('Indicate threshold temperature on plot', default=True, key='threshold')],
            [sg.Button('Submit')]]
col_layout = [[sg.Column(layout, size_subsample_height=4/3, scrollable=True, vertical_scroll_only=True, expand_y=True)]]
window = sg.Window('Inputs', col_layout, resizable=True, finalize=True)

while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    if event == 'Submit':
        window.refresh()
        break

window.close()