import PySimpleGUI as sg

headings = [('Curve label (ex. HD-7)',20,'label_'), ('Top left cell (ex. A1)',20,'tl_'), ('Bottom right cell (ex. D4)',20,'br_'), ('Data folder',70,'dir_')]
header = [[(sg.Text(h[0], size=(h[1],1), justification='left', pad=(0,0))) for h in headings]]

input_rows = []
for row in range(10):
    inputs = [sg.Input(size=(round(1.15*h[1]),1), key=h[2]+str(row), pad=(0,0)) for h in headings]
    browse = [sg.FolderBrowse()]
    input_rows.append(inputs+browse)

layout = [header, input_rows, [sg.Button('Submit')]]

window = sg.Window('Inputs', layout)

while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    if event == 'Submit':
        window.refresh()
        break
window.close()