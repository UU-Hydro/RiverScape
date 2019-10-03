import tkinter
from tkinter import filedialog


def select_directory():
    # Don't draw full GUI, the root window from appearing
    tkinter.Tk().withdraw()
    dir_name = filedialog.askdirectory()

    return dir_name
