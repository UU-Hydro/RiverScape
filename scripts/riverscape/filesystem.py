from tkinter import Tk
from tkinter import filedialog


def select_directory():
    # Don't draw full GUI, the root window from appearing
    window = Tk()
    window.withdraw()
    window.wm_attributes('-topmost', 1)
    dir_name = filedialog.askdirectory()
    window.destroy()
    return dir_name
