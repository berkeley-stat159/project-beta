
""" Functions to save create consistent names and folders in scripts 
    ADD MORE DESCRIPTION 

""" 

import numpy as np
import os

def make_filname(root, index, ext):
    name =  root + '_' + str(index) + '.' + ext
    return name

def make_folder(typ, name):
    os.getcwd()
    os.mkdir(name)
    return name

def save_object(obj, foldername, typ, filename):
    os.getcwd()
    os.chdir(foldername) 
    #np.save(filename) #FIX: NEED TO MAKE FOLDERNAME AND SAVE THIS IN THERE
    path = './' + typ + '/' + foldername + '/' + filename
    return path 

def save_all(arr, fileroot, typ, folder_root, ext):
    """ 
    Saves each element in the array 'arr' based on root and the index of
    the element. 'typ' must be either 'txt', data', 'figures', or 'other'.
        
    Example
    -------
    >>> arr = [plt_obj1, plt_obj2, plt_obj3]
    >>> fileroot = 'RMS'
    >>> ext = 'pdf'
    >>> typ = 'figures'
    >>> folder_root = RMS_plots
    >>> paths = save_all_plt(arr, fileroot, folder_root)
    >>> print(paths)
    ./figures/RMS_plots/RMS_0.pdf
    ./figures/RMS_plots/RMS_1.pdf
    ./figures/RMS_plots/RMS_2.pdf
    """
    assert typ in ['txt', 'data', 'figures', 'other']
    paths = []
    for index, obj in enumerate(arr):
        filename = make_filname(fileroot, index, ext)
        foldername = make_folder(folder_root)
        path = save_object(obj, foldername, typ, filename)
        paths.append(path)
        print('Created ' + filename + ' in ' + './' + typ + '/' + foldername)
    return paths 
