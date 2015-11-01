
""" Functions to save create consistent names and folders in scripts. 
    This will also save an array of objects into seperate files in one folder.  
    ADD MORE
""" 

import numpy as np
import os

BASE_PATH = os.getcwd()

def make_filname(root, index, ext):
    name =  root + '_' + str(index) + '.' + ext
    return name

def make_folder(typ, fold_name):
    assert os.getcwd() == BASE_PATH #Make sure in right directory
    if not os.path.exists(typ):
        os.makedirs(typ) 
    os.getcwd()
    os.chdir(typ)
    os.mkdir(fold_name)
    os.chdir('..') #Return back to working directory  
    return fold_name

def save_object(obj, foldername, typ, filename):
    assert os.getcwd() == BASE_PATH #Make sure in right directory
    os.chdir(typ) 
    os.chdir(foldername)
    np.save(filename, obj) #FIX adds .npz extension 
    os.chdir('..')
    os.chdir('..') 
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
    foldername = make_folder(typ, folder_root)
    for index, obj in enumerate(arr):
        filename = make_filname(fileroot, index, ext)
        path = save_object(obj, foldername, typ, filename)
        paths.append(path)
        print('Created ' + filename + ' in ' + './' + typ + '/' + foldername)
    return paths 
