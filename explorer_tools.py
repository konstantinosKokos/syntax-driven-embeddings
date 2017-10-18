import os
import re

def get_texts(startdir='/home/jovyan/kokos-playground/Data', ignore_list=[]):
    file_list = []
    ignore_list.append('readme.txt')
    ignore_list.append('license.txt')
    for dirpath, _, files in os.walk(startdir):
        for file in files:
            if file in ignore_list: continue
            if file[-3:]=='txt':
                file_list.append(os.path.join(dirpath, file))
    return file_list

def save_paths(file_list, save_dir='/home/jovyan/kokos-playground/Output/file_paths.txt'):
    with open('/home/jovyan/kokos-playground/Output/file_paths.txt','w') as f:
        for file in file_list:
            f.write(file+'\n')