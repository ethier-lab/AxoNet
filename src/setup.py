# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 13:58:28 2019

@author: mritch3
"""

#code taken from https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
import requests, os

os.chdir('..')
home=os.getcwd()




#%%functions
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    file_id = 'TAKE ID FROM SHAREABLE LINK'
    destination = 'DESTINATION FILE ON YOUR DISK'
    download_file_from_google_drive(file_id, destination)
    
    
    
#%% script
dest=home + '\data\data_updated_3-22-2020.mat'

if not (os.path.exists(dest)):
    print('Downloading data_updated_3-22-2020.mat...')
    fileId = '19Lu6DmfuasfeEDde1IkdtxrHvloWdBF8'
    download_file_from_google_drive(fileId, dest)
    