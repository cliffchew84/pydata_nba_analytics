# -*- coding: utf-8 -*-
import dropbox
import pandas as pd


def authenticate_into_dropbox():
    access = pd.read_csv("dropbox.csv")
    return dropbox.Dropbox(access["value"][2])


def load_file_into_my_dropbox(account, var, folder="nba games/"):
    """This function loads files into my own Dropbox Account """
    with open("{}".format(var), 'rb') as f:
        account.files_upload(f.read(), '/{}{}'.format(folder, var), mode=dropbox.files.WriteMode.overwrite)
    return "{} uploaded".format(var)


def download_file_from_dropbox(account, var, folder="nba games/"):
    with open(var, "w") as f:
        metadata, res = account.files_download(path="/{}{}".format(folder, var))
        f.write(res.content)
    return ("{} downloaded".format(var))
