import os
from shutil import copy


src = os.getcwd()
for filename in os.listdir("."):
    if filename.endswith("1.png"):
        copy(src + "/"  + filename, src + '/testing-data/1/{}'.format(filename))
        print("0")
    
    elif filename.endswith("0.png"):
        copy(src + "/"  + filename,  src + '/testing-data/0/{}'.format(filename))
        print("1")

print("Completed")
