import shutil
import os

if not os.path.exists("./data/data/img"):
    os.makedirs("./data/data/img")

if not os.path.exists("./data/data/text"):
    os.makedirs("./data/data/text")

for root,dirs,files in os.walk("./data/data/"):
    if dirs == []:
        continue
    for file in files:
        if file[-3:]=='jpg':
            shutil.move("./data/data/"+file,"./data/data/img/")
        elif file[-3:]=='txt':
            shutil.move("./data/data/"+file,"./data/data/text/")
