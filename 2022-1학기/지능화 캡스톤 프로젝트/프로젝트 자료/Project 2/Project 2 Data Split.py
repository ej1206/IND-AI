import os
import random



def Main():

    OrigPath = "dataset/safetyhelmet/Dataset_All"
    DestPath = "dataset/safetyhelmet/Dataset_Val"

    FileList = os.listdir(OrigPath)

    DataList = []
    ValList  = []
    ValNum   = 1000 # 6 : 2 : 2 = 3000 : 1000 : 1000

    for f in FileList:
        if f.endswith(".png"):
            FileName = f.split('.')[0]
            DataList.append(FileName)

    ValList = random.sample(DataList, ValNum)

    for v in ValList:
        TextName  = '/{}.txt'.format(v)
        ImageName = '/{}.png'.format(v)

        OrigText  = OrigPath + TextName
        DestText  = DestPath + TextName

        OrigImage = OrigPath + ImageName
        DestImage = DestPath + ImageName

        os.rename(OrigText,  DestText)
        os.rename(OrigImage, DestImage)



if __name__ == "__main__":
    Main()