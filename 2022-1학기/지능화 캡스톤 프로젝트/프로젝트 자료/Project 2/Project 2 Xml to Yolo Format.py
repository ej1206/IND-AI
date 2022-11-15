import os
import cv2
import xml.etree.ElementTree as ET


def Main():
    ImagePath = "dataset/safetyhelmet/images/"
    GTPath    = "dataset/safetyhelmet/annotations/"      # XML
    Save_ImagePath = "dataset/safetyhelmet/SavedImage/"
    Save_GTPath    = "dataset/safetyhelmet/SavedGT/"     # TEXT

    FileList = os.listdir(ImagePath)

    for i in FileList:
        ImageName = ImagePath + i
        GTName = GTPath + i.split('.')[0] + '.xml'
        Save_ImageName = Save_ImagePath + i
        Save_GTName = Save_GTPath + i.split('.')[0] + '.txt'

        Image = cv2.imread(ImageName)
        Width = Image.shape[1]
        Height = Image.shape[0]

        Number = 0
        Contents = []

        XML = ET.parse(GTName)
        root = XML.getroot()

        for obj in root.iter("object"):

            Name      = obj.findtext("name")
            Truncated = obj.findtext("truncated")
            Occlude   = obj.findtext("occlude")
            Difficult = obj.findtext("difficult")

            xmin = int(obj.find("bndbox").findtext("xmin"))
            xmax = int(obj.find("bndbox").findtext("xmax"))
            ymin = int(obj.find("bndbox").findtext("ymin"))
            ymax = int(obj.find("bndbox").findtext("ymax"))



            CenterX = round(((xmin + xmax) / 2) / Width, 6)
            CenterY = round(((ymin + ymax) / 2) / Height, 6)
            Width_  = round((xmax - xmin) / Width, 6)
            Height_ = round((ymax - ymin) / Height, 6)

            if Name == "head":
                Class = 0
                Color = (255, 0, 0)

            elif Name == "helmet":
                Class = 1
                Color = (0, 255, 0)
            else:
                print("Check Annotation File : ", ImageName, ", ", Name)
                Class = 2
                Color = (0, 0, 255)

            Contents.append(Class)
            Contents.append(CenterX)
            Contents.append(CenterY)
            Contents.append(Width_)
            Contents.append(Height_)

            cv2.rectangle(Image, (xmin, ymin), (xmax, ymax), Color, 2)

            Number += 5

        cv2.imwrite(Save_ImageName, Image)
        out = open(Save_GTName, 'w')
        for j in range(Number):
            if j % 5 == 0:
                print(Contents[j], Contents[j+1], Contents[j+2], Contents[j+3], Contents[j+4], file=out)


if __name__ == "__main__":
    Main()