import cv2 as cv
import pytesseract
import numpy as np
import zipfile
from PIL import Image
from PIL import ImageDraw
pytesseract.pytesseract.tesseract_cmd = r'C:\\\Program Files\\\Tesseract-OCR\\\tesseract.exe' #to run pytesseract on local pc

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
img_dict = {} #to store images from zip files
text_dict = {} #to store texts from these images
img_name = {} #to store names of the images


def process_zip_file(name):
    """opens a zip file and stores the images in the dictionary using index

   :param name: the name of the zip file containing images
    """
    index = 0

    with zipfile.ZipFile(name, mode = "r") as zip_ref: #open zip file
        zipped_img_list = zip_ref.infolist()
        for zipped_image in zipped_img_list: #iterate through the list to add images to dictionary
            img_file = zip_ref.open(zipped_image)
            image = Image.open(img_file)
            img_dict[index] = image.convert("RGB")
            img_name[index] = zipped_image.filename #add file name to dict according to index
            index +=1


def add_word_list(index):
    """Process the image and run Pytesseract OCR on the image and add the list of words to the text_dict using index provided
    :param index: index of a PIL.Image object in img_dict
    """
    img = img_dict[index]
    opencvImage = cv.cvtColor(np.array(img) , cv.COLOR_RGB2BGR) #convert to BGR format for opencv image
    cv_img_bin=cv.threshold(opencvImage,192,255,cv.THRESH_BINARY)[1] #binarize the image
    text = pytesseract.image_to_string(Image.fromarray(cv_img_bin))
    word_list = text.split()
    text_dict[index] = word_list


def search_word(index, word):
    """ checks whether the image at the index position contains word
    :param index: index of the image
    :param word: word to be searched
    :returns : True if the image contains the word amd false otherwise
    """
    if word in text_dict[index]:
        return True
    else:
        return False


def search_faces(index):
    """finds the faces in the given image and returns the bounding box coordinates
    :param index:index of a PIL image object in img_dict to search faces in
    :returns: a list of bounding boxes coordinates
    """
    image = img_dict[index]
    bounding_boxes = []
    opencvImage = cv.cvtColor(np.array(image) , cv.COLOR_RGB2BGR) #convert to BGR format for opencv image
    gray = cv.cvtColor(opencvImage, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.15,minNeighbors = 5)
    for x,y,w,h in faces:
        bounding_boxes.append((x,y, x+w, y+h))
    return bounding_boxes


def create_face_list(index , bounding_boxes):
    """creates a list of faces from image
    :param index: index of a PIL image object in img_dict
    :param bounding_boxes: list of coordinates of bounding boxes
    :returns: a list containing faces"""
    image = img_dict[index]
    face_list = []
    for box in bounding_boxes:
        face_list.append(image.crop(box))
    return face_list


def create_contact_sheet(face_list):
    """creates a contact sheet from a list of face images
    :param face_list: a list of images of faces
    :return: a contact sheet of images"""
    maxsize = (100,100)
    contact_sheet = Image.new("RGB",(500,200)) #5 columns and 2 rows
    x = 0
    y = 0
    for face in face_list:
        face.thumbnail(maxsize)
        contact_sheet.paste(face,(x,y))
        if x > 500: #if greater than 5 coloumn, start over at next row
            x = 0
            y += 100
        else:
            x += 100
    return contact_sheet


def main():
    file_name = input("Enter the name of the zip file containing images : ")
    word = input("Enter the word to search from the zipped image file : ")
    try:
        process_zip_file(file_name)
        for index in img_dict: #iterate through img_dict
            add_word_list(index)
            if search_word(index, word) == True: # check if the word is in the image
                print("Results found in file {0}".format(img_name[index]))
                bounding_boxes = search_faces(index)
                if bounding_boxes == []: #no faces found in image
                    print("But there were no faces in that file ")
                else: #create contact sheet to display
                    face_list = create_face_list(index, bounding_boxes)
                    contact_sheet = create_contact_sheet(face_list)
                    contact_sheet.save("result for file {0}".format(img_name[index])) #save the contact sheet as another file
        print("Finished searching!")                   
    except:
        print("Error at index {0}".format(index))


main()
