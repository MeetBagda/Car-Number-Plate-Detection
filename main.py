import cv2
import numpy as np
import os
import math
import csv
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Label, Frame, Button, Text, W

# Global variables
plate = ""
image = ""
plate_text = ""

# Color scalars
SCALAR_BLACK = (0, 0, 0)
SCALAR_WHITE = (255, 255, 255)
SCALAR_YELLOW = (0, 255, 255)
SCALAR_GREEN = (0, 255, 0)
SCALAR_TEAL = (72, 100, 50)
SCALAR_RED = (255, 0, 0)

# Classes
class PossibleChar:
    def __init__(self, _contour):
        self.contour = _contour
        self.boundingRect = cv2.boundingRect(self.contour)
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intBoundingRectX = intX
        self.intBoundingRectY = intY
        self.intBoundingRectWidth = intWidth
        self.intBoundingRectHeight = intHeight
        self.intBoundingRectArea = self.intBoundingRectWidth * self.intBoundingRectHeight
        self.intCenterX = (self.intBoundingRectX + self.intBoundingRectX + self.intBoundingRectWidth) / 2
        self.intCenterY = (self.intBoundingRectY + self.intBoundingRectY + self.intBoundingRectHeight) / 2
        self.fltDiagonalSize = math.sqrt((self.intBoundingRectWidth ** 2) + (self.intBoundingRectHeight ** 2))
        self.fltAspectRatio = float(self.intBoundingRectWidth) / float(self.intBoundingRectHeight)

class PossiblePlate:
    def __init__(self):
        self.imgPlate = None
        self.imgGrayscale = None
        self.imgThresh = None
        self.rrLocationOfPlateInScene = None
        self.strChars = ""

# Pre-processing constants
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

def preprocess(imgOriginal):
    if imgOriginal is None:
        raise ValueError("Input image is None")
    imgGrayscale = extractValue(imgOriginal)
    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)
    height, width = imgGrayscale.shape
    imgBlurred = np.zeros((height, width, 1), np.uint8)
    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
    return imgGrayscale, imgThresh

def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape
    imgHSV = np.zeros((height, width, 3), np.uint8)
    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)
    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)
    return imgValue

def maximizeContrast(imgGrayscale):
    height, width = imgGrayscale.shape
    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)
    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
    return imgGrayscalePlusTopHatMinusBlackHat

# Detect Char constants
kNearest = cv2.ml.KNearest_create()

MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8
MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0
MIN_PIXEL_AREA = 80
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0
MAX_CHANGE_IN_AREA = 0.5
MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2
MAX_ANGLE_BETWEEN_CHARS = 12.0
MIN_NUMBER_OF_MATCHING_CHARS = 3
RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30
MIN_CONTOUR_AREA = 100

def loadKNNDataAndTrainKNN():
    allContoursWithData = []
    validContoursWithData = []
    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)
    except:
        print("error, unable to open classifications.txt, exiting program\n")
        input("Press Enter to continue...")  # Cross-platform pause
        return False
    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
    except:
        print("error, unable to open flattened_images.txt, exiting program\n")
        input("Press Enter to continue...")  # Cross-platform pause
        return False
    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
    kNearest.setDefaultK(3)  # Increased k for better generalization
    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
    return True

def detectCharsInPlates(listOfPossiblePlates):
    if len(listOfPossiblePlates) == 0:
        return listOfPossiblePlates
    for possiblePlate in listOfPossiblePlates:
        if possiblePlate.imgPlate is None:
            print("Warning: imgPlate is None, skipping plate")
            continue
        possiblePlate.imgGrayscale, possiblePlate.imgThresh = preprocess(possiblePlate.imgPlate)
        possiblePlate.imgThresh = cv2.resize(possiblePlate.imgThresh, (0, 0), fx=1.6, fy=1.6)
        thresholdValue, possiblePlate.imgThresh = cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh)
        listOfListsOfMatchingCharsInPlate = findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)
        if len(listOfListsOfMatchingCharsInPlate) == 0:
            possiblePlate.strChars = ""
            continue
        for i in range(len(listOfListsOfMatchingCharsInPlate)):
            listOfListsOfMatchingCharsInPlate[i].sort(key=lambda matchingChar: matchingChar.intCenterX)
            listOfListsOfMatchingCharsInPlate[i] = removeInnerOverlappingChars(listOfListsOfMatchingCharsInPlate[i])
        intLenOfLongestListOfChars = 0
        intIndexOfLongestListOfChars = 0
        for i in range(len(listOfListsOfMatchingCharsInPlate)):
            if len(listOfListsOfMatchingCharsInPlate[i]) > intLenOfLongestListOfChars:
                intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
                intIndexOfLongestListOfChars = i
        longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[intIndexOfLongestListOfChars]
        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imgThresh, longestListOfMatchingCharsInPlate)
    return listOfPossiblePlates

def findPossibleCharsInPlate(imgGrayscale, imgThresh):
    listOfPossibleChars = []
    contours = []
    imgThreshCopy = imgThresh.copy()
    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        possibleChar = PossibleChar(contour)
        if checkIfPossibleChar(possibleChar):
            listOfPossibleChars.append(possibleChar)
    return listOfPossibleChars

def checkIfPossibleChar(possibleChar):
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
        possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
        MIN_ASPECT_RATIO < possibleChar.fltAspectRatio and possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    return False

def findListOfListsOfMatchingChars(listOfPossibleChars):
    listOfListsOfMatchingChars = []
    for possibleChar in listOfPossibleChars:
        listOfMatchingChars = findListOfMatchingChars(possibleChar, listOfPossibleChars)
        listOfMatchingChars.append(possibleChar)
        if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:
            continue
        listOfListsOfMatchingChars.append(listOfMatchingChars)
        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))
        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved)
        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)
        break
    return listOfListsOfMatchingChars

def findListOfMatchingChars(possibleChar, listOfChars):
    listOfMatchingChars = []
    for possibleMatchingChar in listOfChars:
        if possibleMatchingChar == possibleChar:
            continue
        fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)
        fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)
        fltChangeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(possibleChar.intBoundingRectArea)
        fltChangeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(possibleChar.intBoundingRectWidth)
        fltChangeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(possibleChar.intBoundingRectHeight)
        if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
            fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
            fltChangeInArea < MAX_CHANGE_IN_AREA and
            fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
            fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):
            listOfMatchingChars.append(possibleMatchingChar)
    return listOfMatchingChars

def distanceBetweenChars(firstChar, secondChar):
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)
    return math.sqrt((intX ** 2) + (intY ** 2))

def angleBetweenChars(firstChar, secondChar):
    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))
    if fltAdj != 0.0:
        fltAngleInRad = math.atan(fltOpp / fltAdj)
    else:
        fltAngleInRad = 1.5708
    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)
    return fltAngleInDeg

def removeInnerOverlappingChars(listOfMatchingChars):
    listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)
    for currentChar in listOfMatchingChars:
        for otherChar in listOfMatchingChars:
            if currentChar != otherChar:
                if distanceBetweenChars(currentChar, otherChar) < (currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:
                        if currentChar in listOfMatchingCharsWithInnerCharRemoved:
                            listOfMatchingCharsWithInnerCharRemoved.remove(currentChar)
                    else:
                        if otherChar in listOfMatchingCharsWithInnerCharRemoved:
                            listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)
    return listOfMatchingCharsWithInnerCharRemoved

def recognizeCharsInPlate(imgThresh, listOfMatchingChars):
    strChars = ""
    height, width = imgThresh.shape
    imgThreshColor = np.zeros((height, width, 3), np.uint8)
    listOfMatchingChars.sort(key=lambda matchingChar: matchingChar.intCenterX)
    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR, imgThreshColor)
    for currentChar in listOfMatchingChars:
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = (currentChar.intBoundingRectX + currentChar.intBoundingRectWidth, currentChar.intBoundingRectY + currentChar.intBoundingRectHeight)
        cv2.rectangle(imgThreshColor, pt1, pt2, SCALAR_GREEN, 2)
        imgROI = imgThresh[currentChar.intBoundingRectY:currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                           currentChar.intBoundingRectX:currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]
        imgROIResized = cv2.resize(imgROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))
        npaROIResized = imgROIResized.reshape((1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))
        npaROIResized = np.float32(npaROIResized)
        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k=3)
        strCurrentChar = str(chr(int(npaResults[0][0])))
        strChars = strChars + strCurrentChar
    return strChars

# Detect Plate constants
PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5

def detectPlatesInScene(imgOriginalScene):
    listOfPossiblePlates = []
    imgGrayscaleScene, imgThreshScene = preprocess(imgOriginalScene)
    listOfPossibleCharsInScene = findPossibleCharsInScene(imgThreshScene)
    listOfListsOfMatchingCharsInScene = findListOfListsOfMatchingChars(listOfPossibleCharsInScene)
    print(f"Found {len(listOfListsOfMatchingCharsInScene)} groups of matching chars")  # Debug
    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
        if len(listOfMatchingChars) >= MIN_NUMBER_OF_MATCHING_CHARS:
            possiblePlate = extractPlate(imgOriginalScene, listOfMatchingChars)
            if possiblePlate.imgPlate is not None and possiblePlate.imgPlate.size > 0:
                listOfPossiblePlates.append(possiblePlate)
                print(f"Detected plate with {len(listOfMatchingChars)} chars")  # Debug
    return listOfPossiblePlates

def findPossibleCharsInScene(imgThresh):
    listOfPossibleChars = []
    intCountOfPossibleChars = 0
    imgThreshCopy = imgThresh.copy()
    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    height, width = imgThresh.shape
    imgContours = np.zeros((height, width, 3), np.uint8)
    for i in range(len(contours)):
        possibleChar = PossibleChar(contours[i])
        if checkIfPossibleChar(possibleChar):
            intCountOfPossibleChars += 1
            listOfPossibleChars.append(possibleChar)
    print(f"Found {intCountOfPossibleChars} possible chars")  # Debug
    return listOfPossibleChars

def extractPlate(imgOriginal, listOfMatchingChars):
    possiblePlate = PossiblePlate()
    if not listOfMatchingChars:
        return possiblePlate
    listOfMatchingChars.sort(key=lambda matchingChar: matchingChar.intCenterX)
    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[-1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[-1].intCenterY) / 2.0
    ptPlateCenter = (int(fltPlateCenterX), int(fltPlateCenterY))
    intPlateWidth = int((listOfMatchingChars[-1].intBoundingRectX + listOfMatchingChars[-1].intBoundingRectWidth - listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)
    intTotalOfCharHeights = sum(matchingChar.intBoundingRectHeight for matchingChar in listOfMatchingChars)
    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)
    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)
    fltOpposite = listOfMatchingChars[-1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[-1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse) if fltHypotenuse != 0 else 0
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)
    possiblePlate.rrLocationOfPlateInScene = (ptPlateCenter, (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg)
    height, width = imgOriginal.shape[:2]
    rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, fltCorrectionAngleInDeg, 1.0)
    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))
    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), ptPlateCenter)
    if imgCropped is not None and imgCropped.size > 0:
        possiblePlate.imgPlate = imgCropped
    else:
        print("Warning: Cropping failed, imgPlate set to None")
    return possiblePlate

# Main functions
def hasNumbers(mystring):
    return any(char.isdigit() for char in mystring)

def select_image():
    global panelA, panelB, image1, plate, plate_text
    path = filedialog.askopenfilename()
    if len(path) > 0:
        blnKNNTrainingSuccessful = loadKNNDataAndTrainKNN()
        if not blnKNNTrainingSuccessful:
            print("\nerror: KNN training was not successful\n")
            return
        path1 = path.split('/')
        k = path1[-1].split('.')[0] + ".png"
        imgOriginalScene = cv2.imread(path)
        if imgOriginalScene is None:
            print("\nerror: image not read from file \n\n")
            input("Press Enter to continue...")  # Cross-platform pause
            return
        else:
            listOfPossiblePlates = detectPlatesInScene(imgOriginalScene)
            listOfPossiblePlates = detectCharsInPlates(listOfPossiblePlates)
            flag = 0
            try:
                with open("plate.csv", "r") as f2:
                    f1 = csv.reader(f2, delimiter=',')
                    l = [row for row in f1]
                    for i in range(len(l)):
                        if l[i][2] == k:
                            text = l[i][3]; xmin = l[i][4]; ymin = l[i][5]; xmax = l[i][6]; ymax = l[i][7]
                            flag = 1
                            break
            except FileNotFoundError:
                l = []
                print("plate.csv not found, starting fresh.")
            os.makedirs('dataset/crops', exist_ok=True)
            if flag == 0:
                if len(listOfPossiblePlates) == 0:
                    print("\nno license plates were detected\n")
                    return
                else:
                    listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars) if possiblePlate.strChars else 0, reverse=True)
                    licPlate = listOfPossiblePlates[0] if listOfPossiblePlates else None
                    if licPlate and licPlate.imgPlate is not None:
                        cv2.imwrite('dataset/crops/' + k, licPlate.imgPlate)
                        gray = cv2.cvtColor(licPlate.imgPlate, cv2.COLOR_BGR2GRAY)
                        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                        if len(licPlate.strChars) == 0:
                            print("\nno characters were detected\n\n")
                            return
                        xmin, ymin, xmax, ymax = drawRectangleAroundPlate(imgOriginalScene, licPlate)
                        text = licPlate.strChars
                        next_id = 1 if not l else int(l[-1][0]) + 1
                        next_date = "2025-09-26" if not l else int(l[-1][1]) + 1  # Current date as placeholder
                        with open("plate.csv", 'a') as f3:
                            writer = csv.writer(f3, delimiter=',')
                            if not l or os.path.getsize("plate.csv") == 0:
                                writer.writerow(['ID', 'Date', 'Filename', 'Text', 'xmin', 'ymin', 'xmax', 'ymax'])
                            writer.writerow([next_id, next_date, k, licPlate.strChars, int(xmin), int(ymin), int(xmax), int(ymax)])
                    else:
                        print("No valid plate detected")
                        return
            if flag == 1:
                imgOriginalScene = cv2.rectangle(imgOriginalScene, (int(xmin), int(ymax)), (int(xmax), int(ymin)), (255, 0, 0), 2)
            plate_text = text
            test = cv2.imread('dataset/crops/' + k)
            writeLicensePlateCharsOnImage1(imgOriginalScene, xmin, ymin, text, test)
            plate = cv2.resize(test, (200, 80))
            image1 = cv2.resize(imgOriginalScene, (390, 240))
            plate = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            plate = Image.fromarray(plate)
            image1 = Image.fromarray(image1)
            plate = ImageTk.PhotoImage(plate)
            image1 = ImageTk.PhotoImage(image1)
            number(plate_text)
            if panelA is None or panelB is None:
                panelA = Label(MainFrame, image=image1)
                panelA.grid(row=4, column=0, sticky=W)
                panelB = Label(MainFrame, image=plate)
                panelB.grid(row=4, column=1, sticky=W)
            else:
                panelA.configure(image=image1)
                panelA.image = image1  # Fixed to use image1
                panelA.grid(row=4, column=0, sticky=W)
                panelB.configure(image=plate)
                panelB.image = plate
                panelB.grid(row=4, column=1, sticky=W)

def drawRectangleAroundPlate(imgOriginalScene, licPlate):
    if licPlate.rrLocationOfPlateInScene is None:
        print("Warning: rrLocationOfPlateInScene is None, using bounding rect")
        x, y, w, h = licPlate.boundingRect if hasattr(licPlate, 'boundingRect') else (0, 0, 0, 0)
        return x, y, x + w, y + h
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene).astype(int)
    intPoints = [(int(p[0]), int(p[1])) for p in p2fRectPoints]
    cv2.line(imgOriginalScene, intPoints[0], intPoints[1], SCALAR_RED, 2)
    cv2.line(imgOriginalScene, intPoints[1], intPoints[2], SCALAR_RED, 2)
    cv2.line(imgOriginalScene, intPoints[2], intPoints[3], SCALAR_RED, 2)
    cv2.line(imgOriginalScene, intPoints[3], intPoints[0], SCALAR_RED, 2)
    xs = [point[0] for point in intPoints]
    ys = [point[1] for point in intPoints]
    return min(xs), min(ys), max(xs), max(ys)

def writeLicensePlateCharsOnImage1(imgOriginalScene, xmin, ymin, text, test):
    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = test.shape
    intFontFace = cv2.FONT_HERSHEY_SIMPLEX
    fltFontScale = 1
    intFontThickness = int(round(fltFontScale * 1.5))
    cv2.putText(imgOriginalScene, text, (int(xmin) - 21, int(ymin)), intFontFace, fltFontScale, SCALAR_WHITE, intFontThickness)

def number(num_detected):
    global text_widget
    s = "Extracted Number is :\n"
    if 'text_widget' not in globals():
        text_widget = Text(MainFrame, font=('Arial', 16, 'bold'), height=5, width=20)  # Changed to Arial
        text_widget.grid(row=5, column=1, sticky=W)
    text_widget.delete(1.0, tk.END)
    text_widget.insert(tk.INSERT, s + num_detected)

# GUI
root = tk.Tk()
panelA = None
panelB = None

root.title("GUI : Number Plate")
root.geometry("880x530")
root.configure(background='white')
Tops = Frame(root, bg='blue', pady=1, width=1850, height=90, relief="ridge")
Tops.grid(row=0, column=0)

Title_Label = Label(Tops, font=('Arial', 20, 'bold'), text="         Number plate detection using Python \t\t", pady=9, bg='white', fg='red', justify="center")
Title_Label.grid(row=0, column=0)
MainFrame = Frame(root, bg='white', pady=2, padx=2, width=1450, height=100, relief="ridge")
MainFrame.grid(row=1, column=0)

Label_1 = Label(MainFrame, font=('Arial', 17, 'bold'), text="", padx=2, pady=2, fg="black", bg="white")  # Changed to Arial
Label_1.grid(row=0, column=0)

Label_9 = Button(MainFrame, font=('Arial', 17, 'bold'), text="  Select Image\n Detect Plate ", padx=2, pady=2, bg="white", fg="black", command=select_image)
Label_9.grid(row=2, column=0, sticky=W)

Label_2 = Label(MainFrame, font=('Arial', 10, 'bold'), text="\t\t    ", padx=2, pady=2, bg="white", fg="black")
Label_2.grid(row=3, column=0, sticky=W)

Label_3 = Label(MainFrame, font=('Arial', 30, 'bold'), text="      \t\t\t", padx=2, pady=2, bg="white", fg="black")
Label_3.grid(row=4, column=0)

Label_3 = Label(MainFrame, font=('Arial', 30, 'bold'), text="      \t\t\t", padx=2, pady=2, bg="white", fg="black")
Label_3.grid(row=5, column=0)

Label_3 = Label(MainFrame, font=('Arial', 30, 'bold'), text="      \t\t\t", padx=2, pady=2, bg="white", fg="black")
Label_3.grid(row=6, column=0)

Label_3 = Label(MainFrame, font=('Arial', 30, 'bold'), text="      \t\t\t", padx=2, pady=2, bg="white", fg="black")
Label_3.grid(row=7, column=0)

Label_3 = Label(MainFrame, font=('Arial', 30, 'bold'), text="      \t\t\t", padx=2, pady=2, bg="white", fg="black")
Label_3.grid(row=8, column=0)

Label_3 = Label(MainFrame, font=('Arial', 10, 'bold'), text="\t\t\t\t          ", padx=2, pady=2, bg="white", fg="black")
Label_3.grid(row=9, column=1)

root.mainloop()