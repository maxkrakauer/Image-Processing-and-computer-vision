"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from mimetypes import init
from typing import List

import numpy as np
import cv2
import matplotlib.pyplot as plt
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2

num=0
denom=0


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 302256102


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    img = cv2.imread(filename)
    if representation==1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(img.dtype)
    print(img.shape)
    #print(img)
    img = img/255.0
    #print(img)
    print(img.dtype)
    return img
    pass


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img = imReadAndConvert(filename,representation)
    #img = img*255
    #img = img*255.0
    #img = img.astype(int)
    print (img)
    img = np.float32(img)
    if representation==2:
        #img = img*255.0
        #img = img.astype(float32)
        #img = np.float32(img)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    #img = img*255.0
    plt.imshow(img); 
    plt.show()
    pass


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    img = np.zeros_like(imgRGB)
    img[:,:,0] = .299*imgRGB[:,:,0] + .587*imgRGB[:,:,0] + .144*imgRGB[:,:,0]
    img[:,:,1] = .596*imgRGB[:,:,1] - .275*imgRGB[:,:,1] - .321*imgRGB[:,:,1]
    img[:,:,2] = .212*imgRGB[:,:,2] - .523*imgRGB[:,:,2] + .311*imgRGB[:,:,2]
    return img
    pass


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    img = np.zeros_like(imgYIQ)
    img[:,:,0] = imgYIQ[:,:,0] + (242179/253408)*imgYIQ[:,:,0] + (157077/253408)*imgYIQ[:,:,0]
    img[:,:,1] = imgYIQ[:,:,1] - (68821/253408)*imgYIQ[:,:,1] - (163923/253408)*imgYIQ[:,:,1]
    img[:,:,2] = imgYIQ[:,:,2] - (280821/253408)*imgYIQ[:,:,2] + (423077/253408)*imgYIQ[:,:,2]
    return img
    pass

def rgb2yiq(img: np.ndarray):
    img = transformRGB2YIQ(img)
    img_flat = img[:,:,0]
    return img, img_flat
    

def setHist(orig: np.ndarray, img: np.ndarray, img_flat: np.ndarray, colors: int):
    if(colors>2):
        img = transformRGB2YIQ(img)
        img_flat = img_flat[:,:,0]
    img_flat=img_flat.flatten()
    img_flat=img_flat*255
    print("printing img_flat inside of setHist")
    print(img_flat)
    histOrg, bins = np.histogram(img_flat, bins=np.arange(257))
    return histOrg, img_flat



def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    img=imgOrig
    img_flat=img
    size = imgOrig.shape[0]*imgOrig.shape[1]
    colors = imgOrig.ndim
    histOrg, img_flat = setHist(imgOrig, img, img_flat, colors)
    cumul = np.cumsum(histOrg)
    lut = np.around((cumul/size)*255)
    temp = np.zeros_like(img_flat)
    #print("printing lut: ")
    #print(lut)
    print("printing img_flat")
    print(img_flat)
    print("before lookup")
    def lookup(x):
        temp[x==img_flat] = lut[x]
    vfunc = np.vectorize(lookup)
    vfunc(np.arange(256))
    print("printing temp")
    print(temp)
    print("after lookup")
    histEQ, bins = np.histogram(temp,bins=np.arange(257))
    print("here is temp before reshape: ", temp)
    temp = temp.reshape(img.shape[0], img.shape[1])
    temp=temp/255
    if(colors>2):
        img[:,:,0] = temp
        print("after img[:,:,0] = temp, printing img")
        print(img)
        img = transformYIQ2RGB(img)
        print("after tranformation, img")
        print(img)
    else:
        img = temp
    print("before imEq = img, printing img")
    print(img)
    imEq = img
    print(img)
    print(imEq.shape)
    print("now printing imEq")
    print(imEq)
    #imEq = np.reshape(imEq, (imgOrig.shape))
    print("before return")
    return imEq, histOrg, histEQ 
    pass

def createQuantImg(img_flat: np.ndarray, quant_amount: int, colors: int, z: List[int], q: List[int], quant_img_list: List[np.ndarray]):
    """
    Take img and  
    """
    img_flat= img_flat*255
    print("printing z: ", z)
    print("printing q: ", q)
    print("printing img_flat: ", img_flat)
    quant_img = np.zeros_like(img_flat)
    def setToQuant(x):
        #new_arr[((25 <= arr) & (arr <=100))] = -22
        if(x==quant_amount):
            quant_img[((z[x] <= img_flat) & (img_flat <=z[x+1]))] = q[x]
        else:
            quant_img[((z[x] <= img_flat) & (img_flat <z[x+1]))] = q[x]
    vecNewImg = np.vectorize(setToQuant)
    vecNewImg(np.arange(quant_amount))
    quant_img_list.append(quant_img)
    return quant_img
    pass

def calculateMSE(img_flat: np.ndarray, mse_list: List[float], quant_img: np.ndarray):
    pixel_amount = img_flat.shape[0]
    print("printing img_flat in calculatemse: ", img_flat)
    print("printing quant_img in calculatemse: ", quant_img)
    mse = np.sqrt(np.sum(np.square((img_flat*255)-quant_img)))/pixel_amount
    mse_list.append(mse)
    pass

def readjustBounds(z: List[int], q: List[int], quant_amount: int):
    print("q in readjustBounds: ", q)
    print("z in readjustBounds: ", z)

    def adjustBound(x):
        z[x+1] = round((q[x]+q[x+1])/2)
    vecAdjustBounds = np.vectorize(adjustBound)
    vecAdjustBounds(np.arange(quant_amount-1))
    return z
    pass

def readjustQuants(img_flat: np.ndarray, quant_amount: int, z: List[int]):
    q = []
    hist, bins = np.histogram(img_flat*255, bins=np.arange(257))
    print ("histogram inside readjustQuants", hist)
    print(z)
    def integralTop(i):
        global num
        print("num is: ", num)
        print("i is: ",i)
        num = num + hist[i]*i
    def integralBot(i):
        global denom
        print("denom is: ", denom)
        denom = denom+hist[i]  
    vfuncTop = np.vectorize(integralTop)
    vfuncBot = np.vectorize(integralBot)
    for j in range(quant_amount): 
        vfuncTop(np.arange(z[j], z[j+1]))
        vfuncBot(np.arange(z[j], z[j+1]))
        frac = -1
        if denom!=0:
            frac=round(num/denom)
            q.append(frac)
    return q

def initBounds(z:List[int], quant_amount:int) -> (List[int]):
    bound_size = 255/quant_amount
    z = np.arange(0,255,bound_size)
    z=np.around(z)
    z = np.append(z,255)
    print ("printing z in initBounds: ", z)
    return z


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    quant_amount = nQuant
    colors=imOrig.ndim
    #img = imOrig*255
    img = imOrig
    img_flat=img
    if(colors>2):
        img, img_flat = rgb2yiq(img)
    img_flat = img_flat.flatten()
    print("printing img_flat at start of quantizeimage: ", img_flat)
    #img_flat = img_flat*255
    z=[]
    q=[]
    mse_list=[]
    quant_img_list=[]
    z = initBounds(z,quant_amount)
    z = z.astype(int)
    for i in range(nIter):
        print()
        print()
        print("iteration ",i)
        print()
        print()
        print ("q at start of for loop: ",q)
        print ("z at start of for loop: ",z)
        global num, denom
        num=0
        denom=0
        q = readjustQuants(img_flat, quant_amount, z)
        #create the new quantisised image. 
        quant_img = createQuantImg(img_flat, quant_amount, colors, z, q, quant_img_list)
        calculateMSE(img_flat, mse_list, quant_img)
        print("q before readjustbounds: ", q)
        z = readjustBounds(z, q, quant_amount)
        print("z after readjustBounds: ", z)
    plt.plot(mse_list)
    for k in range(nIter):
        quant_img_list[k] = quant_img_list[k].reshape(imOrig.shape[0], imOrig.shape[1])
        pass
    if(colors>2):
        img_list=[]
        for k in range (nIter):
            img[:,:,0] = quant_img_list[k]
            quant_img = transformYIQ2RGB(img)
            img_list.append(quant_img)
        quant_img_list=img_list
    print("printing quant_img: ", quant_img_list)
    print("Printing mse_list: ", mse_list)
    return quant_img_list, mse_list
    pass
