'''
============UNG DUNG XOA FONT===============
Chuot phai vao cua so input de keo 1 vung hinh chu nhat
Nhan phim:
    f : de chon tien canh
    b : de chon hau canh
    r : de lam lai tu dau
    s : de luu lai hinh da chon
=============================================

'''

from __future__ import print_function
import argparse
import numpy as np
import cv2 as cv
import sys
from time import gmtime, strftime

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help="duong dan toi image")
args = vars(ap.parse_args())
img = cv.imread(args["image"],1)
pic = img

h, w = img.shape[:2]

#Dinh nghia cac mau
BLUE = [255,0,0]        # Mau hinh chu nhat
BLACK = [0,0,0]         # Chac chan la hau canh
WHITE = [255,255,255]   # Chac chan la tien canh

DRAW_BG = {'color' : BLACK, 'val' : 0}
DRAW_FG = {'color' : WHITE, 'val' : 1}

# Dinh nghia cac flag
rect = (0,0,1,1)
drawing = False         # flag ve duong cong
rectangle = False       # flag ve hinh chu nhat
rect_over = False       # flag kiem tra ve hinh chu nhat
rect_or_mask = 100      # flag kiem tra trang thai ve hinh chu nhat hay mat la
value = DRAW_FG         # Khoi tao ve hau canh
thickness = 1           # Do day cua but
burth = 10

# Thao tac ben cua so input
def onMouse(event,x,y,flags,param):
    global img,img2,drawing,value,mask,rectangle,rect,rect_or_mask,ix,iy,rect_over

    # Ve hinh chu nhat
    if event == cv.EVENT_RBUTTONDOWN:
        rectangle = True
        ix,iy = x,y

    elif event == cv.EVENT_MOUSEMOVE:
        if rectangle == True:
            img = img2.copy()
            cv.rectangle(img,(ix,iy),(x,y),BLUE,2)
            rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
            rect_or_mask = 0

    elif event == cv.EVENT_RBUTTONUP:
        rectangle = False
        rect_over = True
        cv.rectangle(img,(ix,iy),(x,y),BLUE,2)
        rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
        rect_or_mask = 0
        print(" An n de chap nhan thay doi")

    # draw touchup curves
    if event == cv.EVENT_LBUTTONDOWN:
        if rect_over == False:
            print("Dau tien ve hinh chu nhat \n")
        else:
            drawing = True
            cv.circle(img,(x,y),thickness,value['color'],-1)
            cv.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            cv.circle(img,(x,y),thickness,value['color'],-1)
            cv.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv.EVENT_LBUTTONUP:
        if drawing == True:
            drawing = False
            cv.circle(img,(x,y),thickness,value['color'],-1)
            cv.circle(mask,(x,y),thickness,value['val'],-1)

# Thao tac ben cua so output
def onFix(event,x,y,flags,param):
    global img,img2,drawing,value,mask,rectangle,rect,rect_or_mask,ix,iy,rect_over  

    # draw touchup curves
    if event == cv.EVENT_LBUTTONDOWN:
        if rect_over == False:
            print("first draw rectangle \n")
        else:
            drawing = True
            cv.circle(mask,(x,y),burth,value['val'],-1)

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            cv.circle(mask,(x,y),burth,value['val'],-1)

    elif event == cv.EVENT_LBUTTONUP:
        if drawing == True:
            drawing = False
            cv.circle(mask,(x,y),burth,value['val'],-1)

def toCreateMask(img):
    pic = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, threshold = cv.threshold(pic, 0, 255, cv.THRESH_BINARY)
    return threshold

def removeBG(BG, mask_person):
    foreground = BG
    background = BG
    alpha = mask_person

    alpha = cv.GaussianBlur(alpha, (7,7), 0)

    # Chuyen doi uint8 sang float
    foreground = foreground.astype(float)
    background = background.astype(float)
    
    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float)/255
    
    # Multiply the foreground with the alpha matte
    foreground = cv.multiply(alpha, foreground)
    
    # Multiply the background with ( 1 - alpha )
    background = cv.multiply(1.0 - alpha, background)
    background = cv.GaussianBlur(background, (7,7), 0)
    
    outImage = cv.add(foreground, background)
    result = outImage/255;
    return result;

if __name__ == '__main__':

    height, width = img.shape[:2]

    img2 = img.copy()                               
    mask = np.zeros(img.shape[:2],dtype = np.uint8) 
    output = np.zeros(img.shape,np.uint8)           

    # Tao ra 2 cua so
    cv.namedWindow('input', cv.WINDOW_NORMAL)
    cv.setMouseCallback('input',onMouse)
    cv.resizeWindow("input", w, h)
    cv.moveWindow('input', 0, 0)

    cv.namedWindow('output', cv.WINDOW_NORMAL)
    cv.setMouseCallback('output', onFix)
    cv.resizeWindow("output", w, h)
    cv.moveWindow('output', img.shape[1], 0)

    while(1):

        cv.imshow('output',output)
        cv.imshow('input',img)
        k = cv.waitKey(1)

        # bat su kien keyboard
        if k == 27:         # esc de thoat
            break
        elif k == ord('f'): # BG ve
            print("Lay tien canh \n")
            value = DRAW_FG
        elif k == ord('b'): # FG ve
            print("Lay hau canh \n")
            value = DRAW_BG
        elif k == ord('s'): # Luu anh
            
            mask_person = toCreateMask(output)
            mask_person = cv.cvtColor(mask_person, cv.COLOR_GRAY2BGR) 
            result = removeBG(pic, mask_person)
            filename = "IMG" + strftime("%H%M") + ".png";
            cv.imwrite(filename,result*255)
            print("Thanh cong \n")
            break;
        elif k == ord('r'): # reset tat ca
            print("resetting \n")
            rect = (0,0,1,1)
            drawing = False
            rectangle = False
            rect_or_mask = 100
            rect_over = False
            value = DRAW_FG
            img = img = img2.copy().copy()
            mask = np.zeros(img.shape[:2],dtype = np.uint8) 
            output = np.zeros(img.shape,np.uint8)           
        elif k == ord('o'):
            burth = burth + 2;
            thickness = thickness + 1;
        elif k == ord('p'):
            if burth > 2 | thickness > 3:
                burth = burth - 2;
                thickness = thickness - 1;
        elif k == ord('n'): 
            print("""Up date thanh cong\n""")
            if (rect_or_mask == 0):         # grabcut cung rect
                bgdmodel = np.zeros((1,65),np.float64)
                fgdmodel = np.zeros((1,65),np.float64)
                cv.grabCut(img2,mask,rect,bgdmodel,fgdmodel,1,cv.GC_INIT_WITH_RECT)
                rect_or_mask = 1
            elif rect_or_mask == 1:         # grabcut cung mask
                bgdmodel = np.zeros((1,65),np.float64)
                fgdmodel = np.zeros((1,65),np.float64)
                cv.grabCut(img2,mask,rect,bgdmodel,fgdmodel,1,cv.GC_INIT_WITH_MASK)

        mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')
        output = cv.bitwise_and(img2,img2,mask=mask2)

    cv.destroyAllWindows()