import os
import sys
import numpy as np
import cv2
import collections
import argparse
import datetime
import time
cap = cv2.VideoCapture(str(sys.argv[1]))
inputnamelist = sys.argv[1].split('.')
outputname = str(inputnamelist[0]+'_output.'+str(sys.argv[3]))
print(outputname)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*str(sys.argv[2]))
out = cv2.VideoWriter(outputname,fourcc, fps, (width,height))
ret = True
framenumber = 0
cXlast = []
cYlast = []
refPt = []
roidifflist = []
xlast = 0
ylast = 0
framestabilized = True
need_track_feature = True
clicked = False

def click_and_drag(event, x, y, flags, param):
    global refPt, clicked, xlast, ylast
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        xlast = abs(refPt[1][0]-refPt[0][0])/2
        ylast = abs(refPt[1][1]-refPt[0][1])/2
        if refPt[0][0] < refPt[1][0]:
            xlast = refPt[0][0] + xlast
        else:
            xlast = refPt[1][0] + xlast
        if refPt[0][1] < refPt[1][1]:
            ylast = refPt[0][1] + ylast
        else:
            ylast = refPt[1][1] + ylast
        clicked = True

while ret is True:
#while framenumber < 2:
    framenumber = framenumber + 1
    ret, img1color = cap.read()
    if ret is False:
        break
#Find out if we need to get the tracking feature
    if need_track_feature is True:
        previmage = img1color.copy()
        while need_track_feature is True:
#Show the frame and wait for mouse click
            cv2.imshow('Pick a feature to track',img1color)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('s'):
                need_track_feature = False
                cv2.destroyAllWindows()
                break
            cv2.setMouseCallback('Pick a feature to track', click_and_drag)
            if clicked is True:
                print(refPt)
                need_track_feature = False
                clicked = False
                cv2.destroyAllWindows()
                imageroi = img1color[refPt[0][1]:refPt[1][1],refPt[0][0]:refPt[1][0]]
                #imageroi = cv2.cvtColor(imageroi, cv2.COLOR_BGR2GRAY)
                searchx1 = refPt[0][0]
                searchx2 = refPt[1][0]
                searchy1 = refPt[0][1]
                searchy2 = refPt[1][1]
#Get lowest x and y values so that we can use those as the starting point to search up and down by 5 pixels
                if searchx1 > searchx2:
                    searchx1 = searchx2
                if searchy1 > searchy2:
                    searchy1 = searchy2
#pre-populate the cached values as if they're from the last run so that they're available for later use.
                searchx1last = searchx1
                searchy1last = searchy1
#FOR VIDEO HANDLING ONLY - Remember original position to shift the images
                searchxorig = searchx1
                searchyorig = searchy1
#convert to grayscale for analysis
    #imggray = cv2.cvtColor(img1color, cv2.COLOR_BGR2GRAY)
    imggray = img1color.copy()
#remember how big the total image and ROI are 
    origheight, origwidth = img1color.shape[:2]
    roiheight, roiwidth = imageroi.shape[:2]
#Now scan the image vertically and horizontally to find the closest match
    heightscansize = int(img1color.shape[0]-imageroi.shape[0]+1)
    widthscansize = int(img1color.shape[1]-imageroi.shape[1]+1)
#Set up the end of the maximum search time
    searchend = time.time() + 1
    finalroidiff = float('inf')
    difflowered = False
    keepgoing = True
#pull the latest searchx and searchy coordinates from the last known position
    searchx1 = searchx1last
    searchy1 = searchy1last
    while keepgoing is True:
        for ycheck in range((searchy1-15),(searchy1+15)):
            if time.time() > searchend:
                break
            for xcheck in range((searchx1-15),(searchx1+15)):
#set up the roi to search within the original image
                imagecomp = imggray[ycheck:int(ycheck+roiheight),xcheck:int(xcheck+roiwidth)]
#subtract the reference roi from the search area and get the difference of the arrays
                imagecompforsub = imagecomp.astype(np.int8)
                imageroiforsub = imageroi.astype(np.int8)
                imagediff = imagecompforsub - imageroiforsub
                imagediff = np.absolute(imagediff)
                imagediff = np.sum(imagediff)
                imagediff = (imagediff/(np.sum(imageroi)))*100
#if we dropped to a new minimum, save the new minimum diff and save the x and y coordinates we're at.  Set diff lowered flag to true
                if imagediff < finalroidiff:
                    finalroidiff = imagediff
                    searchx2 = xcheck
                    searchy2 = ycheck
                    difflowered = True
#check if we ran out of time
                if time.time() > searchend:
                    break   
#back on the keep going loop, check if the diff lowered in the last search run.  If not, we found a local minimum and don't need to keep going.  If we did, start a new search around the new location
        if difflowered is True:
            keepgoing = True
#check and make sure the new position of the region of interest won't put us over the border of the window, correct back to the edge of border if it would, otherwise take the new roi coordinates
            if (searchx2 - 16) < 0: 
                searchx1 = (-1*(searchx2 - 16))+searchx2
                print('Edgelord Detected 1')
                print(searchx1)
            elif (searchx2 + 16) > (origwidth - roiwidth):
                searchx1 = ((origwidth - roiwidth) - (searchx2 + 16))+searchx2
                print('Edgelord Detected 2')
                print(searchx1)
            else:
                searchx1 = searchx2
            if (searchy2 - 16) < 0: 
                searchy1 = (-1*(searchy2 - 16))+searchy2
                print('Edgelord Detected 3')
                print(searchy1)
            elif (searchy2 + 16) > (origheight - roiheight):
                searchy1 = ((origheight - roiheight) - (searchy2 + 16))+searchy2
                print('Edgelord Detected 4')
                print(searchy1)
            else:
                searchy1 = searchy2
            difflowered = False
        else:
            keepgoing = False
        if time.time() > searchend:
            print('outtatime')
            break   
    print(finalroidiff)
#figure out if the difference from roi is low enough to be acceptable
    if finalroidiff < 10:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            need_track_feature = True
        searchx1last = searchx1
        searchy1last = searchy1
        learnimg = imggray[searchy1last:(searchy1last+roiheight),searchx1last:(searchx1last+roiwidth)]
        imageroi = (imageroi * 0.9) + (learnimg * 0.1)
    else:
        print("Didn't find it, keep looking at last known coordinates.")
        need_track_feature = True
    cv2.imshow('Track ROI',imggray[searchy1last:(searchy1last+roiheight),searchx1last:(searchx1last+roiwidth)])
    mainimg = img1color.copy()
    if need_track_feature is False:
        cv2.rectangle(mainimg,(searchx1last,searchy1last), ((searchx1last+roiwidth),searchy1last+roiheight),(0,255,0),3)
    else:
        cv2.rectangle(mainimg,(searchx1last,searchy1last), ((searchx1last+roiwidth),searchy1last+roiheight),(0,0,255),3)
    cv2.imshow('Main Image',mainimg)
    cv2.waitKey(1)
    cXdiff = searchxorig - searchx1last
    cYdiff = searchyorig - searchy1last
    M = np.float32([[1,0,cXdiff],[0,1,cYdiff]])
    rows,cols,colors = imggray.shape
    stabilizedframe = cv2.warpAffine(img1color.copy(),M,(cols,rows))
    out.write(stabilizedframe)
    print(framenumber)
if ret is False:
#if framenumber == 2:
    cap.release()
    out.release()
    exit()
