import cv2
import numpy as np
import xlsxwriter
import argparse
from imutils import contours

colors = [(255,0,0),(0,255,0), (0,0,255), (255,255,0),(0,255,255), (255,0,255)]

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

def getCorners(box):
    npsum = np.sum(box, axis = 1)
    n1, n2 = np.argmin(npsum), np.argmax(npsum)
    return box[n1], box[n2]

def processImage(INPUT, OUTPUT, DEBUG, VERBOSE):
    image = cv2.imread(INPUT)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    numcnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in numcnts:
        if cv2.contourArea(c) < 1000:
            cv2.drawContours(thresh, [c], -1, 255, -1)
            
    noNums = thresh.copy()
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vertical_kernel, iterations=9)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horizontal_kernel, iterations=4)

    thresh = cv2.bitwise_not(thresh)
    mask = np.zeros((thresh.shape), dtype = np.uint8)

    cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    allCnts = (np.concatenate(cnts))
    rect = cv2.minAreaRect(allCnts)
    box = np.int0(cv2.boxPoints(rect))
    TL, BR = getCorners(box)

    thresh = noNums[TL[1] : BR[1] + 1, TL[0] : BR[0] + 1]

    boxcnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    out = image[TL[1] : BR[1] + 1, TL[0] : BR[0] + 1]

    goodContours = [(c) for c in boxcnts if cv2.isContourConvex(c)]
    goodContours, _ = contours.sort_contours(goodContours, method="top-to-bottom")
    rects = [np.int0(cv2.boxPoints(cv2.minAreaRect(c))) for c in goodContours]
    
    
    rows = []
    currentRow = []
    counter = 0
    currentHeight = 0
    while counter < len(rects):
        if len(currentRow):
            TL, BR = getCorners(rects[counter])
            if abs(TL[1] - currentHeight) <= 10:
                currentRow.append(np.mean([TL, BR], 0))
                counter += 1
            else:
                rows.append(currentRow[:])
                currentRow = []
        else:
            TL, BR = getCorners(rects[counter])
            currentRow.append(np.mean([TL,BR], 0))
            currentHeight = TL[1]
            counter += 1
    rows.append(currentRow)
    rows = [sorted(row, key = lambda e: e[0]) for row in rows]
    #print(*rows, sep="\n")
    debugOutput = np.copy(out)
    for row in rows:
        for i in range(len(row)):
            cv2.circle(debugOutput, (int(row[i][0]), int(row[i][1])), 10, (255,255,0), 2)
    
    minVal = out.shape[1]
    maxVal = 0
    for row in rows:
        maxVal = max(row[len(row) - 1][0] - row[0][0], maxVal)
        for i in range(len(row) - 1):
            minVal = min(row[i+1][0] - row[i][0], minVal)
    
    HEIGHT, WIDTH = round(out.shape[0] / minVal), round(out.shape[1] / minVal)
    
    while DEBUG:
        cv2.imshow("Press any key to close", debugOutput)
        key = cv2.waitKey(1)
        if key != -1:
            break
    
    sideLen = ( thresh.shape[0] / HEIGHT + thresh.shape[1] / WIDTH ) / 2
    if VERBOSE:
        print(thresh.shape)
        print ("Box size: " + str(sideLen))
        print("Estimated Dimensions: " + str(HEIGHT) + " x " + str(WIDTH))
    workbook = xlsxwriter.Workbook(OUTPUT)
    worksheet = workbook.add_worksheet()
        
    for i in range(HEIGHT):
        for j in range(WIDTH):
            color = out[int((i+0.5) * sideLen),int((j+0.5) * sideLen)][::-1]
            cv2.circle(out, (int((j+0.5) * sideLen),int((i+0.5) * sideLen)), 10, (255,255,0), 2)
            cell_format = workbook.add_format({'bold': True, 'bg_color': rgb_to_hex(tuple(color)), 'border_color': 'black', 'border': 1})
            worksheet.write(i, j, "", cell_format)

         
    while DEBUG:
        cv2.imshow("Press any key to close", out)
        key = cv2.waitKey(1)
        if key != -1:
            break
            
    workbook.close()
    if VERBOSE:
        print("Sheet written to " + OUTPUT)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input file name")
    parser.add_argument("output", nargs = "?", default = "out.xlsx", help="output file name (optional, defaults to out.xlsx)")
    parser.add_argument("-d", "--debug", action = "store_true", help="show debug frame")
    parser.add_argument("-v", "--verbose", action = "store_true", help="show verbose outputs")
    args = parser.parse_args()
    processImage(args.input, args.output, args.debug, args.verbose)

if __name__ == "__main__":
   main()
