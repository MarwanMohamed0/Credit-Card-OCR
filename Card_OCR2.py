import cv2
import imutils
from imutils import contours
import numpy as np
import argparse

# -------------------------------
# ARGUMENTS
# -------------------------------
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Path to input image")
args = vars(ap.parse_args())
image_path = args['image']

# -------------------------------
# REFERENCE FONTS
# -------------------------------
FONT_DIGITS = 'font_images/ocr_a_reference.png'
FONT_LETTERS = 'font_images/OCRA.png'

FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}

# -------------------------------
# HELPER FUNCTION: Load Template ROIs
# -------------------------------
def find_ROI(path):
    """Load a font image and return dictionary of individual character ROIs."""
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts, method='left-to-right')[0]

    rois = {}
    for i, c in enumerate(cnts):
        x, y, w, h = cv2.boundingRect(c)
        roi = thresh[y:y+h, x:x+w]
        roi = cv2.resize(roi, (57, 88))
        rois[i] = roi
    return rois

# -------------------------------
# PREPROCESS CARD IMAGE
# -------------------------------
def preprocess_card(image_path):
    """Return sorted contours of potential digit/letter regions on the card."""
    rectkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = imutils.resize(gray, width=300)

    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectkernel)
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    gradX = 255 * (gradX - np.min(gradX)) / (np.max(gradX) - np.min(gradX))
    gradX = gradX.astype('uint8')
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectkernel)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqkernel)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts, method='left-to-right')[0]
    return cnts, gray

# -------------------------------
# MATCH DIGITS
# -------------------------------
def match_digits(cnts, gray, digits_templates):
    locs = []
    output = []

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)
        if 2.5 < ar < 4.0 and 40 < w < 55 and 10 < h < 20:
            locs.append((x, y, w, h))

    locs = sorted(locs, key=lambda x: x[0])

    for gX, gY, gW, gH in locs:
        group = gray[gY-5:gY+gH+5, gX-5:gX+gW+5]
        group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        grpcnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        grpcnts = imutils.grab_contours(grpcnts)
        grpcnts = contours.sort_contours(grpcnts, method='left-to-right')[0]

        for c in grpcnts:
            x, y, w, h = cv2.boundingRect(c)
            roi = cv2.resize(group[y:y+h, x:x+w], (57, 88))
            scores = [cv2.matchTemplate(roi, digits_templates[d], cv2.TM_CCOEFF)[0][0] 
                      for d in digits_templates]
            output.append(str(np.argmax(scores)))
    return output

# -------------------------------
# MATCH LETTERS
# -------------------------------
def match_letters(cnts, gray, letter_templates):
    locs = []
    output = ''

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if 145 < y < gray.shape[0]-8 and 10 < x < gray.shape[1]*5/8:
            locs.append((x, y, w, h))

    locs = sorted(locs, key=lambda x: x[0])

    for gX, gY, gW, gH in locs:
        group = gray[gY-5:gY+gH+5, gX-5:gX+gW+5]
        group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        grpcnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        grpcnts = imutils.grab_contours(grpcnts)
        grpcnts = contours.sort_contours(grpcnts, method='left-to-right')[0]

        for c in grpcnts:
            x, y, w, h = cv2.boundingRect(c)
            roi = cv2.resize(group[y:y+h, x:x+w], (57, 88))
            scores = [cv2.matchTemplate(roi, letter_templates[i][1], cv2.TM_CCOEFF)[0][0]
                      for i in range(len(letter_templates))]
            index = np.argmax(scores)
            output += letter_templates[index][0]
        output += ' '
    return output.strip()

# -------------------------------
# PREPARE LETTER TEMPLATES
# -------------------------------
sample = find_ROI(FONT_LETTERS)
char = [
    ['A', sample[1]], ['B', sample[7]], ['C', sample[12]], ['D', sample[17]],
    ['E', sample[22]], ['F', sample[27]], ['G', sample[32]], ['H', sample[36]],
    ['I', sample[42]], ['J', sample[50]], ['K', sample[54]], ['L', sample[60]],
    ['M', sample[67]], ['N', sample[73]], ['O', sample[82]], ['P', sample[87]],
    ['Q', sample[0]], ['R', sample[6]], ['S', sample[11]], ['T', sample[16]],
    ['U', sample[21]], ['V', sample[26]], ['W', sample[31]], ['X', sample[35]],
    ['Y', sample[41]], ['Z', sample[47]]
]

digits_templates = find_ROI(FONT_DIGITS)

# -------------------------------
# MAIN PROCESS
# -------------------------------
cnts, gray = preprocess_card(image_path)
output_digits = match_digits(cnts, gray, digits_templates)
output_letters = match_letters(cnts, gray, char)

print("\nRESULT:")
print("Card Number     :", "".join(output_digits))
print("Card Type       :", FIRST_NUMBER.get(output_digits[0], "Unknown"))
print("Card Holder Name:", output_letters)
