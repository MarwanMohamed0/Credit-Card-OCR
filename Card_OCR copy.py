import cv2
import imutils
from imutils import contours
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i','--image',required=True,help = "Path to input image")
args = vars(ap.parse_args())

image_path = args['image']
font_path_d = 'font_images/OCRA.png'
font_path_a = 'font_images/ocr_a_reference.png'


# ============================================================
#  FIND ROI FOR LETTERS/DIGITS TEMPLATES
# ============================================================
def find_ROI(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts, method='left-to-right')[0]

    sample = {}
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        roi = thresh[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))
        sample[i] = roi

    return sample


# ============================================================
#  PREPROCESS CARD IMAGE TO FIND CHARACTER CONTOURS
# ============================================================
def preprocessing_find_contours(path):

    rectkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = imutils.resize(gray, width=300)

    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectkernel)

    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))

    gradX = 255 * (gradX - minVal) / (maxVal - minVal)
    gradX = gradX.astype('uint8')

    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectkernel)

    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqkernel)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts, method='left-to-right')[0]

    return cnts


# Load templates
sample = find_ROI(font_path_d)
digits = find_ROI(font_path_a)


# ============================================================
#  LETTER TEMPLATE MAPPING
# ============================================================
char = {
    0 : ['A' , sample[1]],
    1 : ['B', sample[7]],
    2 : ['C', sample[12]],
    3 : ['D', sample[17]],
    4 : ['E', sample[22]],
    5 : ['F', sample[27]],
    6 : ['G', sample[32]],
    7 : ['H', sample[36]],
    8 : ['I', sample[42]],
    9 : ['J', sample[50]],
    10 : ['K', sample[54]],
    11 : ['L', sample[60]],
    12 : ['M', sample[67]],
    13 : ['N', sample[73]],
    14 : ['O', sample[82]],
    15 : ['P', sample[87]],
    16 : ['Q', sample[0]],
    17 : ['R', sample[6]],
    18 : ['S', sample[11]],
    19 : ['T', sample[16]],
    20 : ['U', sample[21]],
    21 : ['V', sample[26]],
    22 : ['W', sample[31]],
    23 : ['X', sample[35]],
    24 : ['Y', sample[41]],
    25 : ['Z', sample[47]],
    26 : ['A', sample[3]],
    27 : ['B', sample[9]],
    28 : ['C', sample[14]],
    29 : ['D', sample[19]],
    30 : ['E', sample[24]],
    31 : ['F', sample[29]],
    32 : ['G', sample[34]],
    33 : ['H', sample[38]],
    34 : ['I', sample[44]],
    35 : ['J', sample[49]],
    36 : ['K', sample[56]],
    37 : ['L', sample[62]],
    38 : ['M', sample[68]],
    39 : ['N', sample[74]],
    40 : ['O', sample[83]],
    41 : ['P', sample[88]],
    42 : ['Q', sample[2]],
    43 : ['R', sample[8]],
    44 : ['S', sample[13]],
    45 : ['T', sample[18]],
    46 : ['U', sample[23]],
    47 : ['V', sample[28]],
    48 : ['W', sample[33]],
    49 : ['X', sample[37]],
    50 : ['Y', sample[43]],
    51 : ['Z', sample[48]],
}


FIRST_NUMBER = {
	"3": "American Express",
	"4": "Visa",
	"5": "MasterCard",
	"6": "Discover Card"
}


cnts = preprocessing_find_contours(image_path)


# ============================================================
#  EXTRACT & RECOGNIZE DIGITS (CARD NUMBER)
# ============================================================
def for_digits(cnts,path):

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = imutils.resize(gray, width=300)

    locs_d = []
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if ar > 2.5 and ar < 4.0:
            if (w > 40 and w < 55) and (h > 10 and h < 20):
                locs_d.append((x, y, w, h))

    locs_d = sorted(locs_d, key=lambda x: x[0])

    output = []

    for (i, (gX, gY, gW, gH)) in enumerate(locs_d):
        group_output = []
        group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
        group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        grpcnts = cv2.findContours(group, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        grpcnts = imutils.grab_contours(grpcnts)
        grpcnts = contours.sort_contours(grpcnts, method='left-to-right')[0]

        for c in grpcnts:
            (x, y, w, h) = cv2.boundingRect(c)
            roi = group[y:y + h, x:x + w]
            roi = cv2.resize(roi, (57, 88))
            scores = []

            for (digit, digitROI) in digits.items():
                result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
                (_, score, _, _) = cv2.minMaxLoc(result)
                scores.append(score)

            group_output.append(str(np.argmax(scores)))

        output.extend(group_output)

    return output


# ============================================================
#  EXTRACT & RECOGNIZE CARD HOLDER NAME (LETTERS)
# ============================================================
def for_alphabets(cnts,path):

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = imutils.resize(gray, width=300)

    locs_a = []
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        if y > 145 and y < (gray.shape[0] - 8) and x < (gray.shape[1] * 5 / 8) and x > 10:
            locs_a.append((x, y, w, h))

    locs_a = sorted(locs_a, key=lambda x: x[0])

    output = ''

    for (i, (gX, gY, gW, gH)) in enumerate(locs_a):
        group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
        group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        grpcnts = cv2.findContours(group, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        grpcnts = imutils.grab_contours(grpcnts)
        grpcnts = contours.sort_contours(grpcnts, method='left-to-right')[0]

        card_name = ''
        for c in grpcnts:
            (x, y, w, h) = cv2.boundingRect(c)
            roi = group[y:y + h, x:x + w]
            roi = cv2.resize(roi, (57, 88))
            scores = []

            for i in range(len(char)):
                result = cv2.matchTemplate(roi, char[i][1], cv2.TM_CCOEFF)
                (_, score, _, _) = cv2.minMaxLoc(result)
                scores.append(score)

            index_max_score = np.argmax(scores)
            card_name = card_name + char[index_max_score][0]

        output = output + " " + card_name

    return output

# ============================================================
#  IMPROVED: EXTRACT EXPIRATION DATE (MM/YY)
# ============================================================
def extract_expiration_date(image_path):
    """
    Extracts expiration date by looking for pairs of digits 
    in the lower-right region of the card, excluding the main card number area.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = imutils.resize(gray, width=300)

    h, w = gray.shape
    
    # Focus on lower portion, right side (typical expiry location)
    # Adjust these based on your card images
    y_start = int(h * 0.60)  # Start at 60% down the card
    y_end = int(h * 0.85)    # End at 85% down
    x_start = int(w * 0.50)  # Start at 50% across (right half)
    x_end = w
    
    exp_region = gray[y_start:y_end, x_start:x_end]
    
    # Apply preprocessing
    exp_region = cv2.GaussianBlur(exp_region, (3, 3), 0)
    exp_region = cv2.threshold(exp_region, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Find contours
    cnts = cv2.findContours(exp_region.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    if len(cnts) == 0:
        return "Unknown", "Unknown"
    
    # Filter contours by size (individual digits, not groups)
    digit_contours = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        
        # Looking for individual digit-like contours
        if 0.3 < aspect_ratio < 1.0 and h > 8 and h < 25 and w > 4 and w < 20:
            digit_contours.append((x, y, w, h))
    
    # Sort left to right
    digit_contours = sorted(digit_contours, key=lambda x: x[0])
    
    # Extract digits
    exp_digits = []
    for (x, y, w, h) in digit_contours:
        roi = exp_region[y:y+h, x:x+w]
        roi = cv2.resize(roi, (57, 88))
        
        scores = []
        for (digit, digitROI) in digits.items():
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
        
        recognized_digit = str(np.argmax(scores))
        exp_digits.append(recognized_digit)
    
    # Need at least 4 digits for MM/YY
    if len(exp_digits) >= 4:
        # Take the first 4 consecutive digits that make sense
        # (MM should be 01-12, YY should be reasonable)
        for i in range(len(exp_digits) - 3):
            month = "{}{}".format(exp_digits[i], exp_digits[i+1])
            year = "{}{}".format(exp_digits[i+2], exp_digits[i+3])
            
            # Basic validation
            if 1 <= int(month) <= 12:
                return month, year
        
        # Fallback: just take first 4
        month = "{}{}".format(exp_digits[0], exp_digits[1])
        year = "{}{}".format(exp_digits[2], exp_digits[3])
        return month, year
    
    return "Unknown", "Unknown"


# ============================================================
#  RUN ALL EXTRACTION MODULES
# ============================================================
output_d = for_digits(cnts , image_path)
output_a = for_alphabets(cnts , image_path)
exp_month, exp_year = extract_expiration_date(image_path)     



# ============================================================
#  FINAL OUTPUT
# ============================================================
print()
print('RESULT...................')
print()
print("Card Number           : {}".format("".join(output_d)))
print()
print("Card Type             : {}".format(FIRST_NUMBER.get(output_d[0], "Unknown")))
print()
print('Card Holder Name      : {}'.format(output_a))
print()
print("Expiration Date       : {}/{}".format(exp_month, exp_year))
print()