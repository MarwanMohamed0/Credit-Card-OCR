import cv2
import imutils
from imutils import contours
import numpy as np
import argparse
import pytesseract
import re

# ===================== ARGUMENTS ==========================
ap = argparse.ArgumentParser()
ap.add_argument('-i','--image', required=True, help="Path to input image")
args = vars(ap.parse_args())
image_path = args['image']

font_path_d = 'font_images/OCRA.png'
font_path_a = 'font_images/ocr_a_reference.png'

# ===================== FIND ROI FOR TEMPLATES ==========================
def find_ROI(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts, method='left-to-right')[0]

    sample = {}
    for i, c in enumerate(cnts):
        x, y, w, h = cv2.boundingRect(c)
        roi = thresh[y:y+h, x:x+w]
        roi = cv2.resize(roi, (57,88))
        sample[i] = roi
    return sample

# ===================== PREPROCESS CARD IMAGE ==========================
def preprocessing_find_contours(path):
    rectkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = imutils.resize(gray, width=300)

    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectkernel)
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    gradX = 255 * (gradX - np.min(gradX)) / (np.max(gradX) - np.min(gradX))
    gradX = gradX.astype("uint8")
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectkernel)

    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqkernel)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts, method='left-to-right')[0]
    return cnts

# ===================== LOAD TEMPLATES ==========================
sample = find_ROI(font_path_d)
digits = find_ROI(font_path_a)

char = {  # Letter mapping
    0 : ['A', sample[1]], 1 : ['B', sample[7]], 2 : ['C', sample[12]], 3 : ['D', sample[17]],
    4 : ['E', sample[22]], 5 : ['F', sample[27]], 6 : ['G', sample[32]], 7 : ['H', sample[36]],
    8 : ['I', sample[42]], 9 : ['J', sample[50]], 10 : ['K', sample[54]], 11 : ['L', sample[60]],
    12 : ['M', sample[67]], 13 : ['N', sample[73]], 14 : ['O', sample[82]], 15 : ['P', sample[87]],
    16 : ['Q', sample[0]], 17 : ['R', sample[6]], 18 : ['S', sample[11]], 19 : ['T', sample[16]],
    20 : ['U', sample[21]], 21 : ['V', sample[26]], 22 : ['W', sample[31]], 23 : ['X', sample[35]],
    24 : ['Y', sample[41]], 25 : ['Z', sample[47]],
}

FIRST_NUMBER = {"3": "American Express","4": "Visa","5": "MasterCard","6": "Discover Card"}

cnts = preprocessing_find_contours(image_path)

# ===================== CARD NUMBER EXTRACTION ==========================
def for_digits(cnts, path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = imutils.resize(gray, width=300)
    locs_d = []

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)
        if 2.5 < ar < 4.0 and 40 < w < 55 and 10 < h < 20:
            locs_d.append((x, y, w, h))

    locs_d = sorted(locs_d, key=lambda x: x[0])
    output = []

    for gX, gY, gW, gH in locs_d:
        group = gray[gY-5:gY+gH+5, gX-5:gX+gW+5]
        group = cv2.threshold(group,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
        grpcnts = cv2.findContours(group, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        grpcnts = imutils.grab_contours(grpcnts)
        grpcnts = contours.sort_contours(grpcnts, method='left-to-right')[0]

        for c in grpcnts:
            x, y, w, h = cv2.boundingRect(c)
            roi = cv2.resize(group[y:y+h, x:x+w], (57,88))
            # --- Corrected template matching ---
            scores = [cv2.minMaxLoc(cv2.matchTemplate(roi, digits[d], cv2.TM_CCOEFF))[1] for d in digits]
            output.append(str(np.argmax(scores)))

    return output

# ===================== CARDHOLDER NAME EXTRACTION ==========================
def for_alphabets(cnts, path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = imutils.resize(gray, width=300)

    locs_a = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if 145 < y < gray.shape[0]-8 and x < gray.shape[1]*0.625 and x > 10:
            locs_a.append((x, y, w, h))

    locs_a = sorted(locs_a, key=lambda x: x[0])
    output = ''

    for gX, gY, gW, gH in locs_a:
        group = gray[gY-5:gY+gH+5, gX-5:gX+gW+5]
        group = cv2.threshold(group,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
        grpcnts = cv2.findContours(group, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        grpcnts = imutils.grab_contours(grpcnts)
        grpcnts = contours.sort_contours(grpcnts, method='left-to-right')[0]

        card_name = ''
        for c in grpcnts:
            x, y, w, h = cv2.boundingRect(c)
            roi = cv2.resize(group[y:y+h, x:x+w], (57,88))
            # --- Corrected template matching ---
            scores = [cv2.minMaxLoc(cv2.matchTemplate(roi, char[i][1], cv2.TM_CCOEFF))[1] for i in char]
            card_name += char[np.argmax(scores)][0]
        output += ' ' + card_name

    return output.strip()

# ===================== TEMPLATE MATCHING EXPIRATION DATE ==========================
def extract_expiration_date(image_path, digits_template):
    import cv2
    import imutils
    import numpy as np
    import re

    img = cv2.imread(image_path)
    if img is None:
        return "Unknown", "Unknown"

    img = imutils.resize(img, width=400)
    h, w, _ = img.shape

    # Crop the region where expiration date usually appears
    exp_region = img[int(h*0.65):int(h*0.85), int(w*0.55):int(w*0.95)]
    gray = cv2.cvtColor(exp_region, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Find contours of possible digits
    cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts, method='left-to-right')[0]

    output = []

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w > 10 and h > 10:  # filter tiny noise
            roi = cv2.resize(gray[y:y+h, x:x+w], (57, 88))
            scores = []
            for d in digits_template:
                result = cv2.matchTemplate(roi, digits_template[d], cv2.TM_CCOEFF)
                (_, score, _, _) = cv2.minMaxLoc(result)
                scores.append(score)
            output.append(str(np.argmax(scores)))

    # Combine digits into MM/YY format
    text = "".join(output)
    # Use regex to find MM/YY pattern
    pattern = r"(0[1-9]|1[0-2])(\d{2})"
    match = re.search(pattern, text)
    if match:
        return match.group(1), match.group(2)

    return "Unknown", "Unknown"



# ===================== RUN EXTRACTIONS ==========================
output_d = for_digits(cnts, image_path)
output_a = for_alphabets(cnts, image_path)
exp_month, exp_year = extract_expiration_date(image_path, digits)

# ===================== FINAL OUTPUT ==========================
print("\nRESULT...................\n")
print("Card Number           : {}".format("".join(output_d)))
print("Card Type             : {}".format(FIRST_NUMBER.get(output_d[0], "Unknown")))
print("Card Holder Name      : {}".format(output_a))
print(f"Expiration Date: {exp_month}/{exp_year}")
