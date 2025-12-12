import cv2
import imutils
from imutils import contours
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
font_path_d = 'font_images/OCRA.png'
font_path_a = 'font_images/ocr_a_reference.png'

FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}

# ============================================================
# LOAD LETTER/DIGIT TEMPLATES
# ============================================================
def find_ROI(path):
    img = cv2.imread(path)
    if img is None:
        print(f"[ERROR] Cannot load template: {path}")
        return None
    
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
# PREPROCESS IMAGE & FIND CONTOURS
# ============================================================
def preprocessing_find_contours(gray):
    gray = imutils.resize(gray, width=300)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    rectkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectkernel)

    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    if maxVal - minVal > 0:
        gradX = 255 * (gradX - minVal) / (maxVal - minVal)
    gradX = gradX.astype('uint8')

    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectkernel)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqkernel)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) > 0:
        cnts = contours.sort_contours(cnts, method='left-to-right')[0]

    return cnts, gray, thresh

# ============================================================
# EXTRACT CARD NUMBER
# ============================================================
def extract_card_number(cnts, gray, digits):
    locs_d = []
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if 2.0 < ar < 5.0 and 35 < w < 65 and 8 < h < 25:
            locs_d.append((x, y, w, h))
    if len(locs_d) == 0:
        return []

    locs_d = sorted(locs_d, key=lambda x: x[0])
    output = []

    for (gX, gY, gW, gH) in locs_d:
        group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
        group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        grpcnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        grpcnts = imutils.grab_contours(grpcnts)
        if len(grpcnts) > 0:
            grpcnts = contours.sort_contours(grpcnts, method='left-to-right')[0]
            for c in grpcnts:
                (x, y, w, h) = cv2.boundingRect(c)
                if w > 5 and h > 5:
                    roi = group[y:y + h, x:x + w]
                    roi = cv2.resize(roi, (57, 88))
                    scores = []
                    for digit, digitROI in digits.items():
                        result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
                        (_, score, _, _) = cv2.minMaxLoc(result)
                        scores.append(score)
                    if len(scores) > 0:
                        output.append(str(np.argmax(scores)))
    return output

# ============================================================
# EXTRACT CARDHOLDER NAME
# ============================================================
def extract_cardholder_name(cnts, gray, char):
    locs_a = []
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        if 140 < y < gray.shape[0] - 8 and x < gray.shape[1]*5/8 and x > 10:
            locs_a.append((x, y, w, h))
    if len(locs_a) == 0:
        return ""

    locs_a = sorted(locs_a, key=lambda x: x[0])
    output = ''
    for (gX, gY, gW, gH) in locs_a:
        group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
        group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        grpcnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        grpcnts = imutils.grab_contours(grpcnts)
        if len(grpcnts) > 0:
            grpcnts = contours.sort_contours(grpcnts, method='left-to-right')[0]
            card_name = ''
            for c in grpcnts:
                (x, y, w, h) = cv2.boundingRect(c)
                if w > 5 and h > 5:
                    roi = group[y:y + h, x:x + w]
                    roi = cv2.resize(roi, (57, 88))
                    scores = []
                    for j in range(len(char)):
                        result = cv2.matchTemplate(roi, char[j][1], cv2.TM_CCOEFF)
                        (_, score, _, _) = cv2.minMaxLoc(result)
                        scores.append(score)
                    if len(scores) > 0:
                        index_max_score = np.argmax(scores)
                        card_name = card_name + char[index_max_score][0]
            if card_name:
                output += " " + card_name
    return output.strip()

# ============================================================
# PROCESS FRAME
# ============================================================
def process_frame(frame, digits, char, debug=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try:
        cnts, processed_gray, thresh = preprocessing_find_contours(gray)
        if debug:
            cv2.imshow('Preprocessed', imutils.resize(processed_gray, width=600))
            cv2.imshow('Threshold', imutils.resize(thresh, width=600))
        card_number_list = extract_card_number(cnts, processed_gray, digits)
        card_number = "".join(card_number_list)
        cardholder_name = extract_cardholder_name(cnts, processed_gray, char)
        card_type = FIRST_NUMBER.get(card_number_list[0], "Unknown") if card_number_list else "Unknown"
        return card_number, card_type, cardholder_name, len(cnts)
    except Exception as e:
        print(f"[ERROR] Processing failed: {str(e)}")
        return "", "Unknown", "", 0

# ============================================================
# WEBCAM LOOP
# ============================================================
def run_webcam_ocr():
    print("\n" + "="*60)
    print("REAL-TIME CREDIT CARD OCR")
    print("="*60)
    print("Loading templates...")

    # Load templates
    sample = find_ROI(font_path_d)
    digits = find_ROI(font_path_a)
    char = {i: [chr(65+i), sample[i]] for i in range(len(sample))}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam!")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    last_card_number, last_card_type, last_cardholder = "", "", ""
    debug_mode = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        margin = 100
        cv2.rectangle(display_frame, (margin, margin), (w-margin, h-margin), (0, 255, 0), 2)
        cv2.putText(display_frame, "C=Capture | D=Debug | S=Save | Q=Quit", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        y_offset = 70
        if last_card_number:
            cv2.putText(display_frame, f"Number: {last_card_number}", (10,y_offset), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
            y_offset += 35
        if last_card_type:
            cv2.putText(display_frame, f"Type: {last_card_type}", (10,y_offset), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
            y_offset += 35
        if last_cardholder:
            cv2.putText(display_frame, f"Name: {last_cardholder}", (10,y_offset), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
        if debug_mode:
            cv2.putText(display_frame, "DEBUG MODE ON", (w-200,30), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
        cv2.imshow('Credit Card OCR - Webcam', display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
        elif key == ord('s'):
            cv2.imwrite('test_card.jpg', frame)
            print("[SAVED] Frame saved as 'test_card.jpg'")
        elif key == ord('c'):
            print("[PROCESSING] Capturing and analyzing card...")
            card_number, card_type, cardholder_name, num_contours = process_frame(frame, digits, char, debug_mode)
            print(f"Contours found: {num_contours}")
            print(f"Card Number: {card_number}")
            print(f"Card Type: {card_type}")
            print(f"Cardholder: {cardholder_name}")
            last_card_number = card_number or "Not detected"
            last_card_type = card_type
            last_cardholder = cardholder_name or "Not detected"

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed successfully!")

if __name__ == "__main__":
    run_webcam_ocr()
