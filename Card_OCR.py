import cv2
import imutils
from imutils import contours
import numpy as np


# ============================================================
#  CONFIGURATION
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
def preprocessing_find_contours(gray):
    rectkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

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

    return cnts, gray


# ============================================================
#  EXTRACT & RECOGNIZE DIGITS (CARD NUMBER)
# ============================================================
def extract_card_number(cnts, gray, digits):
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

            output.append(str(np.argmax(scores)))

    return output


# ============================================================
#  EXTRACT & RECOGNIZE CARD HOLDER NAME (LETTERS)
# ============================================================
def extract_cardholder_name(cnts, gray, char):
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

    return output.strip()


# ============================================================
#  PROCESS SINGLE FRAME
# ============================================================
def process_frame(frame, digits, char):
    """Process a single frame and extract card information"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    try:
        cnts, processed_gray = preprocessing_find_contours(gray)
        
        # Extract card number
        card_number_list = extract_card_number(cnts, processed_gray, digits)
        card_number = "".join(card_number_list)
        
        # Extract cardholder name
        cardholder_name = extract_cardholder_name(cnts, processed_gray, char)
        
        # Determine card type
        card_type = "Unknown"
        if card_number_list:
            card_type = FIRST_NUMBER.get(card_number_list[0], "Unknown")
        
        return card_number, card_type, cardholder_name
    
    except Exception as e:
        return "", "Unknown", ""


# ============================================================
#  MAIN WEBCAM FUNCTION
# ============================================================
def run_webcam_ocr():
    """Run real-time credit card OCR using webcam"""
    
    print("\n" + "="*60)
    print("REAL-TIME CREDIT CARD OCR")
    print("="*60)
    print("Loading templates...")
    
    # Load templates
    sample = find_ROI(font_path_d)
    digits = find_ROI(font_path_a)
    
    # Letter template mapping
    char = {
        0: ['A', sample[1]], 1: ['B', sample[7]], 2: ['C', sample[12]], 3: ['D', sample[17]],
        4: ['E', sample[22]], 5: ['F', sample[27]], 6: ['G', sample[32]], 7: ['H', sample[36]],
        8: ['I', sample[42]], 9: ['J', sample[50]], 10: ['K', sample[54]], 11: ['L', sample[60]],
        12: ['M', sample[67]], 13: ['N', sample[73]], 14: ['O', sample[82]], 15: ['P', sample[87]],
        16: ['Q', sample[0]], 17: ['R', sample[6]], 18: ['S', sample[11]], 19: ['T', sample[16]],
        20: ['U', sample[21]], 21: ['V', sample[26]], 22: ['W', sample[31]], 23: ['X', sample[35]],
        24: ['Y', sample[41]], 25: ['Z', sample[47]], 26: ['A', sample[3]], 27: ['B', sample[9]],
        28: ['C', sample[14]], 29: ['D', sample[19]], 30: ['E', sample[24]], 31: ['F', sample[29]],
        32: ['G', sample[34]], 33: ['H', sample[38]], 34: ['I', sample[44]], 35: ['J', sample[49]],
        36: ['K', sample[56]], 37: ['L', sample[62]], 38: ['M', sample[68]], 39: ['N', sample[74]],
        40: ['O', sample[83]], 41: ['P', sample[88]], 42: ['Q', sample[2]], 43: ['R', sample[8]],
        44: ['S', sample[13]], 45: ['T', sample[18]], 46: ['U', sample[23]], 47: ['V', sample[28]],
        48: ['W', sample[33]], 49: ['X', sample[37]], 50: ['Y', sample[43]], 51: ['Z', sample[48]],
    }
    
    print("Templates loaded successfully!")
    print("\nInstructions:")
    print("- Hold your credit card in front of the webcam")
    print("- Press 'c' to CAPTURE and process the card")
    print("- Press 'q' to QUIT")
    print("="*60 + "\n")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Could not open webcam!")
        return
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    last_card_number = ""
    last_card_type = ""
    last_cardholder = ""
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("[ERROR] Failed to grab frame")
            break
        
        # Create display frame
        display_frame = frame.copy()
        
        # Add instructions overlay
        cv2.putText(display_frame, "Press 'C' to Capture | 'Q' to Quit", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display last captured info
        y_offset = 60
        if last_card_number:
            cv2.putText(display_frame, f"Card Number: {last_card_number}", 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 30
        if last_card_type:
            cv2.putText(display_frame, f"Card Type: {last_card_type}", 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 30
        if last_cardholder:
            cv2.putText(display_frame, f"Cardholder: {last_cardholder}", 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Credit Card OCR - Webcam', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nExiting...")
            break
        
        elif key == ord('c'):
            print("\n[PROCESSING] Capturing and analyzing card...")
            
            # Process the current frame
            card_number, card_type, cardholder_name = process_frame(frame, digits, char)
            
            # Display results
            print("\n" + "="*60)
            print("EXTRACTION RESULTS")
            print("="*60)
            print(f"Card Number      : {card_number if card_number else 'Not detected'}")
            print(f"Card Type        : {card_type}")
            print(f"Cardholder Name  : {cardholder_name if cardholder_name else 'Not detected'}")
            print("="*60 + "\n")
            
            # Store for display
            last_card_number = card_number if card_number else "Not detected"
            last_card_type = card_type
            last_cardholder = cardholder_name if cardholder_name else "Not detected"
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nWebcam closed successfully!")


# ============================================================
#  ENTRY POINT
# ============================================================
if __name__ == "__main__":
    run_webcam_ocr()