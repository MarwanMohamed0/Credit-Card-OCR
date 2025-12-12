import cv2
import imutils
from imutils import contours
import numpy as np

# ============================================================
#  TEST SCRIPT FOR SAVED CARD IMAGE
# ============================================================

font_path_d = 'font_images/OCRA.png'
font_path_a = 'font_images/ocr_a_reference.png'

FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}


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


def preprocessing_find_contours(gray):
    gray = imutils.resize(gray, width=300)
    
    # Show resized image
    cv2.imshow('1. Resized Gray', gray)
    
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    cv2.imshow('2. Bilateral Filter', gray)
    
    rectkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectkernel)
    cv2.imshow('3. Tophat', tophat)
    
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    
    if maxVal - minVal > 0:
        gradX = 255 * (gradX - minVal) / (maxVal - minVal)
    gradX = gradX.astype('uint8')
    
    cv2.imshow('4. Sobel', gradX)
    
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectkernel)
    cv2.imshow('5. Morph Close', gradX)
    
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow('6. Threshold', thresh)
    
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqkernel)
    cv2.imshow('7. Final Threshold', thresh)
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # Draw all contours on a copy
    contour_img = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, cnts, -1, (0, 255, 0), 2)
    cv2.imshow('8. All Contours', contour_img)
    
    if len(cnts) > 0:
        cnts = contours.sort_contours(cnts, method='left-to-right')[0]
    
    return cnts, gray, thresh


def extract_card_number_debug(cnts, gray, digits):
    locs_d = []
    
    # Visualize digit group detection
    debug_img = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
    
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        
        # Draw all contours in blue
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 0), 1)
        
        # Check if it matches digit group criteria
        if ar > 2.0 and ar < 5.0:
            if (w > 35 and w < 65) and (h > 8 and h < 25):
                locs_d.append((x, y, w, h))
                # Draw matching groups in green
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(debug_img, f"AR:{ar:.2f}", (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    
    cv2.imshow('9. Digit Groups Detection', debug_img)
    print(f"\n[DEBUG] Found {len(locs_d)} potential digit groups")
    
    if len(locs_d) == 0:
        print("[WARNING] No digit groups detected!")
        print("Try adjusting these parameters in extract_card_number:")
        print("  - ar > 2.0 and ar < 5.0  (aspect ratio)")
        print("  - w > 35 and w < 65  (width)")
        print("  - h > 8 and h < 25  (height)")
        return []
    
    locs_d = sorted(locs_d, key=lambda x: x[0])
    output = []
    
    for (idx, (gX, gY, gW, gH)) in enumerate(locs_d):
        print(f"\n[DEBUG] Processing digit group {idx+1}/{len(locs_d)}")
        print(f"  Position: ({gX}, {gY}), Size: {gW}x{gH}")
        
        group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
        group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        cv2.imshow(f'Group {idx+1}', imutils.resize(group, width=200))
        
        grpcnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        grpcnts = imutils.grab_contours(grpcnts)
        
        if len(grpcnts) > 0:
            grpcnts = contours.sort_contours(grpcnts, method='left-to-right')[0]
            print(f"  Found {len(grpcnts)} individual digits")
            
            for c in grpcnts:
                (x, y, w, h) = cv2.boundingRect(c)
                if w > 5 and h > 5:
                    roi = group[y:y + h, x:x + w]
                    roi = cv2.resize(roi, (57, 88))
                    scores = []
                    
                    for (digit, digitROI) in digits.items():
                        result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
                        (_, score, _, _) = cv2.minMaxLoc(result)
                        scores.append(score)
                    
                    if len(scores) > 0:
                        best_match = np.argmax(scores)
                        best_score = scores[best_match]
                        output.append(str(best_match))
                        print(f"    Matched digit: {best_match} (score: {best_score:.2f})")
        else:
            print("  [WARNING] No contours found in this group")
    
    return output


# ============================================================
#  MAIN TEST
# ============================================================
print("\n" + "="*60)
print("TESTING SAVED CARD IMAGE")
print("="*60)

# Load templates
print("\nLoading templates...")
sample = find_ROI(font_path_d)
if sample is None:
    print("Failed to load letter templates!")
    exit()
print(f"Loaded {len(sample)} letter templates")

digits = find_ROI(font_path_a)
if digits is None:
    print("Failed to load digit templates!")
    exit()
print(f"Loaded {len(digits)} digit templates")

# Load test image
test_image = 'test_card.jpg'
img = cv2.imread(test_image)

if img is None:
    print(f"\n[ERROR] Could not load {test_image}")
    print("Make sure you've saved a frame by pressing 's' in the webcam app")
    exit()

print(f"\n[SUCCESS] Loaded {test_image}")
print(f"Image size: {img.shape[1]}x{img.shape[0]}")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Show original
cv2.imshow('0. Original', imutils.resize(img, width=600))

# Preprocess
print("\nPreprocessing...")
cnts, processed_gray, thresh = preprocessing_find_contours(gray)

print(f"\nFound {len(cnts)} total contours")

# Extract digits
print("\n" + "="*60)
print("EXTRACTING CARD NUMBER")
print("="*60)
card_number_list = extract_card_number_debug(cnts, processed_gray, digits)
card_number = "".join(card_number_list)

# Results
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"Card Number: {card_number if card_number else 'NOT DETECTED'}")
if card_number_list:
    card_type = FIRST_NUMBER.get(card_number_list[0], "Unknown")
    print(f"Card Type: {card_type}")

print("\n" + "="*60)
print("Press any key to close all windows...")
print("="*60)

cv2.waitKey(0)
cv2.destroyAllWindows()