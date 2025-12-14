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
    if img is None:
        print(f"[ERROR] Cannot load template: {path}")
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    thresh = cv2.dilate(thresh, kernel, iterations=1)

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
#  PREPROCESSING - USING WORKING METHOD FROM TEST FILE
# ============================================================
def preprocessing_find_contours_working(gray, debug=False):
    """Uses the EXACT same preprocessing as test_card_static.py that works"""
    
    # Resize for consistent processing
    gray = imutils.resize(gray, width=600)
    
    if debug:
        cv2.imshow('1. Resized Gray', gray)
    
    # Bilateral filter
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    if debug:
        cv2.imshow('2. Bilateral Filter', gray)
    
    # Morphological kernels
    rectkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    # Top-hat
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectkernel)
    if debug:
        cv2.imshow('3. Tophat', tophat)
    
    # Edge detection using Canny (THIS IS THE KEY DIFFERENCE!)
    edges = cv2.Canny(tophat, 50, 150)
    if debug:
        cv2.imshow('4. Canny Edges', edges)

    # Strengthen edges
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, rectkernel)
    if debug:
        cv2.imshow('5. Morph Close', edges)

    # Threshold
    thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    if debug:
        cv2.imshow('6. Threshold', thresh)
    
    # Additional closing
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqkernel)
    if debug:
        cv2.imshow('7. Final Threshold', thresh)
    
    # Find contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # Visualize all contours
    if debug:
        contour_img = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_img, cnts, -1, (0, 255, 0), 2)
        cv2.imshow('8. All Contours', contour_img)
    
    if len(cnts) > 0:
        cnts = contours.sort_contours(cnts, method='left-to-right')[0]
    
    return cnts, gray, thresh


# ============================================================
#  EXTRACT CARD NUMBER - USING WORKING PARAMETERS
# ============================================================
def extract_card_number_working(cnts, gray, digits, debug=False):
    """Uses the EXACT same parameters as test_card_static.py"""
    
    locs_d = []
    
    # Visualize digit group detection
    if debug:
        debug_img = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
    
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        
        if debug:
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 0), 1)
        
        # EXACT same parameters as working test file
        if ar > 2.0 and ar < 5.0:
            if (w > 35 and w < 65) and (h > 8 and h < 25):
                locs_d.append((x, y, w, h))
                if debug:
                    cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(debug_img, f"AR:{ar:.2f}", (x, y-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    
    if debug:
        cv2.imshow('9. Digit Groups Detection', debug_img)
        print(f"[DEBUG] Found {len(locs_d)} potential digit groups")
    
    if len(locs_d) == 0:
        return []
    
    locs_d = sorted(locs_d, key=lambda x: x[0])
    output = []
    
    for (idx, (gX, gY, gW, gH)) in enumerate(locs_d):
        if debug:
            print(f"[DEBUG] Processing digit group {idx+1}/{len(locs_d)}")
            print(f"  Position: ({gX}, {gY}), Size: {gW}x{gH}")
        
        # Extract group
        group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
        group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        if debug:
            cv2.imshow(f'Group {idx+1}', imutils.resize(group, width=200))
        
        # Find individual digits
        grpcnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        grpcnts = imutils.grab_contours(grpcnts)
        
        if len(grpcnts) > 0:
            grpcnts = contours.sort_contours(grpcnts, method='left-to-right')[0]
            
            if debug:
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
                        if debug:
                            print(f"    Matched digit: {best_match} (score: {best_score:.2f})")
    
    return output


# ============================================================
#  EXTRACT CARDHOLDER NAME - USING WORKING PARAMETERS
# ============================================================
def extract_cardholder_name_working(cnts, gray, char, debug=False):
    """Extract cardholder name using same approach"""
    
    locs_a = []
    h_img, w_img = gray.shape
    
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        
        # Look in lower portion (adjust based on card layout)
        if y > 140 and y < (h_img - 8) and x < (w_img * 5 / 8) and x > 10:
            locs_a.append((x, y, w, h))
    
    if debug:
        print(f"[DEBUG] Found {len(locs_a)} potential name regions")
    
    if len(locs_a) == 0:
        return ""
    
    locs_a = sorted(locs_a, key=lambda x: x[0])
    output = ''
    
    for (i, (gX, gY, gW, gH)) in enumerate(locs_a):
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
                output = output + " " + card_name
    
    return output.strip()


# ============================================================
#  CROP CARD REGION FROM FRAME
# ============================================================
def crop_card_region(frame, margin=80):
    """
    Crop the center region of the frame where card should be placed.
    This improves alignment and reduces noise from background.
    """
    h, w = frame.shape[:2]
    
    # Define crop region (center area with margin)
    x1 = margin
    y1 = margin
    x2 = w - margin
    y2 = h - margin
    
    # Crop the region
    cropped = frame[y1:y2, x1:x2]
    
    return cropped, (x1, y1, x2, y2)


# ============================================================
#  PROCESS SINGLE FRAME
# ============================================================
def process_frame_working(frame, digits, char, debug=False, crop_card=True):
    """Process frame using the working method from test file"""
    
    # Option to crop card region first
    if crop_card:
        frame_to_process, crop_coords = crop_card_region(frame, margin=100)
        if debug:
            cv2.imshow('Cropped Card Region', imutils.resize(frame_to_process, width=600))
    else:
        frame_to_process = frame
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2GRAY)
    
    try:
        # Use the working preprocessing method
        cnts, processed_gray, thresh = preprocessing_find_contours_working(gray, debug)
        
        # Extract card number using working method
        card_number_list = extract_card_number_working(cnts, processed_gray, digits, debug)
        card_number = "".join(card_number_list)
        
        # Extract cardholder name
        cardholder_name = extract_cardholder_name_working(cnts, processed_gray, char, debug)
        
        # Determine card type
        card_type = "Unknown"
        if card_number_list and len(card_number_list) > 0:
            card_type = FIRST_NUMBER.get(card_number_list[0], "Unknown")
        
        return card_number, card_type, cardholder_name, len(cnts)
    
    except Exception as e:
        print(f"[ERROR] Processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return "", "Unknown", "", 0


# ============================================================
#  MAIN WEBCAM FUNCTION
# ============================================================
def run_webcam_ocr_working():
    """Run webcam OCR using the WORKING method from test file"""
    
    print("\n" + "="*60)
    print("CREDIT CARD OCR - USING WORKING TEST METHOD")
    print("="*60)
    print("Loading templates...")
    
    # Load templates
    try:
        sample = find_ROI(font_path_d)
        if sample is None:
            return
        print(f"[SUCCESS] Loaded {len(sample)} letter templates")
    except Exception as e:
        print(f"[ERROR] Failed to load letter templates: {str(e)}")
        return
    
    try:
        digits = find_ROI(font_path_a)
        if digits is None:
            return
        print(f"[SUCCESS] Loaded {len(digits)} digit templates")
    except Exception as e:
        print(f"[ERROR] Failed to load digit templates: {str(e)}")
        return
    
    # Letter template mapping (same as before)
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
    
    print("\n✓ USING EXACT METHOD FROM test_card_static.py")
    print("✓ Canny edge detection enabled")
    print("✓ Card region cropping enabled")
    print("✓ Same detection parameters (ar: 2-5, w: 35-65, h: 8-25)")
    
    print("\nInstructions:")
    print("- Hold card HORIZONTALLY within green rectangle")
    print("- Card should fill most of the green area")
    print("- Ensure good, even lighting")
    print("- Press 'c' to CAPTURE and process")
    print("- Press 'd' to toggle DEBUG mode")
    print("- Press 'r' to toggle CROP mode (ON by default)")
    print("- Press 's' to SAVE frame as test_card.jpg")
    print("- Press 'q' to QUIT")
    print("="*60 + "\n")
    
    # Initialize webcam
    cap = None
    for camera_index in [0, 1]:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"[SUCCESS] Camera {camera_index} opened!")
            break
    
    if cap is None or not cap.isOpened():
        print("[ERROR] Could not open webcam!")
        return
    
    # Camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    
    last_card_number = ""
    last_card_type = ""
    last_cardholder = ""
    debug_mode = False
    crop_mode = True  # Enabled by default
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("[ERROR] Failed to grab frame")
            break
        
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # Draw card placement guide (larger for better visibility)
        margin = 100
        cv2.rectangle(display_frame, (margin, margin), (w-margin, h-margin), (0, 255, 0), 3)
        
        # Instructions overlay
        cv2.putText(display_frame, "Place card within green rectangle - FILL THE AREA", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "C=Capture | D=Debug | R=Crop | S=Save | Q=Quit", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show status
        status_text = f"Crop: {'ON' if crop_mode else 'OFF'} | Debug: {'ON' if debug_mode else 'OFF'}"
        cv2.putText(display_frame, status_text, 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Display last results
        y_offset = 130
        if last_card_number:
            cv2.putText(display_frame, f"Number: {last_card_number}", 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y_offset += 40
        if last_card_type:
            cv2.putText(display_frame, f"Type: {last_card_type}", 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y_offset += 40
        if last_cardholder:
            cv2.putText(display_frame, f"Name: {last_cardholder}", 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.imshow('Credit Card OCR - Working Method', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nExiting...")
            break
        
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
        
        elif key == ord('r'):
            crop_mode = not crop_mode
            print(f"Crop mode: {'ON' if crop_mode else 'OFF'}")
        
        elif key == ord('s'):
            cv2.imwrite('test_card.jpg', frame)
            print("\n[SAVED] Frame saved as 'test_card.jpg'")
            print("You can test this with: python test_card_static.py")
        
        elif key == ord('c'):
            print("\n[PROCESSING] Capturing and analyzing card...")
            print(f"Crop mode: {'ON' if crop_mode else 'OFF'}")
            
            card_number, card_type, cardholder_name, num_contours = process_frame_working(
                frame, digits, char, debug_mode, crop_card=crop_mode
            )
            
            print("\n" + "="*60)
            print("EXTRACTION RESULTS")
            print("="*60)
            print(f"Card Number      : {card_number if card_number else 'Not detected'}")
            print(f"Card Type        : {card_type}")
            print(f"Cardholder Name  : {cardholder_name if cardholder_name else 'Not detected'}")
            print(f"Contours found   : {num_contours}")
            print("="*60)
            
            if not card_number:
                print("\n[TIPS]:")
                print("  • Make sure card fills the green rectangle")
                print("  • Improve lighting (bright, even, no shadows)")
                print("  • Try toggling crop mode (press 'r')")
                print("  • Enable debug mode (press 'd') to see processing steps")
                print("  • Save frame (press 's') and compare with test output")
            print()
            
            last_card_number = card_number if card_number else "Not detected"
            last_card_type = card_type
            last_cardholder = cardholder_name if cardholder_name else "Not detected"
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nWebcam closed successfully!")


if __name__ == "__main__":
    run_webcam_ocr_working()