import cv2
import imutils
from imutils import contours
import numpy as np
import time

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

# Card dimensions for capture window (9cm x 5cm ratio = 1.8:1)
CARD_ASPECT_RATIO = 9.0 / 5.0  # length/width


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
#  IMPROVED TEMPLATE MATCHING - MULTI-METHOD WITH VOTING
# ============================================================
def match_digit_robust(roi, digit_templates):
    """
    Match digit using MULTIPLE matching methods and voting system.
    This is more robust to font variations than single-method matching.
    """
    
    # Resize ROI to standard size
    roi = cv2.resize(roi, (57, 88))
    
    # Create multiple preprocessing variations of the ROI
    # This helps match even if brightness/contrast differs
    roi_variants = []
    
    # Variant 1: Original
    roi_variants.append(roi.copy())
    
    # Variant 2: Inverted (white text on black vs black on white)
    roi_variants.append(cv2.bitwise_not(roi))
    
    # Variant 3: Hard binary threshold
    _, binary = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)
    roi_variants.append(binary)
    
    # Variant 4: Otsu threshold (adaptive)
    _, otsu = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    roi_variants.append(otsu)
    
    # Variant 5: Inverted Otsu
    roi_variants.append(cv2.bitwise_not(otsu))
    
    # Use multiple template matching methods
    methods = [
        cv2.TM_CCOEFF_NORMED,    # Correlation coefficient (normalized)
        cv2.TM_CCORR_NORMED,     # Cross-correlation (normalized)
        cv2.TM_SQDIFF_NORMED,    # Squared difference (normalized)
    ]
    
    # Voting system: each method votes for a digit
    votes = {i: 0 for i in range(10)}
    all_scores = {i: [] for i in range(10)}
    
    # Try each ROI variant with each matching method
    for roi_var in roi_variants:
        for method in methods:
            
            scores_this_round = []
            
            # Compare against all digit templates
            for digit, template in digit_templates.items():
                
                # Ensure template is correct size
                template = cv2.resize(template, (57, 88))
                
                try:
                    # Perform template matching
                    result = cv2.matchTemplate(roi_var, template, method)
                    score = result[0][0]
                    
                    scores_this_round.append((digit, score))
                    all_scores[digit].append(score)
                
                except:
                    continue
            
            # Find best match for this method
            if scores_this_round:
                # For SQDIFF, LOWER score is better
                if method == cv2.TM_SQDIFF_NORMED:
                    best_digit = min(scores_this_round, key=lambda x: x[1])[0]
                else:
                    # For other methods, HIGHER score is better
                    best_digit = max(scores_this_round, key=lambda x: x[1])[0]
                
                # This method votes for best_digit
                votes[best_digit] += 1
    
    # Find digit with most votes
    if sum(votes.values()) > 0:
        best_digit = max(votes, key=votes.get)
        confidence = votes[best_digit] / sum(votes.values())
    else:
        best_digit = 0
        confidence = 0.0
    
    # Calculate average scores for additional info
    avg_scores = {}
    for digit in range(10):
        if all_scores[digit]:
            avg_scores[digit] = np.mean(all_scores[digit])
        else:
            avg_scores[digit] = 0.0
    
    return best_digit, confidence, avg_scores


# ============================================================
#  QUICK DETECTION - CHECK IF 4 GROUPS PRESENT
# ============================================================
def quick_detect_groups(frame, rect_coords):
    """Quick check if 4 digit groups are visible in frame"""
    try:
        x1, y1, x2, y2 = rect_coords
        
        # Crop to card area
        card_region = frame[y1:y2, x1:x2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(card_region, cv2.COLOR_BGR2GRAY)
        gray = imutils.resize(gray, width=600)
        
        # Quick preprocessing
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        rectkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectkernel)
        edges = cv2.Canny(tophat, 50, 150)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, rectkernel)
        thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # Find contours
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        # Count digit groups
        group_count = 0
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            
            if ar > 2.0 and ar < 5.0:
                if (w > 35 and w < 65) and (h > 8 and h < 25):
                    group_count += 1
        
        return group_count
    except:
        return 0


# ============================================================
#  PREPROCESSING - SAME AS WORKING TEST FILE
# ============================================================
def preprocessing_find_contours(gray):
    gray = imutils.resize(gray, width=600)
    
    # Show resized image
    cv2.imshow('1. Resized Gray', gray)
    
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    cv2.imshow('2. Bilateral Filter', gray)
    
    rectkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectkernel)
    cv2.imshow('3. Tophat', tophat)
    
    # Edge detection using Canny
    edges = cv2.Canny(tophat, 50, 150)
    cv2.imshow('4. Canny Edges', edges)

    # Strengthen edges
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, rectkernel)
    cv2.imshow('5. Morph Close', edges)

    thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow('6. Threshold', thresh)
    
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqkernel)
    cv2.imshow('7. Final Threshold', thresh)
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # Draw all contours
    contour_img = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, cnts, -1, (0, 255, 0), 2)
    cv2.imshow('8. All Contours', contour_img)
    
    if len(cnts) > 0:
        cnts = contours.sort_contours(cnts, method='left-to-right')[0]
    
    return cnts, gray, thresh


# ============================================================
#  EXTRACT CARD NUMBER WITH IMPROVED MATCHING
# ============================================================
def extract_card_number_improved(cnts, gray, digits):
    """
    Extract card number using IMPROVED multi-method template matching
    """
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
        return [], []
    
    locs_d = sorted(locs_d, key=lambda x: x[0])
    output = []
    group_images = []
    
    for (idx, (gX, gY, gW, gH)) in enumerate(locs_d):
        print(f"\n[DEBUG] Processing digit group {idx+1}/{len(locs_d)}")
        print(f"  Position: ({gX}, {gY}), Size: {gW}x{gH}")
        
        group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
        group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # Store the group image
        group_images.append(group.copy())
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
                    
                    # USE IMPROVED MATCHING HERE!
                    best_digit, confidence, avg_scores = match_digit_robust(roi, digits)
                    
                    # Accept if confidence is reasonable
                    if confidence > 0.25:  # At least 25% voting agreement
                        output.append(str(best_digit))
                        print(f"    ✓ Matched digit: {best_digit} (confidence: {confidence:.1%}, votes won)")
                    else:
                        # Low confidence - still add but warn
                        output.append(str(best_digit))
                        print(f"    ⚠ Low confidence: {best_digit} ({confidence:.1%}) - may be wrong")
        else:
            print("  [WARNING] No contours found in this group")
    
    return output, group_images


# ============================================================
#  EXTRACT CARDHOLDER NAME WITH IMPROVED MATCHING
# ============================================================
def extract_cardholder_name_improved(cnts, gray, char):
    """Extract cardholder name using improved matching"""
    
    locs_a = []
    h_img, w_img = gray.shape
    
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        
        # Look in lower portion
        if y > 140 and y < (h_img - 8) and x < (w_img * 5 / 8) and x > 10:
            locs_a.append((x, y, w, h))
    
    print(f"\n[DEBUG] Found {len(locs_a)} potential name regions")
    
    if len(locs_a) == 0:
        return "", []
    
    locs_a = sorted(locs_a, key=lambda x: x[0])
    output = ''
    name_images = []
    
    # Convert char list to dictionary format for robust matching
    char_dict = {i: char[i][1] for i in range(len(char))}
    char_labels = {i: char[i][0] for i in range(len(char))}
    
    for (i, (gX, gY, gW, gH)) in enumerate(locs_a):
        group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
        group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # Store name group image
        name_images.append(group.copy())
        
        grpcnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        grpcnts = imutils.grab_contours(grpcnts)
        
        if len(grpcnts) > 0:
            grpcnts = contours.sort_contours(grpcnts, method='left-to-right')[0]
            
            card_name = ''
            for c in grpcnts:
                (x, y, w, h) = cv2.boundingRect(c)
                if w > 5 and h > 5:
                    roi = group[y:y + h, x:x + w]
                    
                    # Use improved matching for letters too
                    best_char_idx, confidence, _ = match_digit_robust(roi, char_dict)
                    
                    if confidence > 0.20:  # Lower threshold for letters (more variation)
                        card_name += char_labels[best_char_idx]
            
            if card_name:
                output = output + " " + card_name
                print(f"[DEBUG] Name group {i+1}: {card_name}")
    
    return output.strip(), name_images


# ============================================================
#  PROCESS CAPTURED IMAGE
# ============================================================
def process_captured_image(img, digits, char):
    """Process captured image with IMPROVED matching"""
    
    print("\n" + "="*60)
    print("PROCESSING CAPTURED IMAGE")
    print("="*60)
    print("Using IMPROVED MULTI-METHOD TEMPLATE MATCHING")
    print("="*60)
    
    # Show original
    cv2.imshow('0. Original', imutils.resize(img, width=600))
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Preprocess
    print("\nPreprocessing...")
    cnts, processed_gray, thresh = preprocessing_find_contours(gray)
    print(f"Found {len(cnts)} total contours")
    
    # Extract digits with IMPROVED matching
    print("\n" + "="*60)
    print("EXTRACTING CARD NUMBER (IMPROVED METHOD)")
    print("="*60)
    card_number_list, digit_groups = extract_card_number_improved(cnts, processed_gray, digits)
    
    # Extract name with IMPROVED matching
    print("\n" + "="*60)
    print("EXTRACTING CARDHOLDER NAME (IMPROVED METHOD)")
    print("="*60)
    cardholder_name, name_groups = extract_cardholder_name_improved(cnts, processed_gray, char)
    
    # Combine results
    card_number = "".join(card_number_list)
    
    # Determine card type
    card_type = "Unknown"
    if card_number_list and len(card_number_list) > 0:
        card_type = FIRST_NUMBER.get(card_number_list[0], "Unknown")
    
    # FINAL RESULTS
    print("\n" + "="*60)
    print("FINAL EXTRACTED TEXT")
    print("="*60)
    print(f"Card Number: {card_number if card_number else 'NOT DETECTED'}")
    print(f"Card Type: {card_type}")
    print(f"Cardholder Name: {cardholder_name if cardholder_name else 'NOT DETECTED'}")
    print("="*60)
    
    return card_number, card_type, cardholder_name


# ============================================================
#  WEBCAM AUTO-CAPTURE WHEN 4 GROUPS DETECTED
# ============================================================
def run_webcam_auto_capture():
    """Auto-capture when 4 digit groups are detected"""
    
    print("\n" + "="*60)
    print("CARD OCR - AUTO-CAPTURE WITH IMPROVED MATCHING")
    print("="*60)
    
    # Load templates
    print("\nLoading templates...")
    sample = find_ROI(font_path_d)
    if sample is None:
        print("Failed to load letter templates!")
        return
    print(f"Loaded {len(sample)} letter templates")

    digits = find_ROI(font_path_a)
    if digits is None:
        print("Failed to load digit templates!")
        return
    print(f"Loaded {len(digits)} digit templates")
    
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
    
    print("\n" + "="*60)
    print("AUTO-CAPTURE MODE WITH IMPROVED MATCHING")
    print("="*60)
    print("✓ Multi-method template matching enabled")
    print("✓ Voting system for better accuracy")
    print("✓ Multiple preprocessing variants")
    print("\nCard dimensions: 9cm x 5cm (ratio 1.8:1)")
    print("\nInstructions:")
    print("- Position card in GREEN RECTANGLE")
    print("- System auto-captures when 4 groups detected")
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
    
    # Calculate capture window dimensions
    window_height = 400
    window_width = int(window_height * CARD_ASPECT_RATIO)
    
    print(f"Waiting for card...\n")
    
    countdown_active = False
    countdown_start = None
    captured_frame = None
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("[ERROR] Failed to grab frame")
            break
        
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # Calculate centered rectangle
        center_x, center_y = w // 2, h // 2
        rect_w, rect_h = window_width, window_height
        x1 = center_x - rect_w // 2
        y1 = center_y - rect_h // 2
        x2 = x1 + rect_w
        y2 = y1 + rect_h
        
        rect_coords = (x1, y1, x2, y2)
        
        # Quick detection check
        if not countdown_active:
            group_count = quick_detect_groups(frame, rect_coords)
            
            # Check if 4 groups detected
            if group_count >= 4:
                print(f"✓ DETECTED {group_count} DIGIT GROUPS!")
                print("Starting 3-second countdown...\n")
                countdown_active = True
                countdown_start = time.time()
                captured_frame = frame.copy()
        
        # Handle countdown
        if countdown_active:
            elapsed = time.time() - countdown_start
            remaining = 3 - int(elapsed)
            
            if remaining > 0:
                cv2.putText(display_frame, f"CAPTURING IN {remaining}...", 
                           (center_x - 200, center_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
                captured_frame = frame.copy()
            else:
                print("=" * 60)
                print("AUTO-CAPTURING NOW!")
                print("=" * 60)
                
                captured_card = captured_frame[y1:y2, x1:x2]
                
                print(f"Captured image size: {captured_card.shape[1]}x{captured_card.shape[0]}")
                print("Starting processing with IMPROVED MATCHING...\n")
                
                cv2.imwrite('test_card.jpg', captured_card)
                print("[SAVED] Captured image saved as 'test_card.jpg'\n")
                
                cv2.destroyWindow('Webcam - Auto-Capture Mode')
                
                # Process with improved matching
                card_number, card_type, cardholder_name = process_captured_image(
                    captured_card, digits, char
                )
                
                print("\n" + "="*60)
                print("Press any key to close all windows...")
                print("="*60)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                print("\n" + "="*60)
                print("SESSION COMPLETE")
                print("="*60)
                break
        
        # Draw card rectangle
        rect_color = (0, 255, 0) if not countdown_active else (0, 255, 255)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), rect_color, 3)
        
        # Add corner markers
        marker_size = 20
        cv2.line(display_frame, (x1, y1), (x1 + marker_size, y1), rect_color, 3)
        cv2.line(display_frame, (x1, y1), (x1, y1 + marker_size), rect_color, 3)
        cv2.line(display_frame, (x2, y1), (x2 - marker_size, y1), rect_color, 3)
        cv2.line(display_frame, (x2, y1), (x2, y1 + marker_size), rect_color, 3)
        cv2.line(display_frame, (x1, y2), (x1 + marker_size, y2), rect_color, 3)
        cv2.line(display_frame, (x1, y2), (x1, y2 - marker_size), rect_color, 3)
        cv2.line(display_frame, (x2, y2), (x2 - marker_size, y2), rect_color, 3)
        cv2.line(display_frame, (x2, y2), (x2, y2 - marker_size), rect_color, 3)
        
        # Instructions
        if not countdown_active:
            cv2.putText(display_frame, "PLACE CARD - AUTO-CAPTURE WHEN READY", 
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "Using IMPROVED Multi-Method Matching", 
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.putText(display_frame, "Press 'q' to QUIT", 
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow('Webcam - Auto-Capture Mode', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nExiting without capture...")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nWebcam closed.")


# ============================================================
#  MAIN ENTRY POINT
# ============================================================
if __name__ == "__main__":
    run_webcam_auto_capture()