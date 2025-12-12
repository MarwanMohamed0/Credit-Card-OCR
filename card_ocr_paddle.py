from paddleocr import PaddleOCR
import cv2
import numpy as np

# Initialize a multilingual OCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en', det=True, rec=True)

# ---------- PROCESSING FUNCTION ----------
def extract_card_info(image):
    result = ocr.ocr(image, cls=True)

    text_blocks = []
    for line in result:
        for box, (txt, prob) in line:
            text_blocks.append(txt)

    # Join all detected text
    all_text = " ".join(text_blocks)

    # Extract 16-digit card number
    import re
    number = re.findall(r"\b\d{4} \d{4} \d{4} \d{4}\b", all_text)
    if not number:
        number = re.findall(r"\d{16}", all_text)

    card_number = number[0] if number else "NOT DETECTED"

    # Extract expiry date
    expiry = re.findall(r"(0[1-9]|1[0-2])[/\- ]?([0-9]{2,4})", all_text)
    expiry_date = expiry[0][0] + "/" + expiry[0][1][-2:] if expiry else "NOT DETECTED"

    # Extract name (heuristic: text without digits, usually uppercase)
    possible_names = [t for t in text_blocks if t.replace(" ", "").isalpha()]
    cardholder = max(possible_names, key=len) if possible_names else "NOT DETECTED"

    return card_number, expiry_date, cardholder


# ---------- MAIN ----------
print("\n========== CREDIT CARD OCR ==========\n")
print("Press 's' to capture card")
print("Press 'q' to quit\n")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow("Card OCR", frame)
    key = cv2.waitKey(1)

    if key == ord('s'):
        print("\n[INFO] Processing...")
        card_number, expiry, name = extract_card_info(frame)

        print("\n===== RESULTS =====")
        print("Card Number :", card_number)
        print("Expiry Date :", expiry)
        print("Cardholder  :", name)
        print("===================\n")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
