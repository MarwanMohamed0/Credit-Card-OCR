import pytesseract
import cv2

# Load an image
img = cv2.imread("Sample_images/credit_card_06.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Use pytesseract to extract text
text = pytesseract.image_to_string(gray)
print(text)