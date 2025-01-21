from PIL import Image
import pytesseract

# Load the image
image_path = "/mnt/data/image.png"
image = Image.open(image_path)

# Extract text from the image using pytesseract
extracted_text = pytesseract.image_to_string(image)

# Display the extracted text
extracted_text
