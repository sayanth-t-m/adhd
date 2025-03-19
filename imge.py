import cv2
import pytesseract
import numpy as np
from PIL import Image
import argparse
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()

# Configure Google Gemini API using the provided API key.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyCZAycv__iT01dkrfRsYJNsZW0kRgWxQEA")  # Replace if needed
genai.configure(api_key=GOOGLE_API_KEY)

def preprocess_image(image_path):
    """
    Preprocess the image to enhance handwritten text recognition.
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to handle different lighting conditions
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Noise removal
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Dilate to connect components
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(opening, kernel, iterations=1)
    
    return dilated, img

def extract_handwritten_text(image_path):
    """
    Extract handwritten text from an image.
    """
    # Preprocess the image
    processed_img, original_img = preprocess_image(image_path)
    
    # Configure Tesseract to expect handwritten text.
    custom_config = '''--oem 3 --psm 6 -l eng+osd -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?()-:;'"'''
    
    # Convert from OpenCV format to PIL format
    pil_img = Image.fromarray(processed_img)
    
    # Extract text using pytesseract
    text = pytesseract.image_to_string(pil_img, config=custom_config)
    
    return text.strip(), processed_img, original_img

def analyze_text_for_dyslexia(text, image_path=None):
    """
    Analyze text for potential signs of dyslexia using Google Gemini
    and return a probability score.
    """
    # Create a Gemini model instance
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    prompt = (
        f"Analyze the following text for signs of dyslexia: '{text}'\n\n"
        f"Common signs of dyslexia in writing include:\n"
        f"1. Letter reversals (b/d, p/q)\n"
        f"2. Spelling errors with phonetically similar words\n"
        f"3. Missing or added letters in words\n"
        f"4. Words with mixed capital and lowercase letters\n"
        f"5. Unusual sentence structure\n\n"
        f"Based on the extracted text, provide a detailed analysis of whether the writer shows "
        f"potential signs of dyslexia. Be specific about which signs are present.\n"
        f"Additionally, provide a probability score (0-100%) indicating your confidence in the assessment."
    )

    response = model.generate_content(prompt)
    analysis_text = response.text

    # Extract probability score from response (if mentioned)
    probability = None
    for line in analysis_text.split("\n"):
        if "probability" in line.lower() or "%" in line:
            try:
                probability = float(
                    "".join(filter(str.isdigit, line.split("%")[0])))  # Extract numeric part
                break
            except ValueError:
                pass
    
    # If probability is not found, set it to 'Unknown'
    probability = f"{probability}%" if probability is not None else "Unknown"
    
    return analysis_text, probability

def main():
    # Set up argument parser with a default image path.
    default_path = r"C:\Users\sayan\Downloads\hand.jpg"
    parser = argparse.ArgumentParser(description='Detect signs of dyslexia from handwritten text')
    parser.add_argument('image_path', nargs='?', default=default_path, help='Path to the image with handwritten text')
    parser.add_argument('--show', action='store_true', help='Display the processed image')
    parser.add_argument('--image-analysis', action='store_true', help='Include image in Gemini analysis')
    args = parser.parse_args()
    
    # Extract text
    text, processed_img, original_img = extract_handwritten_text(args.image_path)
    
    # Print extracted text
    print("\nExtracted Text:")
    print("--------------")
    print(text)
    
    # Analyze for dyslexia
    print("\nAnalyzing for signs of dyslexia...")
    image_for_analysis = args.image_path if args.image_analysis else None
    analysis, probability = analyze_text_for_dyslexia(text, image_for_analysis)
    
    print("\nDyslexia Analysis:")
    print("-----------------")
    print(analysis)
    
    # Print probability score
    print("\nEstimated Probability of Dyslexia:", probability)
    
    # Show images if requested
    if args.show:
        cv2.imshow('Original Image', original_img)
        cv2.imshow('Processed Image', processed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
