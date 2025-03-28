#!/usr/bin/env python3
import cv2
import numpy as np
import os
import io
import google.generativeai as genai
from PIL import Image, ImageTk
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from dotenv import load_dotenv

# Load environment variables (if any)
load_dotenv()

# Set the Google API key directly (for production, use environment variables)
GOOGLE_API_KEY = "AIzaSyCZAycv__iT01dkrfRsYJNsZW0kRgWxQEA"
genai.configure(api_key=GOOGLE_API_KEY)
# Use the recommended model gemini-1.5-flash
MODEL_NAME = 'gemini-1.5-flash'
model = genai.GenerativeModel(MODEL_NAME)

DEBUG = False  # Set True for debug output

# --------------------- Direct Visual Analysis Function ---------------------
def analyze_image_direct_for_dyslexia(image_path):
    """
    Directly analyzes the image using Google Gemini for visual cues that might be
    associated with dyslexia. The prompt now instructs the model to look for key indicators:
    
    1. Letter reversals (e.g. b/d, p/q)
    2. Inconsistent letter formation
    3. Omissions or insertions in words
    4. Transpositions (letters swapped)
    5. Irregular spacing between letters/words
    6. Poor alignment or baseline inconsistencies
    7. Overall visual clutter or disorganization
    
    Based on these, the model should assign a score (0-100%) representing the likelihood that
    the handwriting exhibits dyslexic traits.
    """
    if not model:
        return None, "Gemini API not configured. Please check your API key and model setup."
    try:
        image = Image.open(image_path)
        byte_stream = io.BytesIO()
        image.save(byte_stream, format=image.format)
        image_bytes = byte_stream.getvalue()
    except FileNotFoundError:
        return None, "Image file not found."
    except Exception as e:
        return None, f"Error opening image: {e}"

    prompt = """
    Analyze the following handwritten page for visual cues that may indicate dyslexia.
    
    Please evaluate the image based on the following indicators:
    
    1. **Letter Reversals:** Check if letters such as 'b' and 'd' or 'p' and 'q' appear reversed.
    2. **Inconsistent Letter Formation:** Look for variations in the shape, size, or slant of letters.
    3. **Omissions or Insertions:** Identify if any letters seem to be missing or extra strokes are present.
    4. **Transpositions:** Determine if letters are swapped or out of the expected order.
    5. **Irregular Spacing:** Evaluate whether there is uneven spacing between letters or words.
    6. **Poor Alignment/Baseline:** Check if the writing lacks a consistent line or baseline.
    7. **Visual Clutter:** Observe if the overall page appears disorganized or cluttered.
    
    Based on these observations, provide a concise summary of the key dyslexia indicators present.
    Then, assign a score from 0 to 100% representing the likelihood that the handwriting shows signs of dyslexia.
    Clearly state that this analysis is speculative and should not be considered a professional diagnosis.
    """
    try:
        response = genai.GenerativeModel(MODEL_NAME).generate_content(
            [prompt, {"mime_type": "image/jpeg", "data": image_bytes}]
        )
        return response.text, "N/A"
    except Exception as e:
        return None, f"Error during Gemini API call: {e}"

# --------------------- GUI Application ---------------------
class DyslexiaAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Direct Visual Dyslexia Analyzer")
        self.root.geometry("800x800")
        
        # Frame for image upload and display
        image_frame = tk.Frame(root)
        image_frame.pack(padx=10, pady=5, fill=tk.X)
        
        self.upload_button = tk.Button(image_frame, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(side=tk.TOP, padx=5, pady=5)
        
        # Original image display
        orig_frame = tk.Frame(image_frame)
        orig_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.original_label = tk.Label(orig_frame, text="Original Image")
        self.original_label.pack()
        self.original_canvas = tk.Label(orig_frame)
        self.original_canvas.pack(padx=5, pady=5, expand=True, fill=tk.BOTH)
        
        # Results text area with scrollbar
        result_frame = tk.Frame(root)
        result_frame.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)
        result_scroll = tk.Scrollbar(result_frame)
        result_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text = tk.Text(result_frame, height=15, width=100, wrap=tk.WORD, yscrollcommand=result_scroll.set)
        self.result_text.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        result_scroll.config(command=self.result_text.yview)
        
        # Analyze button (direct visual analysis)
        self.analyze_button = tk.Button(root, text="Direct Visual Analysis for Dyslexia", command=self.analyze_image, state=tk.DISABLED)
        self.analyze_button.pack(pady=5)
        
        # Progress bar (hidden by default)
        self.progress = ttk.Progressbar(root, orient="horizontal", mode="indeterminate", length=400)
        self.progress.pack(pady=10)
        self.progress.pack_forget()
        
        # Placeholder for current image path
        self.image_path = None

        # Disable analyze button if the Gemini API failed to configure
        if not model:
            self.analyze_button.config(state=tk.DISABLED)
            self.log("Gemini API is not configured. Analysis disabled.")

    def log(self, message):
        self.result_text.insert(tk.END, message + "\n")
        self.result_text.see(tk.END)

    def upload_image(self):
        path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if path:
            self.image_path = path
            self.log("Image uploaded: " + path)
            # Load and display original image (resized)
            orig = cv2.imread(path)
            orig_disp = self._resize_for_display(orig)
            orig_disp = cv2.cvtColor(orig_disp, cv2.COLOR_BGR2RGB)
            orig_image = ImageTk.PhotoImage(Image.fromarray(orig_disp))
            self.original_canvas.configure(image=orig_image)
            self.original_canvas.image = orig_image
            self.analyze_button.config(state=tk.NORMAL)

    def _resize_for_display(self, image, max_dim=400):
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            image = cv2.resize(image, (int(w*scale), int(h*scale)))
        return image

    def analyze_image(self):
        if not self.image_path:
            messagebox.showerror("Error", "No image uploaded!")
            return
        self.log("Starting direct visual analysis...")
        # Show and start progress bar
        self.progress.pack(pady=10)
        self.progress.start()
        self.analyze_button.config(state=tk.DISABLED)
        threading.Thread(target=self._process_and_analyze, daemon=True).start()

    def _process_and_analyze(self):
        try:
            analysis, probability = analyze_image_direct_for_dyslexia(self.image_path)
            # Stop and hide progress bar on main thread
            self.root.after(0, self.progress.stop)
            self.root.after(0, self.progress.pack_forget)
            self.root.after(0, self._update_gui_with_results, analysis, probability)
        except Exception as e:
            self.root.after(0, self.progress.stop)
            self.root.after(0, self.progress.pack_forget)
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.root.after(0, lambda: self.log("Error during analysis: " + str(e)))
        finally:
            self.root.after(0, lambda: self.analyze_button.config(state=tk.NORMAL))

    def _update_gui_with_results(self, analysis, probability):
        self.result_text.delete("1.0", tk.END)
        self.log("Direct Visual Analysis for Dyslexia:")
        self.log("-" * 20)
        self.log(analysis if analysis else "No analysis returned.")
        self.log(f"\nEstimated Probability: {probability}")

def main():
    root = tk.Tk()
    app = DyslexiaAnalyzerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
