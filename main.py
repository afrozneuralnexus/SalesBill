import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
import tempfile
import os

# Set page configuration
st.set_page_config(
    page_title="Hotel Bill OCR",
    page_icon="🧾",
    layout="wide"
)

def preprocess_image(image):
    """Preprocess image for OCR"""
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply noise reduction
        denoised = cv2.medianBlur(gray, 3)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            denoised, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 
            10
        )
        
        return binary, denoised
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        # Return original images if preprocessing fails
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return gray, gray
        return image, image

def extract_text(image):
    """Extract text using pytesseract"""
    try:
        # Try different configurations for better results
        configs = [
            r'--oem 3 --psm 6',
            r'--oem 3 --psm 4',
            r'--oem 3 --psm 3'
        ]
        
        best_text = ""
        for config in configs:
            try:
                text = pytesseract.image_to_string(image, config=config)
                if len(text.strip()) > len(best_text.strip()):
                    best_text = text
            except:
                continue
        
        if not best_text.strip():
            # Fallback to basic config
            best_text = pytesseract.image_to_string(image)
        
        # Clean up text
        text = re.sub(r'\s+', ' ', best_text).strip()
        text = re.sub(r'\s\w\s', ' ', text)
        
        return text
    except Exception as e:
        st.error(f"Error in text extraction: {e}")
        return ""

def main():
    st.title("🧾 Hotel Bill OCR Processor")
    st.markdown("Upload a hotel bill image to extract text using OCR")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a hotel bill image", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a hotel bill for text extraction"
    )
    
    if uploaded_file is not None:
        try:
            # Display uploaded image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
                
                # Convert to OpenCV format
                image_cv = np.array(image)
                if len(image_cv.shape) == 3:
                    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
                    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.subheader("Processing Options")
                process_option = st.radio(
                    "Select processing method:",
                    ["Automatic (Recommended)", "Grayscale only", "Binary only"]
                )
                
                if st.button("Extract Text", type="primary"):
                    with st.spinner("Processing image and extracting text..."):
                        # Preprocess image
                        binary_img, gray_img = preprocess_image(image_cv)
                        
                        # Extract text based on selected option
                        if process_option == "Automatic (Recommended)":
                            text_binary = extract_text(binary_img)
                            text_gray = extract_text(gray_img)
                            final_text = text_binary if len(text_binary) > len(text_gray) else text_gray
                            used_method = "Binary" if len(text_binary) > len(text_gray) else "Grayscale"
                        elif process_option == "Grayscale only":
                            final_text = extract_text(gray_img)
                            used_method = "Grayscale"
                        else:
                            final_text = extract_text(binary_img)
                            used_method = "Binary"
                        
                        # Display results
                        st.subheader("📄 Extracted Text")
                        st.info(f"Used method: {used_method}")
                        
                        if final_text.strip():
                            # Text area for easy copying
                            st.text_area(
                                "Extracted Text", 
                                final_text, 
                                height=200,
                                help="You can copy the extracted text from here"
                            )
                            
                            # Display processed images
                            st.subheader("🖼️ Processed Images")
                            proc_col1, proc_col2 = st.columns(2)
                            
                            with proc_col1:
                                st.image(gray_img, caption="Grayscale Image", use_column_width=True, clamp=True)
                            
                            with proc_col2:
                                st.image(binary_img, caption="Binary Image", use_column_width=True, clamp=True)
                            
                            # Download extracted text
                            st.download_button(
                                label="📥 Download Extracted Text",
                                data=final_text,
                                file_name="extracted_bill_text.txt",
                                mime="text/plain"
                            )
                        else:
                            st.warning("No text could be extracted from the image. Please try with a clearer image.")
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.info("Please try uploading a different image or check the image format.")
    
    else:
        # Show instructions when no file is uploaded
        st.info("👆 Please upload a hotel bill image to get started")
        
        # Sample usage instructions
        with st.expander("ℹ️ How to get best results"):
            st.markdown("""
            ### Tips for Best Results:
            
            - **Use clear, high-resolution images**
            - **Ensure good lighting** when taking photos
            - **Position the bill straight** and avoid angles
            - **Include the entire bill** in the frame
            - **Avoid shadows and glares** on the bill
            
            ### Supported Formats:
            - JPG, JPEG, PNG
            
            ### Note:
            This app uses Tesseract OCR for text extraction. 
            For best results, ensure your images are clear and well-lit.
            """)

    # Add information about the app
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This app uses OCR (Optical Character Recognition) to extract text from hotel bill images.
        
        **Features:**
        - Multiple image preprocessing techniques
        - Automatic text extraction
        - Download extracted text
        
        **Technology Stack:**
        - Streamlit
        - OpenCV
        - Tesseract OCR
        - Pillow (PIL)
        """)
        
        st.header("Troubleshooting")
        st.markdown("""
        If you encounter issues:
        1. Try a different image
        2. Ensure the image is clear
        3. Check file format (JPG/PNG)
        4. Try the 'Grayscale only' option
        """)

if __name__ == "__main__":
    main()
