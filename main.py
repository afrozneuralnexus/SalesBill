import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
import pickle
import tempfile
import os

# Set page configuration
st.set_page_config(
    page_title="Hotel Bill OCR",
    page_icon="🧾",
    layout="wide"
)

# Custom preprocessing function (simpler alternative to scikit-image)
def custom_threshold_local(image, block_size=11, offset=10):
    """Custom implementation of local thresholding"""
    # Simple adaptive thresholding as alternative
    return cv2.adaptiveThreshold(
        image, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        block_size, 
        offset
    )

def preprocess_image(image):
    """Preprocess image for OCR"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply noise reduction
        denoised = cv2.medianBlur(gray, 3)
        
        # Apply thresholding (using custom function)
        binary = custom_threshold_local(denoised, 11, 10)
        
        return binary, denoised
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        return image, image

def extract_text(image):
    """Extract text using pytesseract"""
    try:
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(image, config=custom_config)
        
        # Clean up text
        text = re.sub(r'\s+', ' ', text).strip()
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
            
            confidence_threshold = st.slider(
                "Confidence Level", 
                min_value=1, 
                max_value=10, 
                value=6,
                help="Higher values may extract less text but with higher accuracy"
            )
            
            if st.button("Extract Text", type="primary"):
                with st.spinner("Processing image and extracting text..."):
                    try:
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
                        
                    except Exception as e:
                        st.error(f"An error occurred during processing: {str(e)}")
    
    else:
        # Show instructions when no file is uploaded
        st.info("👆 Please upload a hotel bill image to get started")
        
        # Sample usage instructions
        with st.expander("ℹ️ How to get best results"):
            st.markdown("""
            - **Use clear, high-resolution images**
            - **Ensure good lighting** when taking photos
            - **Position the bill straight** and avoid angles
            - **Include the entire bill** in the frame
            - **Avoid shadows and glares** on the bill
            
            **Supported formats:** JPG, JPEG, PNG
            
            **Note:** This app uses Tesseract OCR for text extraction. For best results, ensure your images are clear and well-lit.
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

if __name__ == "__main__":
    # Note for deployment
    st.sidebar.info("""
    **Deployment Note:** 
    For Streamlit Cloud deployment, ensure Tesseract OCR is available in the environment.
    """)
    main()
