import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import re
import pytesseract
from pytesseract import Output
from skimage.filters import threshold_local
from PIL import Image
import pandas as pd
from datetime import datetime
import os
import tempfile
import glob
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Set page configuration
st.set_page_config(
    page_title="Hotel Bill Processing System",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Install required packages (commented out for Streamlit - should be in requirements.txt)
# pytesseract, scikit-image, openpyxl, pandas, joblib, scikit-learn should be in requirements.txt

def load_and_preprocess_image(image_path):
    """Load and preprocess the image with enhanced preprocessing"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image if too large for better processing
    height, width = image.shape[:2]
    if height > 2000 or width > 2000:
        scale = min(2000/height, 2000/width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return image

def enhance_image_quality(image):
    """Enhanced image quality improvement for better OCR"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Multiple enhancement techniques
    # 1. Denoising
    denoised = cv2.medianBlur(gray, 3)
    
    # 2. Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(denoised)
    
    # 3. Adaptive thresholding
    T = threshold_local(contrast_enhanced, 15, offset=12, method="gaussian")
    binary = (contrast_enhanced > T).astype("uint8") * 255
    
    # 4. Morphological operations to clean up the image
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return binary, contrast_enhanced, gray

def extract_text_from_image(image):
    """Enhanced text extraction with multiple OCR configurations"""
    # Try different PSM modes for better results
    configs = [
        r'--oem 3 --psm 6',  # Uniform block of text
        r'--oem 3 --psm 4',  # Single column of text
        r'--oem 3 --psm 3',  # Fully automatic page segmentation
    ]
    
    best_text = ""
    best_config = ""
    
    for config in configs:
        try:
            text = pytesseract.image_to_string(image, config=config)
            if len(text) > len(best_text):
                best_text = text
                best_config = config
        except:
            continue
    
    # Get detailed data with best config
    detailed_data = pytesseract.image_to_data(image, output_type=Output.DICT, config=best_config)
    
    return best_text, detailed_data

def extract_hotel_info(text):
    """Enhanced hotel name and address extraction"""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    hotel_name = ""
    address = ""
    address_lines = []

    # Improved patterns for hotel name
    hotel_patterns = [
        r'^[A-Z][A-Za-z\s&\.\-]{3,}(?:Hotel|Resort|Inn|Suites|Lodge|Motel|Plaza)$',
        r'^(?!.*(?:invoice|bill|receipt|date|total|tax|room|guest))[A-Z][A-Za-z\s&\.\-]{3,}$'
    ]

    # Look for hotel name in first 10 lines
    for i, line in enumerate(lines[:10]):
        # Check if line matches hotel patterns
        for pattern in hotel_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                if not hotel_name:
                    hotel_name = line
                    break
        
        # Address detection (usually follows hotel name)
        if hotel_name and i > lines.index(hotel_name) if hotel_name in lines else i > 0:
            if re.search(r'\d+[\sA-Za-z]+,?[\sA-Za-z]+,?[\sA-Za-z]+', line):
                address_lines.append(line)
    
    address = ' '.join(address_lines[:3])  # Take first 3 address lines max

    return hotel_name, address

def extract_dates(text):
    """Enhanced date extraction with multiple formats"""
    date_patterns = [
        # MM/DD/YYYY or DD/MM/YYYY
        r'\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12][0-9]|3[01])[/-](\d{4}|\d{2})\b',
        # DD-MM-YYYY or MM-DD-YYYY
        r'\b(0?[1-9]|[12][0-9]|3[01])[-/](0?[1-9]|1[0-2])[-/](\d{4}|\d{2})\b',
        # Month name formats
        r'\b(0?[1-9]|1[0-2]|0?[1-9]|[12][0-9]|3[01])\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b',
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(0?[1-9]|[12][0-9]|3[01]),?\s+\d{4}\b',
    ]

    dates = []
    for pattern in date_patterns:
        found_dates = re.findall(pattern, text, re.IGNORECASE)
        # Convert tuples to strings
        for date_match in found_dates:
            if isinstance(date_match, tuple):
                date_str = ' '.join([str(part) for part in date_match if part])
                dates.append(date_str)
            else:
                dates.append(date_match)

    # Remove duplicates while preserving order
    seen = set()
    unique_dates = []
    for date in dates:
        if date not in seen:
            seen.add(date)
            unique_dates.append(date)

    check_in = unique_dates[0] if len(unique_dates) > 0 else ""
    check_out = unique_dates[1] if len(unique_dates) > 1 else ""

    return check_in, check_out

def extract_guest_info(text):
    """Enhanced guest information extraction"""
    guest_name = ""
    room_number = ""

    # Improved guest name patterns
    guest_patterns = [
        r'(?:Guest|Name|Customer|Passenger)\s*:?\s*([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z\.]+)+)',
        r'(?:Guest Name|Customer Name)\s*:?\s*([A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+)',
        r'Name\s+of\s+Guest\s*:?\s*([A-Z][a-zA-Z]+\s+[A-Z][a-zA-Z]+)',
    ]

    for pattern in guest_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Take the longest name (most complete)
            guest_name = max(matches, key=len)
            break

    # Enhanced room number patterns
    room_patterns = [
        r'(?:Room|Rm)\.?\s*(?:No|Number|#)?\.?\s*:?\s*([A-Z]?\d+[A-Z]?)',
        r'Room\s+([A-Z]?\d+[A-Z]?)',
        r'Rm\s+([A-Z]?\d+[A-Z]?)',
        r'(?:Room|Rm)\s*:?\s*([A-Z]?\d+[A-Z]?)',
    ]

    for pattern in room_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            room_number = match.group(1)
            break

    return guest_name, room_number

def extract_line_items(text):
    """Enhanced line item extraction with better filtering"""
    items = []
    lines = text.split('\n')

    # Improved patterns for line items
    item_patterns = [
        r'^([A-Za-z][A-Za-z\s\-&]+?)\s+([\$€£]?\s?\d{1,3}(?:[,.]\d{3})*\.?\d{0,2})$',
        r'^([A-Za-z][A-Za-z\s\-&]+?)\s+([\$€£]?\s?\d+\.\d{2})$',
        r'(.+?)\s+(\$?\d+[,.]?\d*\.?\d{2})\s*$'
    ]

    # Keywords to exclude
    exclude_keywords = ['total', 'subtotal', 'tax', 'vat', 'gst', 'balance', 'amount due', 'grand total']

    for line in lines:
        line = line.strip()
        
        # Skip lines with exclude keywords
        if any(keyword in line.lower() for keyword in exclude_keywords):
            continue
        
        # Skip very short lines or lines that are likely headers
        if len(line) < 4 or line.isupper():
            continue

        for pattern in item_patterns:
            match = re.search(pattern, line)
            if match:
                description = match.group(1).strip()
                amount_str = match.group(2).strip()

                # Clean amount
                amount_clean = re.sub(r'[^\d.]', '', amount_str)
                
                # Validate description
                if (len(description) > 2 and 
                    description[0].isalpha() and 
                    not any(word in description.lower() for word in exclude_keywords)):
                    
                    try:
                        amount_float = float(amount_clean)
                        if 0.01 <= amount_float <= 10000:  # Reasonable amount range
                            items.append({
                                'description': description,
                                'amount': amount_float
                            })
                            break  # Stop checking other patterns for this line
                    except ValueError:
                        continue

    return items

def extract_totals(text):
    """Enhanced financial totals extraction"""
    subtotal = 0.0
    tax = 0.0
    total = 0.0

    # Improved patterns for financial amounts
    patterns = {
        'subtotal': [
            r'(?:Sub\s*Total|Subtotal)\s*:?\s*[\$€£]?\s*(\d{1,3}(?:[,.]\d{3})*\.?\d{0,2})',
            r'(?:Sub\s*Total|Subtotal)\s*[\$€£]?\s*(\d{1,3}(?:[,.]\d{3})*\.?\d{0,2})'
        ],
        'tax': [
            r'(?:Tax|GST|VAT)\s*:?\s*[\$€£]?\s*(\d{1,3}(?:[,.]\d{3})*\.?\d{0,2})',
            r'(?:Tax|GST|VAT)\s*[\$€£]?\s*(\d{1,3}(?:[,.]\d{3})*\.?\d{0,2})'
        ],
        'total': [
            r'(?:Total|Grand\s*Total|Amount\s*Due|Balance\s*Due)\s*:?\s*[\$€£]?\s*(\d{1,3}(?:[,.]\d{3})*\.?\d{0,2})',
            r'(?:Total|Grand\s*Total|Amount\s*Due|Balance)\s*[\$€£]?\s*(\d{1,3}(?:[,.]\d{3})*\.?\d{0,2})'
        ]
    }

    for key, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Take the last occurrence (usually the final amount)
                value_str = matches[-1].replace(',', '').replace(' ', '')
                try:
                    value_float = float(value_str)
                    if key == 'subtotal':
                        subtotal = value_float
                    elif key == 'tax':
                        tax = value_float
                    elif key == 'total':
                        total = value_float
                except ValueError:
                    continue

    return subtotal, tax, total

def extract_invoice_number(text):
    """Enhanced invoice number extraction"""
    patterns = [
        r'(?:Invoice|Bill|Receipt)\s*(?:No|Number|#)?\.?\s*:?\s*([A-Z0-9\-/#]+)',
        r'(?:Folio|Confirmation)\s*(?:No|Number|#)?\.?\s*:?\s*([A-Z0-9\-/#]+)',
        r'(?:Invoice|Bill)\s*#?\s*:?\s*([A-Z0-9\-/#]+)',
        r'[A-Z]{2,}\d{4,}',  # Pattern like INVOICE1234
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[0]

    return ""

class HotelBillProcessor:
    """Model class to handle hotel bill processing and saving"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.processed_data = []
        self.model_version = "1.0"
        
    def fit(self, texts):
        """Fit the vectorizer on text data"""
        if texts:
            self.vectorizer.fit(texts)
    
    def save_model(self, filename="hotel_bill_processor_model.pkl"):
        """Save the trained model"""
        model_data = {
            'vectorizer': self.vectorizer,
            'processed_data': self.processed_data,
            'model_version': self.model_version,
            'timestamp': datetime.now()
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        st.success(f"✅ Model saved as {filename}")
        return filename

def process_hotel_bill(image_path, processor=None):
    """Main function to process hotel bill and extract structured data"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Loading image...")
    original_image = load_and_preprocess_image(image_path)
    progress_bar.progress(20)

    status_text.text("Enhancing image quality...")
    binary_image, contrast_enhanced, gray_image = enhance_image_quality(original_image)
    progress_bar.progress(40)

    status_text.text("Extracting text from enhanced images...")
    text_binary, _ = extract_text_from_image(binary_image)
    text_contrast, _ = extract_text_from_image(contrast_enhanced)
    text_gray, _ = extract_text_from_image(gray_image)
    progress_bar.progress(70)

    # Use the best result
    texts = [text_binary, text_contrast, text_gray]
    final_text = max(texts, key=len)

    status_text.text("Extracting structured data...")
    progress_bar.progress(85)

    # Extract all information
    hotel_name, address = extract_hotel_info(final_text)
    check_in, check_out = extract_dates(final_text)
    guest_name, room_number = extract_guest_info(final_text)
    invoice_number = extract_invoice_number(final_text)
    line_items = extract_line_items(final_text)
    subtotal, tax, total = extract_totals(final_text)

    # Store in processor if provided
    if processor:
        processor.processed_data.append({
            'hotel_name': hotel_name,
            'raw_text': final_text,
            'image_path': image_path
        })

    progress_bar.progress(100)
    status_text.text("Processing completed!")
    
    # Display images in Streamlit
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(original_image, use_column_width=True)
        
        st.subheader("Grayscale Image")
        st.image(gray_image, use_column_width=True, clamp=True)
    
    with col2:
        st.subheader("Contrast Enhanced")
        st.image(contrast_enhanced, use_column_width=True, clamp=True)
        
        st.subheader("Binary Image")
        st.image(binary_image, use_column_width=True, clamp=True)

    return {
        'hotel_name': hotel_name,
        'address': address,
        'invoice_number': invoice_number,
        'guest_name': guest_name,
        'room_number': room_number,
        'check_in_date': check_in,
        'check_out_date': check_out,
        'line_items': line_items,
        'subtotal': subtotal,
        'tax': tax,
        'total': total,
        'raw_text': final_text,
        'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'image_filename': os.path.basename(image_path)
    }

def save_single_bill_to_excel(results, output_filename=None):
    """Save single bill data to Excel with multiple sheets"""
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hotel_name_clean = re.sub(r'[^\w\s-]', '', results['hotel_name'] or 'hotel')[:20]
        output_filename = f"{hotel_name_clean}_{results['guest_name'] or 'guest'}_{timestamp}.xlsx"
        output_filename = re.sub(r'[^\w.-]', '_', output_filename)

    # Create a Pandas Excel writer
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:

        # Sheet 1: Summary Information
        summary_data = {
            'Field': [
                'Hotel Name',
                'Address',
                'Invoice Number',
                'Guest Name',
                'Room Number',
                'Check-in Date',
                'Check-out Date',
                'Subtotal',
                'Tax',
                'Total Amount',
                'Processing Date',
                'Image Filename'
            ],
            'Value': [
                results['hotel_name'],
                results['address'],
                results['invoice_number'],
                results['guest_name'],
                results['room_number'],
                results['check_in_date'],
                results['check_out_date'],
                f"${results['subtotal']:.2f}" if results['subtotal'] else '',
                f"${results['tax']:.2f}" if results['tax'] else '',
                f"${results['total']:.2f}" if results['total'] else '',
                results['processing_date'],
                results['image_filename']
            ]
        }

        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)

        # Sheet 2: Line Items
        if results['line_items']:
            df_items = pd.DataFrame(results['line_items'])
            df_items['amount'] = df_items['amount'].apply(lambda x: f"${x:.2f}")
            df_items.to_excel(writer, sheet_name='Line Items', index=False)
        else:
            # Create empty line items sheet
            pd.DataFrame({'description': [], 'amount': []}).to_excel(writer, sheet_name='Line Items', index=False)

        # Sheet 3: Raw Text
        cleaned_raw_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', results['raw_text'])
        df_raw = pd.DataFrame({'Raw Extracted Text': [cleaned_raw_text]})
        df_raw.to_excel(writer, sheet_name='Raw Text', index=False)

    return output_filename

def save_multiple_bills_to_master_excel(all_results, master_filename='all_hotel_bills_master.xlsx'):
    """Save all processed bills to a master Excel file"""
    
    # Prepare data for master sheets
    summary_records = []
    all_line_items = []
    
    for i, result in enumerate(all_results):
        # Summary record
        summary_records.append({
            'Bill ID': i + 1,
            'Hotel Name': result['hotel_name'],
            'Guest Name': result['guest_name'],
            'Room Number': result['room_number'],
            'Invoice Number': result['invoice_number'],
            'Check-in Date': result['check_in_date'],
            'Check-out Date': result['check_out_date'],
            'Subtotal': result['subtotal'],
            'Tax': result['tax'],
            'Total Amount': result['total'],
            'Processing Date': result['processing_date'],
            'Image Filename': result['image_filename']
        })
        
        # Line items
        for item in result['line_items']:
            all_line_items.append({
                'Bill ID': i + 1,
                'Hotel Name': result['hotel_name'],
                'Guest Name': result['guest_name'],
                'Description': item['description'],
                'Amount': item['amount']
            })
    
    # Create master Excel file
    with pd.ExcelWriter(master_filename, engine='openpyxl') as writer:
        # Master Summary Sheet
        if summary_records:
            df_master_summary = pd.DataFrame(summary_records)
            df_master_summary.to_excel(writer, sheet_name='All Bills Summary', index=False)
        
        # Master Line Items Sheet
        if all_line_items:
            df_master_items = pd.DataFrame(all_line_items)
            df_master_items.to_excel(writer, sheet_name='All Line Items', index=False)
        else:
            pd.DataFrame({'Bill ID': [], 'Hotel Name': [], 'Guest Name': [], 'Description': [], 'Amount': []}).to_excel(writer, sheet_name='All Line Items', index=False)
    
    return master_filename

def display_extracted_summary(results):
    """Display a clean summary of extracted data"""
    st.subheader("📋 Extracted Data Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("🏨 Hotel Name", results['hotel_name'] or "Not found")
        st.metric("📍 Address", results['address'] or "Not found")
        st.metric("👤 Guest Name", results['guest_name'] or "Not found")
        st.metric("🚪 Room Number", results['room_number'] or "Not found")
        
    with col2:
        st.metric("📄 Invoice Number", results['invoice_number'] or "Not found")
        st.metric("📅 Check-in", results['check_in_date'] or "Not found")
        st.metric("📅 Check-out", results['check_out_date'] or "Not found")
        st.metric("📦 Line Items", len(results['line_items']))
    
    # Financial Summary
    st.subheader("💰 Financial Summary")
    fin_col1, fin_col2, fin_col3 = st.columns(3)
    
    with fin_col1:
        st.metric("Subtotal", f"${results['subtotal']:.2f}" if results['subtotal'] else "Not found")
    with fin_col2:
        st.metric("Tax", f"${results['tax']:.2f}" if results['tax'] else "Not found")
    with fin_col3:
        st.metric("Total", f"${results['total']:.2f}" if results['total'] else "Not found")
    
    # Line Items
    if results['line_items']:
        st.subheader("📦 Line Items")
        items_df = pd.DataFrame(results['line_items'])
        items_df['amount'] = items_df['amount'].apply(lambda x: f"${x:.2f}")
        st.dataframe(items_df, use_container_width=True)
    
    # Raw Text (collapsible)
    with st.expander("View Raw Extracted Text"):
        st.text_area("Raw Text", results['raw_text'], height=200)

# Main Streamlit App
def main():
    st.title("🏨 Hotel Bill Processing System")
    st.markdown("---")
    
    # Initialize session state
    if 'processed_results' not in st.session_state:
        st.session_state.processed_results = []
    if 'processor' not in st.session_state:
        st.session_state.processor = HotelBillProcessor()
    
    # Sidebar
    st.sidebar.title("Upload Bills")
    st.sidebar.markdown("Upload one or multiple hotel bill images for processing.")
    
    uploaded_files = st.sidebar.file_uploader(
        "Choose hotel bill images",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        accept_multiple_files=True,
        help="Upload clear images of hotel bills for processing"
    )
    
    # Main content area
    if uploaded_files:
        st.success(f"📁 {len(uploaded_files)} file(s) uploaded successfully!")
        
        # Process buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            process_single = st.button("🔄 Process All Bills", type="primary")
        
        with col2:
            clear_results = st.button("🗑️ Clear Results")
        
        with col3:
            download_all = st.button("📥 Download All Results")
        
        if clear_results:
            st.session_state.processed_results = []
            st.rerun()
        
        if process_single and uploaded_files:
            all_results = []
            
            for uploaded_file in uploaded_files:
                # Save uploaded file to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Process the bill
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        result = process_hotel_bill(tmp_path, st.session_state.processor)
                        all_results.append(result)
                        st.session_state.processed_results.append(result)
                    
                    # Display results for this bill
                    st.subheader(f"Results for: {uploaded_file.name}")
                    display_extracted_summary(result)
                    
                    # Save individual Excel file
                    individual_excel = save_single_bill_to_excel(result)
                    
                    # Provide download button for individual file
                    with open(individual_excel, 'rb') as f:
                        st.download_button(
                            label=f"📥 Download {os.path.basename(individual_excel)}",
                            data=f,
                            file_name=os.path.basename(individual_excel),
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"download_{uploaded_file.name}"
                        )
                    
                    # Clean up
                    os.unlink(tmp_path)
                    if os.path.exists(individual_excel):
                        os.unlink(individual_excel)
                    
                    st.markdown("---")
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    continue
            
            # Fit processor with extracted texts
            if all_results:
                texts = [result['raw_text'] for result in all_results]
                st.session_state.processor.fit(texts)
                
                # Save model
                model_filename = st.session_state.processor.save_model()
                
                with open(model_filename, 'rb') as f:
                    st.sidebar.download_button(
                        label="🤖 Download Trained Model",
                        data=f,
                        file_name=model_filename,
                        mime="application/octet-stream"
                    )
                
                # Clean up model file
                if os.path.exists(model_filename):
                    os.unlink(model_filename)
        
        # Download all results
        if download_all and st.session_state.processed_results:
            if len(st.session_state.processed_results) > 1:
                master_excel = save_multiple_bills_to_master_excel(st.session_state.processed_results)
                
                with open(master_excel, 'rb') as f:
                    st.sidebar.download_button(
                        label="📚 Download Master Excel File",
                        data=f,
                        file_name=master_excel,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                # Clean up
                if os.path.exists(master_excel):
                    os.unlink(master_excel)
            else:
                st.warning("Upload multiple bills to download master file.")
        
        # Display overall statistics
        if st.session_state.processed_results:
            st.sidebar.markdown("---")
            st.sidebar.subheader("📊 Processing Statistics")
            st.sidebar.metric("Total Bills Processed", len(st.session_state.processed_results))
            st.sidebar.metric("Total Line Items", sum(len(result['line_items']) for result in st.session_state.processed_results))
            
            # List processed bills
            st.sidebar.subheader("Processed Bills")
            for i, result in enumerate(st.session_state.processed_results, 1):
                st.sidebar.text(f"{i}. {result['hotel_name'] or 'Unknown Hotel'}")
    
    else:
        # Welcome message and instructions
        st.markdown("""
        ## Welcome to the Hotel Bill Processing System!
        
        This application uses advanced OCR and image processing to extract structured data from hotel bills.
        
        ### 🚀 How to use:
        1. **Upload** hotel bill images using the file uploader in the sidebar
        2. **Process** the bills using the processing button
        3. **Review** the extracted data and download results
        
        ### 📋 What information is extracted:
        - Hotel name and address
        - Guest information and room number
        - Check-in and check-out dates
        - Invoice number
        - Line item charges
        - Financial totals (subtotal, tax, total)
        
        ### 💡 Tips for best results:
        - Use clear, high-quality images
        - Ensure bills are well-lit and properly aligned
        - Avoid glare and shadows on the bill
        - Crop to include only the bill content when possible
        """)
        
        # Example image (optional)
        st.info("💡 Supported formats: JPG, JPEG, PNG, BMP, TIFF")

    # Footer
    st.markdown("---")
    st.markdown(
        "**Hotel Bill Processing System** | "
        "Built with Streamlit, OpenCV, and Tesseract OCR"
    )

if __name__ == "__main__":
    main()
