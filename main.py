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
from google.colab.patches import cv2_imshow

# Install required packages
!sudo apt install tesseract-ocr
!pip install pytesseract scikit-image openpyxl pandas

def load_and_preprocess_image(image_path):
    """Load and preprocess the image"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def enhance_image_quality(image):
    """Enhance image quality for better OCR"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    T = threshold_local(denoised, 11, offset=10, method="gaussian")
    binary = (denoised > T).astype("uint8") * 255
    return binary, denoised

def extract_text_from_image(image):
    """Extract text using pytesseract"""
    custom_config = r'--oem 3 --psm 6'
    text_basic = pytesseract.image_to_string(image, config=custom_config)
    detailed_data = pytesseract.image_to_data(image, output_type=Output.DICT)
    return text_basic, detailed_data

def extract_hotel_info(text):
    """Extract hotel name and address"""
    lines = text.split('\n')
    hotel_name = ""
    address = ""
    
    # Usually hotel name is in the first few lines
    for i, line in enumerate(lines[:5]):
        if line.strip() and len(line.strip()) > 5:
            if not hotel_name:
                hotel_name = line.strip()
            elif not address and i > 0:
                address = line.strip()
                break
    
    return hotel_name, address

def extract_dates(text):
    """Extract check-in and check-out dates"""
    date_patterns = [
        r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
        r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}',
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}'
    ]
    
    dates = []
    for pattern in date_patterns:
        found_dates = re.findall(pattern, text, re.IGNORECASE)
        dates.extend(found_dates)
    
    check_in = dates[0] if len(dates) > 0 else ""
    check_out = dates[1] if len(dates) > 1 else ""
    
    return check_in, check_out

def extract_guest_info(text):
    """Extract guest name and room number"""
    guest_name = ""
    room_number = ""
    
    # Look for guest name patterns
    guest_patterns = [
        r'(?:Guest|Name|Mr\.|Mrs\.|Ms\.)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        r'(?:Guest Name|Customer)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
    ]
    
    for pattern in guest_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            guest_name = match.group(1)
            break
    
    # Look for room number
    room_patterns = [
        r'(?:Room|Rm)\s*(?:No|Number|#)?\.?\s*:?\s*(\d+)',
        r'Room\s+(\d+)'
    ]
    
    for pattern in room_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            room_number = match.group(1)
            break
    
    return guest_name, room_number

def extract_line_items(text):
    """Extract itemized charges"""
    items = []
    lines = text.split('\n')
    
    # Pattern to match line items with amounts
    item_pattern = r'(.+?)\s+(\$?\d+[,.]?\d*\.?\d{2})'
    
    for line in lines:
        # Skip header lines and total lines
        if any(keyword in line.lower() for keyword in ['total', 'subtotal', 'tax', 'balance', 'amount due']):
            continue
        
        match = re.search(item_pattern, line)
        if match:
            description = match.group(1).strip()
            amount = match.group(2).replace('$', '').replace(',', '')
            
            # Filter out noise
            if len(description) > 3 and description[0].isalpha():
                try:
                    amount_float = float(amount)
                    if amount_float > 0:
                        items.append({
                            'description': description,
                            'amount': amount_float
                        })
                except ValueError:
                    pass
    
    return items

def extract_totals(text):
    """Extract financial totals"""
    subtotal = 0.0
    tax = 0.0
    total = 0.0
    
    # Patterns for different total types
    patterns = {
        'subtotal': r'(?:Subtotal|Sub Total)\s*:?\s*\$?\s*(\d+[,.]?\d*\.?\d{2})',
        'tax': r'(?:Tax|GST|VAT)\s*:?\s*\$?\s*(\d+[,.]?\d*\.?\d{2})',
        'total': r'(?:Total|Grand Total|Amount Due|Balance)\s*:?\s*\$?\s*(\d+[,.]?\d*\.?\d{2})'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).replace('$', '').replace(',', '')
            try:
                if key == 'subtotal':
                    subtotal = float(value)
                elif key == 'tax':
                    tax = float(value)
                elif key == 'total':
                    total = float(value)
            except ValueError:
                pass
    
    return subtotal, tax, total

def extract_invoice_number(text):
    """Extract invoice/bill number"""
    patterns = [
        r'(?:Invoice|Bill|Receipt)\s*(?:No|Number|#)?\.?\s*:?\s*([A-Z0-9-]+)',
        r'(?:Folio|Confirmation)\s*(?:No|Number|#)?\.?\s*:?\s*([A-Z0-9-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return ""

def process_hotel_bill(image_path):
    """Main function to process hotel bill and extract structured data"""
    print("Loading image...")
    original_image = load_and_preprocess_image(image_path)
    
    print("Enhancing image quality...")
    binary_image, gray_image = enhance_image_quality(original_image)
    
    print("Extracting text from enhanced image...")
    text_binary, _ = extract_text_from_image(binary_image)
    
    print("Extracting text from grayscale image...")
    text_gray, _ = extract_text_from_image(gray_image)
    
    # Use the better result
    final_text = text_binary if len(text_binary) > len(text_gray) else text_gray
    
    print("Extracting structured data...")
    
    # Extract all information
    hotel_name, address = extract_hotel_info(final_text)
    check_in, check_out = extract_dates(final_text)
    guest_name, room_number = extract_guest_info(final_text)
    invoice_number = extract_invoice_number(final_text)
    line_items = extract_line_items(final_text)
    subtotal, tax, total = extract_totals(final_text)
    
    # Display extracted text
    print("\n=== EXTRACTED TEXT ===")
    print("-" * 50)
    print(final_text)
    print("-" * 50)
    
    # Display images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(gray_image, cmap='gray')
    axes[1].set_title('Grayscale Image')
    axes[1].axis('off')
    
    axes[2].imshow(binary_image, cmap='gray')
    axes[2].set_title('Binary Image')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
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
        'raw_text': final_text
    }

def save_to_excel(results, output_filename='hotel_bill_data.xlsx'):
    """Save extracted data to Excel with multiple sheets"""
    
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
                'Total Amount'
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
                f"${results['total']:.2f}" if results['total'] else ''
            ]
        }
        
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        # Sheet 2: Line Items
        if results['line_items']:
            df_items = pd.DataFrame(results['line_items'])
            df_items['amount'] = df_items['amount'].apply(lambda x: f"${x:.2f}")
            df_items.to_excel(writer, sheet_name='Line Items', index=False)
        
        # Sheet 3: Raw Text
        df_raw = pd.DataFrame({'Raw Extracted Text': [results['raw_text']]})
        df_raw.to_excel(writer, sheet_name='Raw Text', index=False)
    
    print(f"\n✅ Data saved to {output_filename}")
    
    # Display summary
    print("\n=== EXTRACTED DATA SUMMARY ===")
    print(f"Hotel Name: {results['hotel_name']}")
    print(f"Guest Name: {results['guest_name']}")
    print(f"Room Number: {results['room_number']}")
    print(f"Invoice Number: {results['invoice_number']}")
    print(f"Check-in: {results['check_in_date']}")
    print(f"Check-out: {results['check_out_date']}")
    print(f"\nFinancial Summary:")
    print(f"  Subtotal: ${results['subtotal']:.2f}")
    print(f"  Tax: ${results['tax']:.2f}")
    print(f"  Total: ${results['total']:.2f}")
    print(f"\nLine Items Found: {len(results['line_items'])}")

# Main execution for Colab
if __name__ == "__main__":
    from google.colab import files
    
    print("Please upload your hotel bill image...")
    uploaded = files.upload()
    
    image_files = list(uploaded.keys())
    if image_files:
        image_path = image_files[0]
        print(f"\nProcessing: {image_path}")
        
        # Process the image and extract data
        results = process_hotel_bill(image_path)
        
        # Save to Excel
        output_filename = 'hotel_bill_data.xlsx'
        save_to_excel(results, output_filename)
        
        # Download the Excel file
        files.download(output_filename)
        
        print("\n✅ Processing completed! Excel file has been downloaded.")
    else:
        print("No files uploaded.")
