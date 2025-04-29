from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional, Union
from pydantic import BaseModel
import pytesseract
from PIL import Image
import io
import re
import cv2
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Models
class LabTest(BaseModel):
    test_name: str
    test_value: str
    test_unit: Optional[str] = ""
    bio_reference_range: Optional[str] = ""
    lab_test_out_of_range: bool = False

class LabReportResponse(BaseModel):
    is_success: bool = True
    data: List[LabTest] = []
    error: Optional[str] = None

# Create FastAPI app
app = FastAPI(title="Lab Test Extractor API",
             description="API for extracting lab test information from medical reports")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def enhance_image(image_bytes):
    """
    Enhanced preprocessing pipeline for better OCR results
    """
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to numpy array for OpenCV processing
    img_np = np.array(image)
    
    # Check if image is color or grayscale
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        # Convert to grayscale if it's color
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    
    # Apply multiple preprocessing techniques and keep the best result
    processed_images = []
    
    # Technique 1: Basic thresholding
    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(("otsu_threshold", thresh1))
    
    # Technique 2: Adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 31, 2)
    processed_images.append(("adaptive_threshold", adaptive_thresh))
    
    # Technique 3: Bilateral filtering to preserve edges while removing noise
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    _, thresh_bilateral = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(("bilateral_filter", thresh_bilateral))
    
    # Technique 4: Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, thresh_enhanced = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(("contrast_enhanced", thresh_enhanced))
    
    # Technique 5: Noise removal using morphological operations
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)
    _, thresh_opening = cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(("noise_removed", thresh_opening))
    
    # Return a dictionary of all processed images for multiple OCR attempts
    return {name: Image.fromarray(img) for name, img in processed_images}

def extract_text_from_image(processed_images):
    """
    Extract text from multiple processed versions of the image
    and return the best result based on text length
    """
    results = {}
    
    # Try different OCR configurations
    configs = [
        '--oem 3 --psm 6',  # Default mode - assume a single uniform block of text
        '--oem 3 --psm 4',  # Assume a single column of text of variable sizes
        '--oem 3 --psm 11',  # Sparse text. Find as much text as possible in no particular order
        '--oem 3 --psm 3 -l eng'   # Fully automatic page segmentation, but no OSD (default)
    ]
    
    best_text = ""
    max_length = 0
    
    # Try each image processing technique with each OCR configuration
    for img_name, img in processed_images.items():
        for config in configs:
            try:
                text = pytesseract.image_to_string(img, config=config)
                results[f"{img_name}_{config}"] = text
                
                # Keep track of the result with the most text
                if len(text) > max_length:
                    max_length = len(text)
                    best_text = text
            except Exception as e:
                logger.error(f"OCR error with {img_name} and config {config}: {str(e)}")
    
    # Also get data with layout information
    try:
        data = pytesseract.image_to_data(processed_images['adaptive_threshold'], 
                                         config='--oem 3 --psm 6', 
                                         output_type=pytesseract.Output.DICT)
    except Exception as e:
        logger.error(f"Error extracting structured data: {str(e)}")
        data = None
    
    return best_text, data

def parse_lab_tests(text, data=None):
    """
    Enhanced parser to extract lab tests from OCR text
    """
    lab_tests = []
    
    # Process the extracted text to identify potential test data
    # Split text into lines for better processing
    lines = text.split('\n')
    clean_lines = [line.strip() for line in lines if line.strip()]
    
    # Pattern 1: Test with numeric value, optional unit, and reference range
    # Example: "Hemoglobin 15.4 g/dl 13.5-18"
    pattern1 = re.compile(
        r"(?P<test_name>[A-Za-z\s\(\)\-\'/]+)\s*:?\s*"
        r"(?P<value>[<>]?\d+\.?\d*)\s*"
        r"(?P<unit>[A-Za-z%/\.]+)?\s*"
        r"(?:(?:Reference|Ref|Normal|Bio|Range).*?:?\s*(?P<range>[<>]?\d+\.?\d*\s*[-–]\s*[<>]?\d+\.?\d*))?",
        re.IGNORECASE
    )
    
    # Pattern 2: For qualitative results (Positive/Negative)
    pattern2 = re.compile(
        r"(?P<test_name>[A-Za-z\s\(\)\-\'\/]+)\s*:?\s*"
        r"(?P<value>Positive|Negative|Reactive|Non-Reactive|POSITIVE|NEGATIVE)\s*",
        re.IGNORECASE
    )
    
    # Pattern 3: Alternative format with reference range at the end
    pattern3 = re.compile(
        r"(?P<test_name>[A-Za-z\s\(\)\-\'\/]+)\s*:?\s*"
        r"(?P<value>[<>]?\d+\.?\d*)\s*"
        r"(?P<unit>[A-Za-z%/\.]+)?\s*"
        r".*?(?P<range>\d+\.?\d*\s*[-–]\s*\d+\.?\d*)\s*",
        re.IGNORECASE
    )
    
    # Process line by line to increase accuracy
    for line in clean_lines:
        # Skip short lines that are unlikely to contain test information
        if len(line) < 5:
            continue
            
        # Try all patterns
        match = pattern1.search(line)
        if not match:
            match = pattern2.search(line)
        if not match:
            match = pattern3.search(line)
            
        if match:
            test_name = match.group("test_name").strip()
            
            # Skip common headers or non-test lines
            if test_name.lower() in ["test", "test name", "name", "parameter", "investigation"]:
                continue
                
            # Get test value
            if "value" in match.groupdict() and match.group("value"):
                test_value = match.group("value").strip()
            else:
                continue  # Skip if no value found
                
            # Get unit if available
            unit = ""
            if "unit" in match.groupdict() and match.group("unit"):
                unit = match.group("unit").strip()
                
            # Get reference range if available
            ref_range = ""
            out_of_range = False
            if "range" in match.groupdict() and match.group("range"):
                ref_range = match.group("range").strip()
                
                # Determine if out of range for numeric values
                if not any(qual in test_value.lower() for qual in ["positive", "negative", "reactive"]):
                    try:
                        # Handle '<' and '>' in values
                        numeric_value = test_value
                        compare_less = False
                        compare_greater = False
                        
                        if '<' in test_value:
                            numeric_value = test_value.replace('<', '')
                            compare_less = True
                        elif '>' in test_value:
                            numeric_value = test_value.replace('>', '')
                            compare_greater = True
                            
                        value_num = float(numeric_value)
                        
                        # Parse reference range
                        range_parts = re.split(r'[-–]', ref_range)
                        if len(range_parts) == 2:
                            min_val = float(range_parts[0].strip().replace('<', '').replace('>', ''))
                            max_val = float(range_parts[1].strip().replace('<', '').replace('>', ''))
                            
                            # Determine if out of range
                            if compare_less:
                                # For '<' values, only consider below range
                                out_of_range = value_num < min_val
                            elif compare_greater:
                                # For '>' values, only consider above range
                                out_of_range = value_num > max_val
                            else:
                                # Normal comparison
                                out_of_range = value_num < min_val or value_num > max_val
                    except (ValueError, IndexError):
                        # If conversion fails, assume it's not out of range
                        out_of_range = False
            
            # Add to results
            lab_tests.append({
                "test_name": test_name,
                "test_value": test_value,
                "test_unit": unit,
                "bio_reference_range": ref_range,
                "lab_test_out_of_range": out_of_range
            })
    
    # Process tabular data if available
    if data:
        try:
            tabular_tests = extract_from_tabular_data(data)
            # Merge with existing tests, avoiding duplicates
            existing_test_names = {test["test_name"].lower() for test in lab_tests}
            for test in tabular_tests:
                if test["test_name"].lower() not in existing_test_names:
                    lab_tests.append(test)
                    existing_test_names.add(test["test_name"].lower())
        except Exception as e:
            logger.error(f"Error processing tabular data: {str(e)}")
    
    # Special handling for common tests in specific formats
    lab_tests = enhance_results_with_special_cases(text, lab_tests)
    
    return lab_tests

def extract_from_tabular_data(data):
    """
    Extract test information from tabular data
    """
    tabular_tests = []
    
    if not data or 'text' not in data:
        return tabular_tests
    
    # Create a list of lines with their positions
    lines = []
    current_line = []
    current_line_num = -1
    
    for i in range(len(data['text'])):
        if data['text'][i].strip():
            if current_line_num != data['line_num'][i]:
                if current_line:
                    lines.append(current_line)
                current_line = []
                current_line_num = data['line_num'][i]
            
            current_line.append({
                'text': data['text'][i],
                'left': data['left'][i],
                'top': data['top'][i],
                'width': data['width'][i],
                'height': data['height'][i]
            })
    
    # Add the last line if it exists
    if current_line:
        lines.append(current_line)
    
    # Process lines to identify potential rows in a table
    for line in lines:
        # Sort by left position (column order)
        sorted_line = sorted(line, key=lambda x: x['left'])
        
        # Need at least 2 elements to possibly have test and value
        if len(sorted_line) < 2:
            continue
        
        # Check if the first item could be a test name
        test_name = sorted_line[0]['text'].strip()
        if len(test_name) < 2 or test_name.isdigit():
            continue
        
        # Check if the second item could be a value
        value_text = sorted_line[1]['text'].strip()
        value_match = re.match(r'[<>]?\d+\.?\d*', value_text)
        if not value_match and not any(val in value_text.lower() for val in ["positive", "negative"]):
            continue
        
        # We have a potential test name and value
        test_value = value_text
        
        # Check for unit (usually in 3rd column)
        test_unit = ""
        if len(sorted_line) > 2:
            unit_text = sorted_line[2]['text'].strip()
            if re.match(r'[A-Za-z%/\.]+', unit_text) and len(unit_text) < 10:
                test_unit = unit_text
        
        # Check for reference range (usually in last column)
        ref_range = ""
        if len(sorted_line) > 3:
            range_text = sorted_line[-1]['text'].strip()
            range_match = re.match(r'\d+\.?\d*\s*[-–]\s*\d+\.?\d*', range_text)
            if range_match:
                ref_range = range_text
        
        # Determine if out of range
        out_of_range = False
        if ref_range and test_value and not any(qual in test_value.lower() for qual in ["positive", "negative", "reactive"]):
            try:
                numeric_value = test_value.replace('<', '').replace('>', '')
                value_num = float(numeric_value)
                
                range_parts = re.split(r'[-–]', ref_range)
                if len(range_parts) == 2:
                    min_val = float(range_parts[0].strip())
                    max_val = float(range_parts[1].strip())
                    out_of_range = value_num < min_val or value_num > max_val
            except (ValueError, IndexError):
                out_of_range = False
        
        # Add to results
        tabular_tests.append({
            "test_name": test_name,
            "test_value": test_value,
            "test_unit": test_unit,
            "bio_reference_range": ref_range,
            "lab_test_out_of_range": out_of_range
        })
    
    return tabular_tests

def enhance_results_with_special_cases(text, lab_tests):
    """
    Add special case handling for common lab tests
    """
    # Pregnancy test special handling
    pregnancy_test_match = re.search(r'(urine\s+for\s+pregnancy|pregnancy\s+test).*?(positive|negative)', 
                                    text, re.IGNORECASE)
    if pregnancy_test_match:
        test_name = "Urine For Pregnancy"
        test_value = pregnancy_test_match.group(2).upper()
        
        # Check if we already have this test
        existing = False
        for test in lab_tests:
            if "pregnancy" in test["test_name"].lower():
                existing = True
                break
                
        if not existing:
            lab_tests.append({
                "test_name": test_name,
                "test_value": test_value,
                "test_unit": "",
                "bio_reference_range": "",
                "lab_test_out_of_range": False  # Qualitative results don't have out of range
            })
    
    # Blood glucose special handling
    glucose_match = re.search(r'(glucose|sugar).*?(\d+\.?\d*)\s*(mg/dl|mmol/L)', text, re.IGNORECASE)
    if glucose_match:
        test_name = "Blood Glucose"
        test_value = glucose_match.group(2)
        test_unit = glucose_match.group(3)
        
        # Check if we already have this test
        existing = False
        for test in lab_tests:
            if "glucose" in test["test_name"].lower() or "sugar" in test["test_name"].lower():
                existing = True
                break
                
        if not existing:
            # Common reference range for random blood glucose
            ref_range = "70-140"
            
            # Determine if out of range
            out_of_range = False
            try:
                value_num = float(test_value)
                out_of_range = value_num < 70 or value_num > 140
            except ValueError:
                pass
                
            lab_tests.append({
                "test_name": test_name,
                "test_value": test_value,
                "test_unit": test_unit,
                "bio_reference_range": ref_range,
                "lab_test_out_of_range": out_of_range
            })
    
    return lab_tests

@app.post("/get-lab-tests", response_model=LabReportResponse)
async def get_lab_tests(file: UploadFile = File(...)):
    """
    Extract lab tests from a lab report image
    
    - **file**: Lab report image file (JPEG, PNG, PDF)
    
    Returns a list of detected lab tests with values and reference ranges
    """
    try:
        # Check file type
        if file.content_type not in ["image/jpeg", "image/png", "application/pdf", "image/jpg"]:
            raise HTTPException(
                status_code=400, 
                detail="Only JPEG, PNG and PDF files are supported"
            )
        
        # Read file content
        contents = await file.read()
        
        # Preprocess the image
        processed_images = enhance_image(contents)
        
        # Extract text from processed images
        text, tabular_data = extract_text_from_image(processed_images)
        
        # Log the extracted text for debugging
        logger.info(f"Extracted text: {text[:200]}...")  # Log first 200 chars
        
        # Parse the extracted text to find lab tests
        lab_tests = parse_lab_tests(text, tabular_data)
        
        # Convert to response model format
        result_tests = [
            LabTest(
                test_name=test["test_name"],
                test_value=test["test_value"],
                test_unit=test["test_unit"],
                bio_reference_range=test["bio_reference_range"],
                lab_test_out_of_range=test["lab_test_out_of_range"]
            )
            for test in lab_tests
        ]
        
        return LabReportResponse(
            is_success=True,
            data=result_tests
        )
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return LabReportResponse(
            is_success=False,
            data=[],
            error=str(e)
        )

@app.get("/")
def read_root():
    return {"message": "Welcome to Lab Test Extractor API. Use /get-lab-tests endpoint to process lab reports."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
