import requests
import re
import PyPDF2
import fitz,os
import pytesseract
from pdf2image import convert_from_path
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
def pdf_to_text(filepath):
    text = ""
    try:
        with fitz.open(filepath) as doc:
            text = ""
            for page in doc:
                text += page.get_text().strip()
    except:
        print("An exception occurred")
    return text

def pdf_ocr(pdf_path):
    images = convert_from_path(pdf_path,poppler_path=r"C:\poppler\bin")
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

file = open('refusals.txt', 'r',encoding='utf-8')
txt = file.read()
file.close()
pattern = r"\d{10}"  # Matches exactly 10 digits
matches = re.findall(pattern,txt)

print (matches)




for match in matches:

    thelink = "https://madrid.wipo.int/documentaccess/documentAccess?docid="+match
    response = requests.get(thelink)
    pdfdata = response.content

    with open("thepdf.pdf", 'wb') as f:
        f.write(pdfdata)
        print (match)
        text = pdf_ocr("thepdf.pdf")
    #text = pdf_to_text("thepdf.pdf")
    print (text)


