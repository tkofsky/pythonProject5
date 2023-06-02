import requests
import re
import PyPDF2
import fitz,os
import csv
import time
import pytesseract
from pdf2image import convert_from_path
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
pdfcount=0
fndcount = 0
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
    #match = "1710147001"
    start_time = time.time()
    thelink = "https://madrid.wipo.int/documentaccess/documentAccess?docid="+match
    response = requests.get(thelink)
    pdfdata = response.content

    with open("thepdf.pdf", 'wb') as f:
        f.write(pdfdata)
        print (match)
        text = pdf_ocr("thepdf.pdf")
        pdfcount +=1
        fnd = text.find('12(1)(d)')
        if fnd!=-1:
            rpattern = r"TMA\d+"  # Matches exactly 10 digits
            rpattern = r"\d{7}"  # Matches exactly 10 digits
            ind = text.find("Filing date and number")
            if ind==-1:
                ind = text.index("Date et num")
            if ind!=-1:
                endind = text.find("Date and signature")
                if endind==-1:
                    endind = text.find("Date et signature")
                    print ('**FR')
                if endind!=-1:
                    segtext = text[ind:endind]
                    rmatch= re.findall(rpattern, segtext)
                    fndcount = fndcount+len(rmatch)
                    if len(rmatch)!=0:
                        ####

                        ###
                        print ('***number of matches for 7 digits is',len(rmatch))
                        print ("total up to now",fndcount)
    #text = pdf_to_text("thepdf.pdf")
        end_time = time.time()
        file = open('xlog4.txt', 'a')
        xtime = end_time - start_time
        file.write(str(len(text))+","+ str(xtime)+","+ match+"\n")
        print (len(text),xtime,match)

    print ("pdf#",pdfcount)



