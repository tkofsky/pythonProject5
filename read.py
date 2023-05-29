import requests
import re
import PyPDF2


def pdf_to_text(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        text = ''
        for page in range(reader.numPages):
            text += reader.getPage(page).extractText()

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
    text = pdf_to_text("thepdf.pdf")
    print (text)


