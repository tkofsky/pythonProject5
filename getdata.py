import requests, time
import xml.etree.ElementTree as ET
import fitz, os
# pip uninstall fitz
# pip install PyMuPDF
from datetime import datetime
from bs4 import BeautifulSoup

#
def get_text(filepath):
    text = ""
    try:
        with fitz.open(filepath) as doc:
            text = ""
            for page in doc:
                text += page.get_text().strip()
    except:
        print("An exception occurred")
    return text


# ####Replace <YOUR_API_KEY> with your actual API key
api_key = "lMpMjH1JGCTGZ5JWrQCsL5Vs9wkmcTaE"

#api_key = "Y9qefSMPTiXqLchqz6Cu5qIumaRPvg4P"

# Define the URL for the TSDR API #
url = "https://tsdrapi.uspto.gov/ts/cd/casedocs/bundle.xml"

with open('input.txt', 'r',encoding="utf8") as file:
    for line in file:
        # Remove the newline character at the end of the line
        line = line.rstrip()

        # Split the string by space
        split_string = line.split()

        # Get the second value (index 1) of the split string
        sn = split_string[1]
        ####################################
        pub = '1'
        if sn.startswith("79") and pub == '0':
            date = split_string[3]
        else:
            date = split_string[2]

        try:
            parsed_date = datetime.strptime(date, "%Y%m%d")
        except:
            date = split_string[2]
            parsed_date = datetime.strptime(date, "%Y%m%d")

        fromDate = str(parsed_date).split()[0]
        # Print the second value
        print(sn)
        print(fromDate)

        # Define the query parameters
        params = {
            "sn": sn,
            "fromDate": fromDate,
            "toDate": "2024-12-31",
            "fields": "all",
            "perPage": 100,
            "startPage": 1,
            "sortOrder": "desc",
            "sortField": "eventDate"
        }

        # Add the API key to the headers
        headers = {
            "USPTO-API-KEY": f"{api_key}"
        }

        # Make the API request
        response = requests.get(url, params=params, headers=headers)

        xml_string = response.text.replace("<DocumentList xmlns='urn:us:gov:doc:uspto:trademark'>", "<root>").replace(
            "</DocumentList>", "</root>")

        if xml_string=="": # if null try again
            time.sleep(2)
            response = requests.get(url, params=params, headers=headers)
            xml_string = response.text.replace("<DocumentList xmlns='urn:us:gov:doc:uspto:trademark'>",
                                               "<root>").replace("</DocumentList>", "</root>")



        root = ET.fromstring(xml_string)


        # Find all UrlPath tags
        upath_tags = root.findall('.//UrlPath')

        date_obj = datetime.strptime(fromDate, '%Y-%m-%d')
        formatted_date = date_obj.strftime('%B-%d-%Y')

        # Filter First office-action link from xml
        link_list = []
        for tag in upath_tags:

            try:
                if '/OOA' in tag.text: #if 'office-action' in tag.text:
                    link_list.append(tag.text)
                    break
            except:
                print("An exception occurred")
        if len(link_list) == 0:
            for tag in upath_tags:
                try:
                    if 'office-action' in tag.text: #79338872 may want to flip /ooa with office  if '/OOA' in tag.text:
                        link_list.append(tag.text)
                        break
                except:
                    print("An exception occurred")

        if len(link_list) > 0:          ############## get the link from either office action or oa
            try:
                print(link_list[0])
                response1 = requests.get(link_list[0])
                pdf_data = response1.content
                pdf_name = 'output/' + sn + '.pdf'
                if '/OOA' in link_list[0]:
                    soup = BeautifulSoup(pdf_data, 'html.parser')
                    # Extract the text from the soup object
                    pdf_data = soup.get_text()
                    text = pdf_data
                else:
                    with open(pdf_name, 'wb') as f:
                        f.write(pdf_data)
                    text = get_text(pdf_name)
            except:
                print("An exception occurred")

            pub = '1'
            if sn.startswith("79") and pub == '0':       ##### from pdf or html to txt file

                file1 = open(r'output/' + sn + '.txt', 'w', encoding="utf-8")

                if text == "":
                    try:
                        response1 = requests.get(link_list[0])
                        pdf_data = response1.content
                        pdf_name = 'output/' + sn + '.pdf'
                        if '/OOA' in link_list[0]:
                            soup = BeautifulSoup(pdf_data, 'html.parser')
                            # Extract the text from the soup object
                            pdf_data = soup.get_text()
                            text = pdf_data
                        else:
                            with open(pdf_name, 'wb') as f:
                                f.write(pdf_data)
                            text = get_text(pdf_name)
                    except:
                        print("An exception occurred")

                #file1.writelines(text)
                file1.writelines(formatted_date + "***" + text)
            else:
                file1 = open(r'output/' + sn + '.txt', 'w', encoding="utf-8")

                if text == "":
                    try:
                        response1 = requests.get(link_list[0])
                        pdf_data = response1.content

                        pdf_name = 'output/' + sn + '.pdf'
                        if '/OOA' in link_list[0]:
                            soup = BeautifulSoup(pdf_data, 'html.parser')
                            # Extract the text from the soup object
                            pdf_data = soup.get_text()
                            text = pdf_data
                        else:
                            with open(pdf_name, 'wb') as f:
                                f.write(pdf_data)
                            text = get_text(pdf_name)
                    except:
                        print("An exception occurred")

                file1.writelines(text)

            if os.path.exists(pdf_name):
                os.remove(pdf_name)
                print(f"{pdf_name} has been deleted.")
            else:
                print(f"{pdf_name} does not exist.")
        time.sleep(1)
