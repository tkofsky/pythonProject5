import os
import re
from pathlib import Path
import sys
import imaplib
import quopri
import getpass
import email
import email.header
import datetime
from dateutil.relativedelta import relativedelta
from bs4 import BeautifulSoup
import os
import calendar
#####1201121212zzz99
htmldir = 'c:\weekly tsdrtxt'
filename = 'refusalforeign.txt'
thehtmlfile = os.path.join(htmldir, filename)

htmldir = 'c:\weekly tsdrtxt'
filename = 'refusaldomestic.txt'
thehtmlfilerefused = os.path.join(htmldir, filename)
cnt=0
xcount = 0
#1222290032312
fp = open(thehtmlfile, 'w')
fprefused = open(thehtmlfilerefused, 'w')

fp.write(
    "Refusal Type" + "|" + "Serial Number" + "|" + "Date of Refusal" + "|" + "Due Date" + "|" + "Trademark" + "|" + "Applicant" + "|" + "Correspondent" + "|" + "Address" + "|" + "Address" + "|" + "Address" + "|" + "Address" + "|" + "email" + "\n")
fprefused.write(
    "Refusal Type" + "|" + "Serial Number" + "|" + "Date of Refusal" + "|" + "Due Date" + "|" + "Trademark" + "|" + "Applicant" + "|" + "Correspondent" + "|" + "Address" + "|" + "Address" + "|" + "Address" + "|" + "Address" + "|" + "email" + "\n")

notusa = 0

#usethisdate = "December-05-2022"
usethisdate = ""
#

for filename in os.listdir(htmldir):
    if filename.endswith(".txt") and "refusal" not in filename:
        #filename="90131713.txt"

        print(filename)
        xcount = xcount + 1
        thehtmlfile = os.path.join(htmldir, filename)

        thehtmlfile = os.path.join(htmldir, filename)
        thetextfile =  open(thehtmlfile, "r",encoding="utf8")
       ####################MADRID DATE USE IF NO DATE
        thetext = thetextfile.read()
        index = thetext.find("***")
        usethisdate = thetext[0:index]
       ###############################################
        # get date using tr td
        # table = soup.find_all('table')
        # rows = table[3].find_all('tr')
        # for row in rows:a

        othertext = "prior-filed"  # sometimes it says no pending or reg, but then it saysHowever, a mark in a prior-filed pending application may present a bar to registration of applicant’s mark.

        noconflict = "database of registered and pending marks and has found no"  # found no conflicting marks"# that would bar registration under Trademark Act Section 2(d)"
        noconflicttwo = "database of registered and pending marks and found no"  # found no conflicting marks"# that would bar registration under Trademark Act Section 2(d)"
        noconflictfour = "database of registered and pending marks andfound no"  # found no conflicting marks"# that would bar registration under Trademark Act Section 2(d)"
        nonconflictthree = "no similar registered or pending mark"

        xnoconflict = "databaseofregisteredandpendingmarksandhasfoundno"  # found no conflicting marks"# that would bar registration under Trademark Act Section 2(d)"
        xnoconflicttwo = "databaseofregisteredandpendingmarksandfoundno"  # found no conflicting marks"# that would bar registration under Trademark Act Section 2(d)"
        xnoconflictfour = "databaseofregisteredandpendingmarksandfoundno"  # found no conflicting marks"# that would bar registration under Trademark Act Section 2(d)"
        xnonconflictthree = "nosimilarregisteredorpendingmark"


        # registered and pending marks and found no
        #registered and pending marks and found no
        # noconflict="found no"
        desciptionflag = ''
        desciptionflagtwoe = ""
        desciptionflagtwod = ""
        desciptionflagcbd = ""
        desciptionflagdc = ""
        spacesdesciptionflagtwod = ""

###########################SERIAL NUMBER
        sn = filename
        sn = sn.replace(".txt","")
        serialnumber = sn
##########################MARK##########################
        print (sn)
        themark=""
        try:
            findmarkindex = thetext.index("\nMark:")
            findmarkindex_end = thetext.index("\n",findmarkindex+1)
            mark = thetext[findmarkindex+6:findmarkindex_end:1]
            themark = mark.strip()
        except:
            print ("no mark")

        tmptext = thetext
        thetext = thetext.replace("\n"," ")
        thetext = thetext.replace("  ", " ")
        #**** possible all that is needed is to get rid of "marksand" to "marks and"

        if "section 2(e)(2) refusal" in (thetext.lower()):
            desciptionflagtwoe = "2(e)"
        if "section 2(e)(1) refusal" in (thetext.lower()):
            desciptionflagtwoe = "2(e)"
            # print ("T")
        if "merely descriptive refusal" in (thetext.lower()):
            desciptionflagtwoe = "2(e)"

        if "section 2(d" in (thetext.lower()):
            if noconflict in (thetext.lower()) or  noconflictfour in (thetext.lower()) or noconflicttwo in (thetext.lower()) or nonconflictthree in (
            thetext.lower()):  # if 2d happens top be there because due to the phrase above then
                desciptionflagtwod = ""
                # print ("*****")
            else:
                desciptionflagtwod = " 2(d)"
                spacesdesciptionflagtwod = " 2(d)"

        textnospace = thetext.replace(" ", "")
        if "section2(d" in (textnospace.lower()):
            if xnoconflict in (textnospace.lower()) or xnoconflictfour in (textnospace.lower()) or xnoconflicttwo in (
            textnospace.lower()) or xnonconflictthree in (
                    textnospace.lower()):  # if 2d happens top be there because due to the phrase above then
                desciptionflagtwod = ""
                # print ("*****")
            else:
                desciptionflagtwod = " 2(d)"

        if spacesdesciptionflagtwod!=desciptionflagtwod:
            print ("****")
            cnt+=1
            print (cnt)

        if "the filing date of pending" in (thetext.lower()):
            desciptionflagtwod = " 2(d)"

        if "prior-filed" in (thetext.lower()) or "prior filed" in (thetext.lower()):
            desciptionflagtwod = " 2(d)"

        if "agriculture improvement act" in (thetext.lower()):
            desciptionflagcbd = "cbd"

        if "csa" in (thetext.lower()):
            desciptionflagcbd = "cbd"

        if "fdca" in (thetext.lower()):
            desciptionflagcbd = "cbd"

        if "disclaimer required" in (thetext.lower()):
            desciptionflagdc = "dc"
        if "requirement:disclaimer" in (thetext.lower()):
            desciptionflagdc = "dc"
        if "requirement: disclaimer" in (thetext.lower()):
            desciptionflagdc = "dc"
        if "applicant must disclaim" in (thetext.lower()):
            desciptionflagdc = "dc"



        #####################################################################################
        deadlineperiod = 3
        if "response to this letter within six months" in (thetext.lower()):
            deadlineperiod = 6
        if "response within six months" in (thetext.lower()):
            deadlineperiod = 6
        if "within three months" in (thetext.lower()):
            deadlineperiod = 3



        thedate = ""
        theapplicant = ""
        theemail = ""
        email = ""
        thetext = tmptext
        xdate = thetext

        match = re.search('\w{3,9}?\s\d{1,2}?,\s\d{4}?', xdate, re.IGNORECASE)
        match = re.search('\w{3,9}?\s\d{2}?,\s\d{4}?', xdate, re.IGNORECASE)
        if match:
            thedate = match.group(0)
            tmpdate = thedate.replace(" ", "-")
            thedate = tmpdate.replace(",", "")
            if thedate.find("USP") != -1:
                thedate = ''

        # if thedate.find()
        # thedate="Dec-11-2020"
        #####  thedate = datetime.datetime.strptime(thedate, "%B-%d-%Y")
        if thedate == '':
            thedate = "-"#usethisdate in cases when the doc/refusal starts without header

        if serialnumber[0:2]=='79': # us the *** date there may be others below but not the real date
            thedate = usethisdate

        uszip = "no"
        xzip = thetext
        # match = re.search('[a-zA-Z]{2} \d{5}([-]|\s*)?(\d{4})?$', xzip, re.IGNORECASE)
        xzip = xzip.replace("\n", " ")
        xzip = xzip.replace("\xa0", " ")
        xzip = xzip.replace(",", " ")
        xzip = xzip.replace("  ", " ")

        # **********************check for 2 letter state plus 5-4 zip digits
        ###### example: NY 10020
        xzip = xzip.strip()
        uszip = "no"
        regexx = "[[?<=,\s*][a-zA-Z]{2}\s*\d{5}([-]|\s*)?(\d{4})?"
        regexy = "[a-zA-Z]{2}\s*\d{5}([-]|\s*)?(\d{4})?"
        match = re.search(regexx, xzip, re.IGNORECASE)
        if match:
            thezip = match.group(0)
            uszip = "yes"

        # thedate=soup.find("div", {"id": "docDateField"})

        #emailname = soup.find_all(attrs={"name": "email"})
        #######################################EMAIL
        try:
            findemailindex = thetext.index("Correspondence Email Address:")
            findemailindex_end = thetext.index("\n", findemailindex + 1)
            emailname = thetext[findemailindex + 30:findemailindex_end:1]
            emailname = emailname.strip()
            email = emailname
            email = email.replace('NONFINAL OFFICE ACTION',"")
        except:
            print ("no email field")
        ############################################ ADDRESS

        fulladdr=""
        try:
            findaddrindex = thetext.index("Correspondence Address:")
            findaddrindex_end = thetext.index("Applicant:", findaddrindex + 1)
            fulladdr = thetext[findaddrindex + 24:findaddrindex_end:1]
            fulladdr = fulladdr.strip()
            fulladdr = fulladdr.replace("UNITED STATES","")

            ad = fulladdr.split("\n")
            addr1 = ""
            addr2 = ""
            addr3 = ""
            addr4 = ""
            addr5 = ""
            if ad[0]:
                addr1 = ad[0]
            if ad[1]:
                addr2 = ad[1]
            if ad[2]:
                addr3 = ad[2]
            if ad[3]:
                addr4 = ad[3]
            if ad[4]:
                addr5= ad[4]

        except:
            print ("no addr field")

        ############################################APPLICANT
        try:
            findappindex = thetext.index("Applicant:")
            findappindex_end = thetext.index("\n", findappindex + 1)
            theapplicant = thetext[findappindex + 9:findappindex_end:1]
            theapplicant = theapplicant.strip()
        except:
            print ("no applicant field")

        if themark == "":
            themark = "Design Mark"

        theapplicant = theapplicant.replace(":","")
        theapplicant = theapplicant.strip()
        appcheck = theapplicant.lower()

        fulladdr = fulladdr.lower()
        ################################################
        uszip = "no"
        xzip = fulladdr

        regexx = "[[?<=,\s*][a-zA-Z]{2}\s*\d{5}([-]|\s*)?(\d{4})?"

        # regexy = "[a-zA-Z]{2}\s*\d{5}([-]|\s*)?(\d{4})?"
        fixedzip = xzip.replace("\xa0", " ")
        fixedzip = xzip.replace(",", " ")
        fixedzip = fixedzip.replace("  ", " ")



        match = re.search(regexx, fixedzip, re.IGNORECASE)
        if match:
            thezip = match.group(0)
            uszip = "yes"
        else:
            notusa += 1
            print(fulladdr)

        #############################################################

        appcheck = appcheck.replace("ltd", "")
        appcheck = appcheck.replace("inc.", "")
        appcheck = appcheck.replace("limited", "")
        appcheck = appcheck.replace("gmbh", "")
        appcheck = appcheck.replace("corporation", "")
        appcheck = appcheck.replace(" corp", "")
        appcheck = appcheck.replace("llc", "")
        appcheck = appcheck.replace("shenzhen", "")

        appcheck = appcheck.replace(",", "")
        appcheck = appcheck.replace(",", "")
        appcheck = appcheck.replace(",", "")
        appcheck = appcheck.replace(".", "")
        if "stiles" in appcheck.lower():
            print()
        xapp = appcheck
        appcheck = appcheck.split(" ")

        prose = "no"
        prosefulladdr = fulladdr

        for wordpart in appcheck:
            # print (appcheck,prosefulladdr)

            if wordpart.lower() in prosefulladdr.lower():
                print(xapp)
                print(prosefulladdr.lower())
                if len(wordpart) > 3:
                    if "llp" in prosefulladdr.lower():  # check if firm
                        print("xxxxxx")
                        print(prosefulladdr)
                    else:
                        prose = "yes"
                    break
                else:
                    print(" ")

        if email.find("trademark") != -1 or email.find("legal") != -1 or email.find("law") != -1 or email.find(
                "ip@") != -1:
            prose = "no"

        #### six months after
        if thedate == "USPQ2d-71-1475":
            print("aaaaaaaaa")

        try:
            date = datetime.datetime.strptime(thedate, "%B-%d-%Y")

            month = date.month - 1 + deadlineperiod # 6 or 3
            year = date.year + month // 12
            month = month % 12 + 1
            day = min(date.day, calendar.monthrange(year, month)[1])
            dtestring = str(month) + "-" + str(day) + "-" + str(year)
            tmpdate = datetime.datetime.strptime(dtestring, "%m-%d-%Y")

            # sixmonths=datetime.dtestring(year, month, day)
            mm = tmpdate.strftime("%b")
            yy = tmpdate.strftime("%Y")
            dd = tmpdate.strftime("%d")
            sixmonths = mm + "-" + dd + "-" + yy
        except:
            print("err")
            sixmonths = ""
        #####

        refusalstr = desciptionflagtwod + desciptionflagtwoe + desciptionflagcbd + desciptionflagcbd + desciptionflagdc  # needs to have at least one refusal of 2d or 2e or cbd
        if "china " in fulladdr.lower():
            uszip = "no"
            # print("\n" + str(serialnumber) + "|" + str(themark) + "|" + str(addr1) + " " + str(addr2) + " " + str(addr3) + str(addr4) + str(addr5) + "|" + str(email) + "|" + str(theapplicant) + "|" + str(thedate))
        if "puerto rico" in fulladdr.lower():
            uszip = "yes"

        if uszip == "no" and len(refusalstr) > 0:
            # showrow="yes"
            # fp.write(str(serialnumber) + "|" + str(thedate) + "|" + str(theapplicant) + "|" + str(themark) + "|" + str(addr1) + " " + str(addr2) + " " + str(addr3) + " " + str(addr4) + " " + str(addr5) + "|" + str(email)+"\n")
            print (filename)
            fp.write(
                desciptionflagtwod + "," + desciptionflagtwoe + "," + desciptionflagcbd + "," + desciptionflagdc + "|" + serialnumber + "|" + str(
                    thedate) + "|" + str(sixmonths) + "|" + str(themark) + "|" + str(theapplicant) + "|" + str(
                    addr1) + "|" + str(addr2) + "|" + str(addr3) + "|" + str(addr4) + "|" + str(addr5) + "|" + str(
                    email) + "\n")
            # fp.write("\n"+str(serialnumber)+"|"+str(themark)+"|"+str(addr1)+" "+str(addr2)+" "+str(addr3)+str(addr4)+" "+str(addr5)+"|"+str(email)+"|"+str(theapplicant)+"|"+str(thedate))
            print(
                desciptionflagtwod + "," + desciptionflagtwoe + "," + desciptionflagcbd + "," + desciptionflagdc + "|" + serialnumber + "|" + str(
                    thedate) + "|" + str(sixmonths) + "|" + str(themark) + "|" + str(theapplicant) + "|" + str(
                    addr1) + "|" + str(addr2) + "|" + str(addr3) + "|" + str(addr4) + "|" + str(addr5) + "|" + str(
                    email) + "\n")

        if prose == "yes" and uszip == "yes" and len(refusalstr) > 0:
            fprefused.write(
                desciptionflagtwod + "," + desciptionflagtwoe + "," + desciptionflagcbd + "," + desciptionflagdc + "|" + serialnumber + "|" + str(
                    thedate) + "|" + str(sixmonths) + "|" + str(themark) + "|" + str(theapplicant) + "|" + str(
                    addr1) + "|" + str(addr2) + "|" + str(addr3) + "|" + str(addr4) + "|" + str(addr5) + "|" + str(
                    email) + "\n")

            print(
                desciptionflagtwod + "," + desciptionflagtwoe + "," + desciptionflagcbd + "|" + serialnumber + "|" + str(
                    thedate) + "|" + str(sixmonths) + "|" + str(themark) + "|" + str(theapplicant) + "|" + str(
                    addr1) + "|" + str(addr2) + "|" + str(addr3) + "|" + str(addr4) + "|" + str(addr5) + "|" + str(
                    email) + "\n")

print (cnt)