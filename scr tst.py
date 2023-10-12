from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.common.action_chains import ActionChains

from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
from PIL import Image
import time
from selenium.webdriver.support.ui import Select
#options = webdriver.ChromeOptions()

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--user-agent="Mozilla/5.0 (Windows Phone 10.0; Android 4.2.1; Microsoft; Lumia 640 XL LTE) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.135 Mobile Safari/537.36 Edge/12.10166"')


driver = webdriver.Chrome()

#options.add_argument('--ignore-certificate-errors')
#options.add_argument("--test-type")
#options.binary_location = ("C:\Users\User\PycharmProjects\selenium")
#driver = webdriver.Chrome(chrome_options=options)
driver.get('https://matrix.itasoftware.com/')

#aa=driver.find_elements_by_xpath("// a[contains(text(),'One Way')]").click()

elem = driver.find_element(By.XPATH,"//*[contains(text(), 'One Way')]")
elem.click()


if (driver.find_element(By.XPATH,"//*[contains(@id,'cityPair-org')]"))!=0:
    depart = driver.find_element(By.XPATH,"//*[@id='searchPanel-0']/div/table/tbody/tr[2]/td/div/div[2]/div/div/div[2]/div/div/div/input")

else:
    depart = driver.find_element(By.XPATH,"//*[contains(@id='cityPair-org')]")



depart.click()
depart.send_keys("JFK")

time.sleep(2)
elem = driver.find_element_by_xpath("//*[contains(text(), 'One-way')]")
elem.click()





if (driver.find_elements_by_xpath("//*[contains(@id,'cityPair-dest')]"))!=0:
    depart = driver.find_element_by_xpath("//*[@id='searchPanel-0']/div/table/tbody/tr[2]/td/div/div[2]/div/div/div[4]/div/div/div/input")

else:
    depart = driver.find_element_by_xpath("//*[contains(@id='cityPair-dest')]")


#if (driver.find_elements_by_xpath("//*[contains(@id,'cityPair-dest')]"))!=0:
#    dest = driver.find_element_by_xpath("//*[@id='searchPanel-0']/div/table/tbody/tr[2]/td/div/div[2]/div/div/div[4]/div/div/div/input)")

#else:
 #   dest = driver.find_element_by_xpath("//*[contains(@id='cityPair-dest')]")

#dest = driver.find_elements_by_xpath("//[input)]]]")




depart.click()
depart.send_keys("lax")

elem = driver.find_element_by_xpath("//*[contains(text(), 'One-way')]")
elem.click()

if (driver.find_elements_by_xpath("//*[contains(@id,'cityPair-outsDate-2')]"))!=0:
    depdate = driver.find_elements_by_xpath("//*[@id='searchPanel-0']/div/table/tbody/tr[2]/td/div/div[2]/div/div/div[2]/div/div/div/input")
else:
    depdate = driver.find_elements_by_xpath("//*[@id='cityPair-outDate-2']")

depdate=driver.find_element_by_xpath("//*[@id='searchPanel-0']/div/table/tbody/tr[2]/td/div/div[2]/div/div/div[9]/div[1]/div[1]/div[2]/input")

depdate.click()
depdate.send_keys("02/04/2019")

elem = driver.find_element_by_xpath("//*[contains(text(), 'One-way')]")
elem.click()

if (driver.find_elements_by_xpath("//*[@id='cityPair-outDate-2']"))!=0:
    depdate = driver.find_elements_by_xpath("//*[@id='searchPanel-0']/div/table/tbody/tr[2]/td/div/div[2]/div/div/div[2]/div/div/div/input")
else:
    depdate = driver.find_elements_by_xpath("//*[@id='cityPair-outDate-2']")

if (driver.find_elements_by_xpath("//*[@id='searchButton-0']"))!=0:
    bclick = driver.find_elements_by_id('searchButton-0')
else:
    bclick = driver.find_elements_by_id('searchButton-1')


bclick[0].click()
#originbox = driver.find_element_by_id("cityPair-orig-0")
#originbox.send_keys("JFK")


#elem=driver.find_elements_by_xpath("//*[@class='gwt-SuggestBox IR6M2QD-r-d IR6M2QD-a-l']")
#elem.click()
#elem[0].send_keys("lax")
#elem = driver.find_element_by_xpath("//*[@class='gwt-SuggestBox IR6M2QD-r-d IR6M2QD-a-l']")
#elem[0].click()
#toptab = driver.find_element_by_css_selector('gwt-SuggestBox.IR6M2QD-r-d.IR6M2QD-a-l')
#elem.send_keys("lax")


#### wait to click on time bar link
fastrack = WebDriverWait(driver, 2).until(ec.visibility_of_element_located((By.XPATH, "//*[@id='contentwrapper']/div[1]/div/div[6]/div[2]/a[2]/span")))
fastrack.click()
time.sleep(2)

#<div class="IR6M2QD-n-d">Searching for flights</div>
# wait for searching to be done to return back to modify search

#searching = WebDriverWait(driver, 2).until(ec.presence_of_element_located((By.XPATH, "//*[@id='contentwrapper']/div[1]/div/div[6]/div[4]/div[3]/a[5]")))
#searching= WebDriverWait(driver, 5).until(ec.presence_of_element_located((By.XPATH, "//*[contains(text(), 'All']")))
searching= WebDriverWait(driver, 5).until(ec.presence_of_element_located((By.PARTIAL_LINK_TEXT, "All")))


allclick=driver.find_element_by_link_text("All")
allclick.click()
time.sleep(2)

searching= WebDriverWait(driver, 5).until(ec.presence_of_element_located((By.PARTIAL_LINK_TEXT, "All")))

searching= WebDriverWait(driver, 5).until(ec.presence_of_element_located((By.PARTIAL_LINK_TEXT, "Modify search")))
aa=driver.execute_script("return document.documentElement.outerHTML")
time.sleep(2)

####------------------------GET THE DOM FOR PARSING LATER

bb = driver.find_element_by_xpath("//*[@id='contentwrapper']/div[1]/div/div[6]/div[4]/div[2]")
#bb = driver.find_element_by_xpath("//*[@id='contentwrapper']/div[1]/div/div[6]/div[4]")
action = ActionChains(driver)
#action.move_to_element(driver.find_element_by_xpath('//*[@id="contentwrapper"]/div[1]/div/div[6]/div[4]/div[2]/div/div/div[5]/div[16]/div[3]/div/div/div/div/div')).perform()
action.move_to_element(driver.find_elements_by_tag_name('input')[44])
for dnum in range (22,75):
    xd=str(74-dnum)
    #time.sleep(2)
    print (xd)

    action.move_to_element(driver.find_element_by_xpath('//*[@id="contentwrapper"]/div[1]/div/div[6]/div[4]/div[2]/div/div/div[5]/div['+xd+']/div[3]/div/div/div/div/div')).perform()
    #action.move_to_element(driver.find_elements_by_tag_name('input')[55])
bb = driver.find_element_by_xpath("//*[@id='contentwrapper']/div[1]/div/div[6]/div[4]/div[2]")
print (bb.text)
#################################################

###################Go back to Search Page##############
allclick=driver.find_element_by_link_text("Modify search")
allclick.click()
###############################
#searching.click()
#buttonshow = WebDriverWait(driver, 20).until(ec.presence_of_element_located((By.XPATH, "//*[@id='contentwrapper']/div[1]/div/div[96]/div[4]/div[2]/div/div/div[5]/div[1]/div[1]/button")))
#elem = driver.find_element_by_xpath("//*[@id='contentwrapper']/div[1]/div/div[1]/div[1]/table/tbody/tr/td[2]/div/a")
#elem.click()


###  //*[@id="cityPair-orig-1"]
#if (driver.find_elements_by_xpath("//*[contains(@id,'cityPair-org')]"))==0:
 #   depart = driver.find_element_by_xpath("//*[@id='searchPanel-0']/div/table/tbody/tr[2]/td/div/div[2]/div/div/div[2]/div/div/div/input")
 #   depart.click()
 #   depart.send_keys("JFK")
#else:

   #$aa=driver.find_elements_by_xpath("//*[contains(@id,'cityPair-org')]"))
    #depart = driver.find_elements_by_xpath("//input")
    #depart[1].click()
    #depart[1].send_keys("BOS")

elem = driver.find_element_by_xpath("//*[contains(text(), 'One-way')]")
elem.click()

time.sleep(2)


#if (driver.find_elements_by_xpath("//*[contains(@id,'cityPair-dest')]"))!=0:
 #   depart = driver.find_element_by_xpath("//*[@id='searchPanel-0']/div/table/tbody/tr[2]/td/div/div[2]/div/div/div[4]/div/div/div/input")

#else:
#    depart = driver.find_element_by_xpath("//*[contains(@id='cityPair-dest')]")


#if (driver.find_elements_by_xpath("//*[contains(@id,'cityPair-dest')]"))!=0:
#    dest = driver.find_element_by_xpath("//*[@id='searchPanel-0']/div/table/tbody/tr[2]/td/div/div[2]/div/div/div[4]/div/div/div/input)")

#else:
 #   dest = driver.find_element_by_xpath("//*[contains(@id='cityPair-dest')]")

#dest = driver.find_elements_by_xpath("//[input)]]]")

######################################################################################################################
depart = driver.find_elements_by_tag_name('input')[31]
depart.click()
depart.send_keys("bos")

time.sleep(2)
elem = driver.find_element_by_xpath("//*[contains(text(), 'One-way')]")
elem.click()

dest = driver.find_elements_by_tag_name('input')[28]
dest.click()
dest.send_keys("dfw")

time.sleep(2)
elem = driver.find_element_by_xpath("//*[contains(text(), 'One-way')]")
elem.click()


depdate = driver.find_elements_by_tag_name('input')[36]
depdate.click()
depdate.send_keys("03/03/2019")

time.sleep(2)
elem = driver.find_element_by_xpath("//*[contains(text(), 'One-way')]")
elem.click()

bsearch = driver.find_elements_by_tag_name('button')[0]
bsearch.click()


#### wait then click on TIMEBAR
fastrack = WebDriverWait(driver, 2).until(ec.visibility_of_element_located((By.XPATH, "//*[@id='contentwrapper']/div[1]/div/div[6]/div[2]/a[2]/span")))
fastrack.click()
time.sleep(2)

###Wait and then click on ALL
searching= WebDriverWait(driver, 5).until(ec.presence_of_element_located((By.PARTIAL_LINK_TEXT, "All")))
allclick=driver.find_element_by_link_text("All")
allclick.click()
time.sleep(2)

###Wait and then click on Modify Search
searching= WebDriverWait(driver, 5).until(ec.presence_of_element_located((By.PARTIAL_LINK_TEXT, "Modify search")))
aa=driver.execute_script("return document.documentElement.outerHTML")
time.sleep(2)

####------------------------GET THE DOM FOR PARSING LATER
bb = driver.find_element_by_xpath("//*[@id='contentwrapper']/div[1]/div/div[6]/div[4]/div[2]/div")

##bb = driver.find_element_by_xpath("//*[@id='contentwrapper']/div[1]/div/div[6]/div[4]")
print (bb.text)



###################Go back to Search Page##############
allclick=driver.find_element_by_link_text("Modify search")
allclick.click()
###############################