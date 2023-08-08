from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Chrome()  # or webdriver.Chrome(), depending on your browser

driver.get('https://www.nba.com/stats/leaders')

# Wait for the page to load
WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))

# Get all the elements on the page
#elements = driver.find_elements(By.CSS_SELECTOR, '*')
elements = driver.find_elements(By.CSS_SELECTOR, '*')
#elements = driver.find_elements_by_css_selector('*')

# Add a hover effect to each element
for element in elements:
    driver.execute_script("""
        var element = arguments[0];
        element.addEventListener('mouseover', function() {
            element.style.backgroundColor = '#0f0';  // Change the background color to red
        });
        element.addEventListener('mouseout', function() {
            element.style.backgroundColor = '';  // Reset the background color
        });
    """, element)

# Keep the browser window open until the user closes it
input('Press ENTER to close the automated browser')

driver.quit()
