if __name__ == '__main__':
    import time
    import undetected_chromedriver as uc
    uc.options.binary_location ="chromedriver.exe"
    #driver = uc.Chrome(headless=False,use_subprocess=False)
    driver = uc.Chrome(headless=False,use_subprocess=False)

   # driver.get("https://opensea.io/rankings/trending/")
    driver.get("https://www5.services.mrq.gouv.qc.ca/MrqAnonyme/BR/BR01/BR01A1_01A_ConsulterBNR_PC/P_Consultation.aspx?CLNG=A#Recherche1_K1Ancre1")
    time.sleep(1000)