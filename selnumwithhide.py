if __name__ == '__main__':

    import undetected_chromedriver as uc
    uc.options.binary_location =r"chromedriver.exe"
    driver = uc.Chrome(headless=False,use_subprocess=False)


    driver.get("https://opensea.io/rankings/trending/")
