import requests
import pickle
import sys
import re
import json
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from pyvirtualdisplay import Display
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')
sys.setrecursionlimit(8000)

class CrawlingWikitree:
    categories = {
        '경제':'68',
        '엔터':'54',
        '라이프':'79',
        '스포츠':'11',
        '사회':'55',
        '정치':'4',
        '문화':'88',
        '월드':'90',
    }
    base_url = 'https://www.wikitree.co.kr'
    
    def __init__(self, click_cnt):
        self.click_cnt = click_cnt
        self.driver = self._get_driver()
        
    def _get_driver(self):
        '''
        selenium
        '''
        display = Display(visible=0, size=(1920, 1080))
        display.start()
        
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')

        driver = webdriver.Chrome('/opt/ml/shkim/chromedriver', chrome_options=chrome_options)
        return driver

    def _driver_wait(self):
        WebDriverWait(self.driver, 2).until(
            expected_conditions.invisibility_of_element((By.CSS_SELECTOR, "__initial_loading"))
        )

    def generate_article_json(self, category):
        self.category = category
        self.driver.get(self.base_url + '/categories/' + self.categories[self.category])
        self._driver_wait()
        
        for _ in range(self.click_cnt):
            self.driver.find_element_by_class_name('more_btn').click()
            self.driver.implicitly_wait(4)

        bsObject = BeautifulSoup(self.driver.page_source, "html.parser")

        content = bsObject.select('#content > .section > .list_card_4 > div:first-child li')

        json_result = []
        for li in tqdm(content, total=len(content)):
            href = li.find("a")['href']
            info = self._get_article(href)
            json_result.append(info)
        
        with open(f"article_{self.category}.json", "w", encoding="utf-8") as f:
            json.dump(json_result, f, ensure_ascii=False)

    def _get_article(self, href):
        info = dict()
        article_id = href.split('/')[-1]
        url = self.base_url + href

        self.driver.get(url)
        self._driver_wait()

        bsObject = BeautifulSoup(self.driver.page_source, "html.parser")
        
        title = bsObject.select('#article')[0].text
        abstractive = [div.text.replace('\n', '').strip() for div in bsObject.select('.lead > div')]
        article = [p.text for p in bsObject.select('#wikicon > p') if p.text != '']
        article = [{'index': idx, 'sentence': sentence} for idx, sentence in enumerate(article)]
        date = bsObject.select('.date_time')[0].text

        info['category'] = self.category
        info['id'] = article_id
        info['publish_date'] = date
        info['extractive'] = [0] if len(article) > 0 else []
        info['abstractive'] = abstractive
        info['title'] = title
        info['article'] = article

        return info

def main():
    crawling_obj = CrawlingWikitree(click_cnt=1)
    categories = crawling_obj.categories.keys()
    for category in categories:
        print(f"generate {category} articles...")
        crawling_obj.generate_article_json(category=category)


if __name__ == "__main__":
    main()
