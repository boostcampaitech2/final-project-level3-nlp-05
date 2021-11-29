import json
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from pyvirtualdisplay import Display
from datetime import datetime, timedelta
from tqdm import tqdm

class CrawlingNaverNews:
    base_url = 'https://news.naver.com/main/ranking/popularDay.naver?date='
    
    def __init__(self):
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

    def get_date_list(self):
        date_list = []
        today = datetime.today()
        for i in range(5):
            date = today - timedelta(days=i)
            date_list.append(date.strftime("%Y%m%d"))
        return date_list

    def get_naver_ranking_news_title(self):
        json_result = dict()

        date_list =self.get_date_list()
        for date in date_list:
            print(date)
            json_result[date] = []
            link_list = self._get_article_links(date)
            for link in tqdm(link_list, total=len(link_list)):
                info = self._get_article_info(link, date)
                if len(info) :
                    json_result[date].append(info)
        
        with open(f"naver_titles.json", "w", encoding="utf-8") as f:
            json.dump(json_result, f, ensure_ascii=False)
            

    def _get_article_links(self, date):
        self.driver.get(self.base_url+date)
        self._driver_wait()
        
        bsObject = BeautifulSoup(self.driver.page_source, "html.parser")
        content = bsObject.select('._officeCard0 > .rankingnews_box > .rankingnews_list > li > a')
        link_list = [link['href'] for link in content]

        return link_list

    def _get_article_info(self, link, origin_date):
        info = dict()
        url = 'https://news.naver.com' + link

        self.driver.get(url)
        self._driver_wait()

        bsObject = BeautifulSoup(self.driver.page_source, "html.parser")
        
        title = bsObject.select('#articleTitle')[0].text
        date = bsObject.select('.sponsor > .t11')[0].text
        categories =[ span.text for span in  bsObject.select('.guide_categorization_item')]
        date = date[:10].replace('.','-')

        if origin_date == date.replace("-",""):
            info['title'] = title
            info['categories'] = categories
            info['publish_date'] = date

        return info


def main():
    crawling_obj = CrawlingNaverNews()
    crawling_obj.get_naver_ranking_news_title()

if __name__ == "__main__":
    main()