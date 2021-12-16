import os
import argparse
from datetime import date, timedelta
import json
from selenium import webdriver
from bs4 import BeautifulSoup
from pyvirtualdisplay import Display
from tqdm import tqdm
import platform
import warnings

warnings.filterwarnings("ignore")

class CrawlingDaumNews:
    categories = {
        '사회':'society',
        '정치':'politics',
        '경제':'economic',
        '국제': 'foreign',
        '문화':'culture',
        '연예':'entertain',
        '스포츠':'sports',
        'IT':'digital'
        }
    
    def __init__(self):
        self.driver = self._get_driver()
        self.error_urls = []
        
    def _get_driver(self):
        os_type = "Mac" if platform.system() == "Darwin" else "Linux"
        chromedriver_path = ""
        if os_type == "Mac":
            chromedriver_path = "/opt/homebrew/bin/chromedriver"
        else:
            chromedriver_path = "./chromedriver"

        if os_type == "Linux":
            display = Display(visible=0, size=(1920, 1080))
            display.start()

        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(chromedriver_path, options=chrome_options)
        return driver

    def generate_article_json(self, date, category, start_page, page_count):
        save_dir = f"./data/{date}"
        file_name = f"daum_articles_{date}_{category}_{start_page:03d}.json"
        if os.path.isfile(os.path.join(save_dir, file_name)):
            print(f'{file_name} is already generated.')
            return

        title_data = self._get_title_data(date, category)
        
        json_result = []
        
        articles = [article for article in title_data['articles'] if start_page <= int(article["id"].split("_")[0]) < start_page+page_count]

        if articles:
            for article in tqdm(articles, total=len(articles)):
                info = self._get_article(category, article["url"])
                if info:
                    json_result.append(info)
        
            with open(os.path.join(save_dir, file_name), "w", encoding="utf-8") as f:
                json.dump(json_result, f, ensure_ascii=False)
            
            if self.error_urls:
                print("fail url list")
                for error_url in self.error_urls:
                    print(error_url)

    def _get_title_data(self, date, category):
        with open(f"./data/{date}/daum_titles_{date}_{category}.json", "r", encoding="utf-8") as f:
            title_data = json.load(f)
        return title_data

    def _get_article(self, category, url):
        info = dict()
        article_id = url.split('/')[-1]

        self.driver.implicitly_wait(3)
        self.driver.get(url)

        bsObject = BeautifulSoup(self.driver.page_source, "html.parser")
        
        try:
            title = bsObject.select('#cSub .tit_view')[0].text
            abstractive = bsObject.select('.summary_view')
            abstractive = abstractive[0].get_text(strip=True, separator="\n").splitlines() if abstractive else []
            article = [p.text for p in bsObject.select('#harmonyContainer > section p') if p.text != '']
            article = self._corpus_to_sentence(article)
            article = [{'index': idx, 'sentence': sentence} for idx, sentence in enumerate(article)]
            date = bsObject.select('.info_view .num_date')[0].text
            date_lst = date.replace('. ','-').split("-")
            date = "-".join(date_lst[:-1]) + " " + date_lst[-1]
        except Exception as e:
            print(f"Parsing error ({url}): {e}")
            self.error_urls.append(url)
            return info
        
        info['category'] = category
        info['id'] = article_id
        info['publish_date'] = date
        info['extractive'] = [0] if len(article) > 0 else []
        info['abstractive'] = abstractive
        info['title'] = title
        info['article'] = article

        return info

    def _corpus_to_sentence(self,article):
        splited_article = []
        for corpus in article:
            sentences = corpus.split(". ")
            for sentence in sentences:
                if sentence:
                    new_sentence = sentence + "." if sentence[-1] != "." and sentence[-1] == "다" else sentence
                    splited_article.append(new_sentence)
        return splited_article

def get_args():
    parser = argparse.ArgumentParser(description="crawling daum news titile.")
    parser.add_argument(
        "--date",
        default=(date.today() - timedelta(1)).strftime("%Y%m%d"),
        type=str,
        help="date of news"
    )
    parser.add_argument(
        "--category",
        default="all",
        type=str,
        help="category of news",
        choices=["all", "society", "politics", "economic", "foreign", "culture", "entertain", "sports", "digital"]
    )
    parser.add_argument(
        "--start_page",
        default=1,
        type=int,
        help="start page no",
    )
    parser.add_argument(
        "--page_count",
        default=10,
        type=int,
        help="page count",
    )
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    crawling_obj = CrawlingDaumNews()
    
    date = args.date
    category = {v:k for k, v in crawling_obj.categories.items()}[args.category]
    
    crawling_obj.generate_article_json(date=date, category=category, start_page=args.start_page, page_count=args.page_count)

if __name__ == "__main__":
    main()
