#!/bin/bash

# 1. crawling

# 1.1 news title and url crawling
python ./crawling/daum_news_title_crawling.py --categories "society" "economic"

# 1.2 news content crawling
#python ./crawling/daum_news_crawling_multiprocessing.py --category society --page_count 50 --max_page 1000
python ./crawling/daum_news_crawling_multiprocessing.py --category economic --page_count 50 --max_page 1000

# 2. clustering


# 3. summarization