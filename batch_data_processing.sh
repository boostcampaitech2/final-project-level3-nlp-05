#!/bin/bash

# 1. crawling

# 1.1 news title and url crawling
echo "news title crawling..."
python ./crawling/daum_news_title_crawling.py --date 20211214 --categories "all"

# 1.2 news content crawling
echo ""
echo "news content crawling..."
python ./crawling/daum_news_crawling_multiprocessing.py --date 20211214 --category society --page_count 50 --max_page 1000
python ./crawling/daum_news_crawling_multiprocessing.py --date 20211214 --category politics --page_count 50 --max_page 1000
python ./crawling/daum_news_crawling_multiprocessing.py --date 20211214 --category economic --page_count 50 --max_page 1000
python ./crawling/daum_news_crawling_multiprocessing.py --date 20211214 --category foreign --page_count 50 --max_page 1000
python ./crawling/daum_news_crawling_multiprocessing.py --date 20211214 --category culture --page_count 50 --max_page 1000
python ./crawling/daum_news_crawling_multiprocessing.py --date 20211214 --category entertain --page_count 50 --max_page 1000
python ./crawling/daum_news_crawling_multiprocessing.py --date 20211214 --category sports --page_count 50 --max_page 1000
python ./crawling/daum_news_crawling_multiprocessing.py --date 20211214 --category digital --page_count 50 --max_page 1000

# 2. clustering
echo ""
echo "clustering..."
python ./clustering/retriever.py --date 20211214 --category society
python ./clustering/retriever.py --date 20211214 --category politics
python ./clustering/retriever.py --date 20211214 --category economic
python ./clustering/retriever.py --date 20211214 --category foreign
python ./clustering/retriever.py --date 20211214 --category culture
python ./clustering/retriever.py --date 20211214 --category entertain
python ./clustering/retriever.py --date 20211214 --category sports
python ./clustering/retriever.py --date 20211214 --category digital

# 3. summarization