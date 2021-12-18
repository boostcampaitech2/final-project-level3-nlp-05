#!/bin/bash

date=$1
categories="society politics economic foreign culture entertain sports digital"

# 1. crawling

# 1.1 news title and url crawling
echo "news title crawling..."
for category in $categories
do
    python ./crawling/daum_news_title_crawling.py --date $date --categories $category
done

# 1.2 news content crawling
echo ""
echo "news content crawling..."
for category in $categories
do
    python ./crawling/daum_news_crawling_multiprocessing.py --date $date --category $category --page_count 50 --max_page 1000
done

# 2. clustering
echo ""
echo "clustering..."
for category in $categories
do
    python ./clustering/retriever.py --date $date --category $category
done

# 3. summarization
echo ""
echo "summary..."
for category in $categories
do
    python ./summary/inference.py --date $date --category $category
done