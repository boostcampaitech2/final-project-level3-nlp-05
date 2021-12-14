# How to start?
requirements 설치
```
sh ./crawling_requirements.sh
```

<br>

# 크롤링
- [x] 위키트리 뉴스기사 크롤링
  - 요약문과 뉴스 본문 크롤링
  - AI Hub 문서요약 텍스트 - 신문 기사 샘플 데이터와 유사한 형태로 만들기
- [x] 네이트 일자별 카테고리별 뉴스 키워드 크롤링
- [x] 네이버 일자별 랭킹 뉴스 제목 크롤링

<br>

# 다음 뉴스 크롤링 방법

## 1. 다음 뉴스 기사 제목 및 url 크롤링

```bash
python daum_news_title_crawling.py --date 20211208 --category society
```

- arguments
  - `--date`: 크롤링하고자 하는 날짜 지정
  - `--category`: 크롤링하고자 하는 카테고리 지정
- 실행 결과 기사 제목과 url 정보를 포함하고 있는 json 파일 생성됨

<br>

## 2. 다음 뉴스 기사 본문 크롤링 (multiprocessing)

```bash
python daum_news_crawling_multiprocessing.py --date 20211208 --category society --page_count 50 --max_page 453
```

- arguments
  - `--date`: 크롤링하고자 하는 날짜 지정
  - `--category`: 크롤링하고자 하는 카테고리 지정
  - `--page_count`: 한 프로세스에서 크롤링할 기사의 페이지 개수, `기사 개수 = 페이지 개수 * 15`
  - `--max_page`: 1.을 통해 생성된 json 파일의 가장 마지막 기사의 마지막 페이지 번호
- 실행 결과 각 페이지별 기사 본문을 포함하고 있는 json 파일들이 생성된다.
- 2021년 12월 8일 사회 기사 크롤링 시 약 18분 정도가 걸리는 것을 확인할 수 있었다.

<br>

# result structure

## wikitree_crawling.py result

```
{
    "id": "353465974", 
    "category": "종합", 
    "publish_date": "2019-07-22 00:00:00", 
    "extractive": [0],
    "abstractive": ["요약 문장1","요약 문장2"],
    "title": "충주시, 민간지원 보조사업 대형축제 운영 감사 돌입", 
    "article": [
        {"index": 0, "sentence": "보조금 집행 위법행위·지적사례 늘어"}, 
        {"index": 1, "sentence": "특별감사반, 2017~2018년 축제 점검"},
    ]
}
```

## naver_news_crawling.py result

```
{
    "20190722": [
      {
        "title": "화이자·모더나 '오미크론 신종 변이 연구 중'", 
        "categories": ["세계", "정치", "경제"], 
        "publish_date": "2019-07-22", 
      },
      { 
        "title": "전단지 넣은 70대 할머니 무릎 꿇린 미용실 점주…비판 일자 사과", 
        "categories": ["세계"], 
        "publish_date": "2019-07-22", 
      },
    ],
    "20190721: [
      {
        "title": "함께 끌고, 밀고…CCTV에 기록된 ‘우리의 아름다운 동행’", 
        "category": ["세계"], 
        "publish_date": "2019-07-21", 
      },
      { 
        "title": "새 변이 ‘오미크론’…우려 변이로 지정", 
        "category": ["사회"], 
        "publish_date": "2019-07-21", 
      },
      
    ]
      
}
```

## nate_crawling.py
- terminal 출력

## daum_news_title_crawling.py

- 

```
{
  "date": "20211205",
  "category": "국제",
  "articles": [
      {
        'id': '015_08',
        'title': '헝다 \'디폴트\' 가능성 시사..中 "개별 사건" 파장 최소화 주력', 
        'url': 'https://v.daum.net/v/20211205151346812',
      },
      {
        'id': '015_09', 
        'title': '백건우 "삶을 얼마나 깊이 깨달았느냐에 따라 음악도 깊어져"', 
        'url': 'https://v.daum.net/v/20211205151104773',
      },
  ]
}
```

## daum_news_crawling.py

```
{
    "id": "353465974", 
    "category": "종합", 
    "publish_date": "2019-07-22 00:00:00", 
    "extractive": [0],
    "abstractive": ["요약 문장1","요약 문장2"],
    "title": "충주시, 민간지원 보조사업 대형축제 운영 감사 돌입", 
    "article": [
        {"index": 0, "sentence": "보조금 집행 위법행위·지적사례 늘어"}, 
        {"index": 1, "sentence": "특별감사반, 2017~2018년 축제 점검"},
    ]
}
```