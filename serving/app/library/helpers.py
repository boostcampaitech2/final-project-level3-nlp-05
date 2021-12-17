import json
import os
import re

def get_date_list(data_root):
    """
    이용 가능한 날짜 리스트 반환 함수
    """
    date_list = os.listdir(data_root)
    date_list = sorted([date for date in date_list if re.match(r"\d{8}", date)], reverse=True)
    return date_list

def get_json_data(path: str):
    """
    json 파일을 불러오는 함수
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)    
    return data

def get_merge_data(clustering_data, summary_data):
    """
    클러스터링, 요약, tts 데이터를 병합해주는 함수
    """
    id_list = [data["id"] for data in clustering_data]
    merge_data = []
    for id in id_list:
        clustering = [data for data in clustering_data if data["id"] == id][0]
        summary = [data for data in summary_data if data["id"] == id][0]
        merge_data.append({
            "id": id,
            "category": clustering["category"],
            "title": clustering["title"],
            "article": clustering["text"],
            "summary": summary["summary"],
            "audio_file": "souljaboy_gohard.mp3"
        })
    return merge_data