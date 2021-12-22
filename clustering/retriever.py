import json
import argparse
import string
import timeit

import os
from glob import glob
from numpy.core.fromnumeric import shape

import pandas as pd
import numpy as np
from collections import defaultdict
from scipy import spatial
from sklearn import cluster

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn import metrics

from konlpy.tag import Hannanum
from datetime import date, timedelta

import warnings

warnings.filterwarnings("ignore")

dict_categories = {
        "society": "사회",
        "politics":"정치",
        "economic":"경제",
        "foreign":"국제", 
        "culture": "문화",
        "entertain":"연예", 
        "sports":"스포츠",
        "digital":"IT"
    }   

parser = argparse.ArgumentParser() 

def get_args():
    """ retrieve arguments for clustering """
    parser.add_argument(
        '--date', 
        default=(date.today() - timedelta(1)).strftime("%Y%m%d"),
        type=str, 
        help="date of news"
    )
    parser.add_argument(
        "--category",
        default="politics",
        type=str,
        help="category of news",
        choices=["society", "politics", "economic", "foreign", "culture", "entertain", "sports", "digital"]
    )
    parser.add_argument(
        "--tfidf_max_features",
        default=None,
        type=int,
        help="Max features for building TFIDF-Vectorizer"
    )
    parser.add_argument(
        "--topk_keywords",
        default=5,
        type=int,
        help="Number of keywords to display per cluster"
    )
    parser.add_argument(
        "--topk_cluster",
        default=3,
        type=int,
        help="Number of news articles to display per cluster"
    )
    parser.add_argument(
        "--grid_numbers",
        default=20,
        type=int,
        help="Size of grid search increments"
    )
    args = parser.parse_args()
    return args

def filter_sentence_articles(df):
    """ filter articles by string & sentence length """
    drop_index_list = [] 
    for i in range(len(df['article'])):
        if len(df['article'][i]) < 300 or df['article'][i].count('다.') < 3: # 문장이 7이상으로 수정
            drop_index_list.append(i)         
    df = df.drop(drop_index_list)
    df.index = range(len(df)) 
    return df

def preprocess(sent):
    """ preprocessing before vectorizing """
    #《》 【 】［］「」 ｢｣  ≪≫〈〉 -> () 
    punct_mapping = {'〜':'~',"‘": "'", "´": "'", "°": "", "™": "tm", "√": " 제곱근 ", "×": "x", "m²": "제곱미터",
                    "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', '∞': 'infinity',
                    'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '',
                     '³': '3', 'π': 'pi','·':',','%':'퍼센트'}
    parantheses = '《【{［[「｢≪〈〉》】］]}」｣≫'
    for p in punct_mapping:
        sent = sent.replace(p, punct_mapping[p])
    for p in parantheses:
        if p in parantheses[:len(parantheses)//2+1]: sent = sent.replace(p, '(')
        else: sent = sent.replace(p, ')')

    exclude = '☆☏$^▲;\"?\'!@☎☞�:◎▷‥◀/◇*▶▼🎧_+`%↑#\\◈※△-·◆…■|&★'
    for p in exclude:
        sent = sent.replace(p, '')
    return sent


def json_to_df(json_path, idx, date, category):
    """ convert json file to desired pandasDateFrame format """
    df = pd.read_json(json_path)
    df.drop(['id','extractive', 'abstractive'], axis=1, inplace=True)
    data = {
        "id": [],
        "category": [],
        "source": [],
        "publish_date": [],
        "origin_title": [],
        "title": [],
        "origin_text": [],
        "text": [],
        "article": [],
        "concat": []
    }
    for i, row in df.iterrows():
        # id
        category_values = list(dict_categories.values())
        data["id"].append(f"{category_values.index(category) + 1}-{idx:04d}-{date}")
        idx += 1

        # category, source, publish_date
        data["category"].append(row["category"])
        data["source"].append(row["source"])
        data["publish_date"].append(row["publish_date"])

        # origin_title, title
        title = row["title"]
        data["origin_title"].append(title)
        new_title = preprocess(title)
        data["title"].append(new_title)
        
        # origin_text, text, article, concat
        text = row["text"]
        data["origin_text"].append(text)
        new_texts = []
        article = []
        for i in range(len(text)):
            paragraph = []
            for j in range(len(text[i])):
                sentence = text[i][j]["sentence"]
                new_sentence = preprocess(sentence)
                new_obj = {"index": text[i][j]["index"], "sentence": new_sentence}
                paragraph.append(new_obj)
                article.append(new_sentence)
            new_texts.append(paragraph)
        data["text"].append(new_texts)
        new_article = " ".join(article)
        data["article"].append(new_article)
        data["concat"].append(" ".join([new_title, new_article]))
    
    res_df = pd.DataFrame(data)
    return res_df, idx


def retrieve_optimal_eps(df, vector, grid_numbers = 10, grid_lower = 0.2, grid_upper = 0.7, penalty_rate = 0.1):
    """ retrieve DBSCAN model with optimal epsilon value by grid search """
    eps_grid = np.linspace(grid_lower, grid_upper, num = grid_numbers)
    eps_best = eps_grid[0]
    overall_score = -1
    model_best = None
    silhouette_score_optimal = 0
    percentage_discarded_optimal = 0
    overall_score_max = 0

    for eps in eps_grid:
        model = DBSCAN(eps=eps, min_samples=5, metric = "cosine") # Cosine Distance 
        result = model.fit_predict(vector)
        df['cluster'] = result
        unlabeled_counts = len(df[df['cluster'] == -1]) + len(df[df['cluster'] == 0])

        # Extract performance metric
        silhouette_score = metrics.silhouette_score(vector, result)
        percentage_discarded = unlabeled_counts/len(df)

        # scoring guideline -> (percentage undiscarded + silhouette score)
        overall_score =  silhouette_score - penalty_rate*percentage_discarded 
        # 50% 데이터 버리면 0.5*0.05 = 0.025 정도의 페널티 -> 20% 

        if overall_score > overall_score_max:
            silhouette_score_optimal = silhouette_score
            percentage_discarded_optimal = percentage_discarded
            overall_score_max = overall_score
            eps_best = eps
            model_best = model
        # print(f'eps: {eps:.2f}, overall_score: {overall_score:.2f}, silhouette_score: {silhouette_score:.2f}, percentage_discarded: , {percentage_discarded:.2f}')
    print(f'"Epsilon:", {eps_best:.2f}, Silhouette score:", {silhouette_score_optimal:.2f}, Proportion of News Discarded {percentage_discarded_optimal:.2f}')
    return model_best


def print_clustered_data(df, result, print_titles = True):
    """ print cluster details of news discarded """
    for cluster_num in set(result):
        # -1,0은 노이즈 판별이 났거나 클러스터링이 안된 경우
        if(cluster_num == -1 or cluster_num == 0): 
            continue
        else:
            print("cluster num : {}".format(cluster_num))
            temp_df = df[df['cluster'] == cluster_num] # cluster num 별로 조회
            
            if print_titles:
                for title in temp_df['title']:
                    print(title) # 제목으로 살펴보자
                print()
    unlabeled_counts = len(df[df['cluster'] == -1]) + len(df[df['cluster'] == 0])
        
    print(f'분류 불가능한 기사 개수 : {unlabeled_counts}')
    print(f'분류 불가 비율 : {100*unlabeled_counts/len(df):.3f}%')


def retrieve_featured_article(df, centers, dict):
    """ retrieve details of the featured article among a cluster """
    feature_vector_idx = []
    feature_title = []
    feature_article = []
    feature_id = []

    for i in range(1, len(centers)-1):
        temp = dict[i][0].to_dict()
        dist = []

        for idx, vector in temp.items():
            dist.append([idx, spatial.distance.cosine(centers[i+1],vector)])
            
        for order,(idx,_) in enumerate(sorted(dist, key = lambda t: -t[1])):
            if df['article'][idx].count('다.') >= 7:
                break
            elif order == len(dist) - 1: 
                idx = 0   
        feature_vector_idx.append(idx)
        feature_title.append(df['title'][idx])
        feature_article.append(df['article'][idx])
        feature_id.append(df['id'][idx])

    return feature_vector_idx, feature_title, feature_article, feature_id


def retrieve_topk_clusters(df, topk):
    """ retrieve top k clusters within a category """
    cluster_counts = df['cluster'].groupby(df['cluster']).count()
    sorted_clusters = sorted(zip(cluster_counts[2:].index,cluster_counts[2:]), reverse = True, key = lambda t: t[1])
    return [k for k,_ in sorted_clusters][:topk]


def get_cluster_details_dbscan(centers, feature_names, feature_title, feature_article, feature_id, top_n_features):
    """ return cluster details by dictionary format """
    cluster_details = {}
    
    #개별 군집별로 iteration하면서 핵심단어, 그 단어의 중심 위치 상대값, 대상 제목 입력
    for cluster_num in range(1, len(centers)-1): # -1, 0 제외
        # 개별 군집별 정보를 담을 데이터 초기화. 
        cluster_details[cluster_num] = {}
        cluster_details[cluster_num]['cluster'] = cluster_num
        
        # cluster_centers_.argsort()[:,::-1] 로 구한 index 를 이용하여 top n 피처 단어를 구함.
        top_k_idx = centers[cluster_num+1].argsort()[::-1][:top_n_features] 
        top_features = [feature_names[ind] for ind in top_k_idx]
        cluster_details[cluster_num]['top_features'] = top_features

        # top title, article
        cluster_details[cluster_num]['title'] = feature_title[cluster_num-1]
        cluster_details[cluster_num]['article'] = feature_article[cluster_num-1]
        cluster_details[cluster_num]['id'] = feature_id[cluster_num-1]
    return cluster_details

def print_cluster_details(cluster_details):    
    """ print keywords, title, article of a specific cluster """
    for cluster_num in cluster_details.keys():
        print(f'####### Cluster - {cluster_num}')
        print('Top features: ',cluster_details[cluster_num]['top_features'])
        print('Title :',cluster_details[cluster_num]['title'])
        print('Article :',cluster_details[cluster_num]['article'])
        print('='*50)

def generate_json(df, day, category, summary_category, cluster_details, retrive_topk_clusters):
    """ generate json files into desired summarization & service input format """
    result_serving = []
    result_summary = []

    for cluster_num in retrive_topk_clusters:
        top_features = cluster_details[cluster_num]['top_features']
        id = cluster_details[cluster_num]['id']    
        article = df.loc[df['id'] == id, :]

        dict_summary = {}
        dict_summary['id'] = id
        dict_summary['category'] = category
        dict_summary['title'] = article["title"].values[0]
        dict_summary['text'] = article['text'].values[0]
        result_summary.append(dict_summary)

        dict_serving = {}
        dict_serving['id'] = id
        dict_serving['category'] = category
        dict_serving['source'] = article['source'].values[0]
        dict_serving['top_features'] = top_features
        dict_serving['origin_title'] = article["origin_title"].values[0]
        dict_serving['origin_text'] = article['origin_text'].values[0]
        result_serving.append(dict_serving)
        
    with open(f'./data/{day}/cluster_for_serving_{day}_{category}.json', 'w') as f:
        json.dump(result_serving, f, ensure_ascii=False)
    with open(f'./data/{day}/cluster_for_summary_{day}_{summary_category}.json', 'w') as f:
        json.dump(result_summary, f, ensure_ascii=False)


##################################################################################
def main():
    start = timeit.default_timer()
    args = get_args()

    # 페이지별로 분리된 json 파일 통합
    file_path = f'./data/{args.date}'
    save_file_name = f"cluster_for_summary_{args.date}_{dict_categories[args.category]}.json"
    if os.path.isfile(os.path.join(file_path, save_file_name)):
        print(f'{save_file_name} is already generated.')
        return
    file_name = f"daum_articles_{args.date}_{dict_categories[args.category]}"
    file_list = sorted([file for file in glob(os.path.join(file_path, "*")) if file_name in file])

    # json-df, 전처리까지, concat
    df = pd.DataFrame()
    idx = 1
    for file in file_list:
        sub_df, idx = json_to_df(file, idx, args.date, dict_categories[args.category])
        df = pd.concat([df, sub_df])
    df = df.reset_index(drop=True)

    print(f'{len(df)} articles exist for Category : {dict_categories[args.category]}', '\n')

    # 3문장 300자 필터
    df = filter_sentence_articles(df)

    han = Hannanum() 
    df['concat_nouns'] = ''

    # Preprocessing nouns from concated (title + article)
    print(f"Preprocessing nouns...")
    for i in range(len(df['concat'])):
        tmp = ' '.join(han.nouns(df['concat'][i]))
        df['concat_nouns'][i] = tmp
    nouns = ["".join(noun) for noun in df['concat_nouns']]

    # TFIDF Vectorizing
    tfidf_vectorizer = TfidfVectorizer(min_df = 5, ngram_range=(1,2), max_features= args.tfidf_max_features)
    vector = tfidf_vectorizer.fit_transform(nouns).toarray()
    print(f'Shape of TFIDF Matrix: {vector.shape}', '\n')
    
    # DBSCAN
    vector = normalize(np.array(vector))

    # retrieve DBSCAN w/ optimal eps
    if len(df) < 200:
        model = DBSCAN(eps=0.5, min_samples=3, metric = "cosine") # Cosine Distance
    else:
        model = retrieve_optimal_eps(df, vector, grid_numbers = args.grid_numbers, grid_lower = 0.3, grid_upper = 0.7)
    result = model.fit_predict(vector)
    df['cluster'] = result

    # print clustered data
    print_clustered_data(df,result)

    # dict building for center caculation
    df['vector'] = vector.tolist()
    dict = defaultdict(list)
    for i in range(-1, df['cluster'].nunique()-1):
        dict[i].append(df[df['cluster'] == i]['vector'])

    # center for each cluster
    centers = [np.mean(np.array((list(dict[i][0]))), axis = 0) for i in dict.keys()]

    # fetches 
    _, feature_title, feature_article, feature_id  = retrieve_featured_article(df, centers, dict)

    # fetches corresponding vocabs from TFIDF Vectorizer
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    cluster_details = get_cluster_details_dbscan(centers, feature_names, feature_title, feature_article, feature_id, top_n_features=args.topk_keywords)

    print('Cluster Details...')
    print(cluster_details)

    execution_time = timeit.default_timer() - start
    print(f"Program Executed in {execution_time:.2f}s", '\n') # returns seconds
    print_cluster_details(cluster_details)

    topk_list= retrieve_topk_clusters(df, args.topk_cluster)
    generate_json(df, args.date, dict_categories[args.category], args.category, cluster_details, topk_list)


if __name__ == "__main__":
    main()