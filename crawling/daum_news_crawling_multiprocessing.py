import argparse
import multiprocessing
import subprocess
import time
from datetime import date, timedelta

def worker(cmd):
    subprocess.run(cmd, shell=True)

def get_args():
    parser = argparse.ArgumentParser(description="multiprocessing daum news crawling")
    parser.add_argument(
        "--date",
        default=(date.today() - timedelta(1)).strftime("%Y%m%d"),
        type=str,
        help="date of news"
    )
    parser.add_argument(
        "--category",
        default="society",
        type=str,
        help="category of news",
        choices=["society", "politics", "economic", "foreign", "culture", "entertain", "sports", "digital"]
    )
    parser.add_argument(
        "--page_count",
        default=50,
        type=int,
        help="page count",
    )
    parser.add_argument(
        "--max_page",
        default=1000,
        type=int,
        help="page count",
    )    
    args = parser.parse_args()
    return args    

if __name__ == "__main__":
    args = get_args()

    start = time.perf_counter()
    
    processes = []
    for i in range(1, args.max_page+1, args.page_count):
        cmd = f"python ./crawling/daum_news_crawling.py --date {args.date} --category {args.category} --start_page {i} --page_count {args.page_count}"
        p = multiprocessing.Process(target=worker, args=(cmd,))
        p.start()
        processes.append(p)
        
    for process in processes:
        process.join()
        
    finish = time.perf_counter()
    
    print(f"total {(finish - start)/60:.2f} minutes")