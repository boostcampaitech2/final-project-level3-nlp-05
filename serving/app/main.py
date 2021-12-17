from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .library.helpers import *
import json

import os
from pathlib import Path
import datetime

app = FastAPI()

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

DATA_ROOT = os.path.join(Path(os.getcwd()).parent, "data")
CATEGORIES_DICT = {
    'society': '사회',
    'politics': '정치',
    'economic': '경제',
    'foreign': '국제',
    'culture': '문화',
    'entertain': '연예',
    'sports': '스포츠',
    'digital': 'IT'
}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, sel_date: str = None):
    date_list = get_date_list(DATA_ROOT)
    #date = (datetime.date.today() - datetime.timedelta(1)).strftime("%Y%m%d")
    date = "20211215"
    if sel_date is not None:
        date = sel_date    

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "date_list": date_list,
            "date": date
        }
    )   

@app.get("/page/{category}", response_class=HTMLResponse)
async def page(request: Request, category: str, sel_date: str = None):
    date_list = get_date_list(DATA_ROOT)
    #date = (datetime.date.today() - datetime.timedelta(1)).strftime("%Y%m%d")
    date = "20211215"
    if sel_date is not None:
        date = sel_date

    clustering_data = get_json_data(os.path.join(DATA_ROOT, f"{date}/clustering_{date}_{CATEGORIES_DICT[category]}.json"))
    summary_data = get_json_data(os.path.join(DATA_ROOT, f"{date}/summary_{date}_{CATEGORIES_DICT[category]}.json"))
    json_data = get_merge_data(clustering_data, summary_data)

    return templates.TemplateResponse(
        "page.html",
        {
            "request": request,
            "data_root": DATA_ROOT,
            "date_list": date_list,
            "date": date,
            "category_eng": category,
            "category_kor": CATEGORIES_DICT[category],
            "json_data": json_data,
        }
    )