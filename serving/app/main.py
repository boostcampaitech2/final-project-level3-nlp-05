from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from .library.helpers import *
import json

app = FastAPI()

templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    date = "20211205"
    category = "문화"
    with open(f"./static/data/daum_articles_{date}_{category}.json", "r", encoding="utf-8") as f:
        article_data = json.load(f)
    article = article_data[2]
    data = {
        "title":article['title'],
        "summary": article['abstractive'],
        "article":article['article'],
        "audio_link":"",
        "audio_file": "sample-audio.mp3",
        "podcast_link":"https://www.podbbang.com/",
        "category":article['category'],
        "pub_date":article['publish_date'],

    }
    return templates.TemplateResponse(
        "page.html",
        {"request": request, "data": data}
    )   

@app.get("/page/{page_name}", response_class=HTMLResponse)
async def page(request: Request, page_name: str):
    
    data = {
        "title":"",
        "summary": "",
        "article":"",
        "audio_link":"",
        "pod_cast_link":"",
        "category":"",
        "pub_date":"",
    }
    return templates.TemplateResponse(
        "page.html",
        {"request": request, "data": data}
    )