from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from routes import tokenize, steamming, named_recog, pos_tagging

app = FastAPI()

app.include_router(tokenize.router)
app.include_router(steamming.router)
app.include_router(named_recog.router)
app.include_router(pos_tagging.router)

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    context = {'request': request}
    return templates.TemplateResponse(
        name="index.html",
        context=context
    )