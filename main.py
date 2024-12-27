from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

# Initialisation de l'application FastAPI
app = FastAPI()

# Configuration du dossier contenant les templates
templates = Jinja2Templates(directory="templates")

# =========================================
# Routes
# =========================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Page d'accueil avec un bouton pour accéder à l'analyse.
    """
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/analyze", response_class=HTMLResponse)
async def analyze_page(request: Request):
    """
    Page d'analyse avec un popup indiquant que la fonctionnalité n'est pas encore disponible.
    """
    return templates.TemplateResponse("analyze.html", {"request": request})
