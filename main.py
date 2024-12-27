from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# Initialisation de l'application FastAPI
app = FastAPI()

# =========================================
# ROUTES
# =========================================

@app.get("/", response_class=HTMLResponse)
async def main_page():
    return """
    <html>
    <head>
        <title>Accueil</title>
    </head>
    <body>
        <h1>Hello World</h1>
    </body>
    </html>
    """

# =========================================
# Lancer l'application (si nécessaire localement)
# =========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
