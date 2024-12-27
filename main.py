from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>S.O.P.H.I.A - Accueil</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f4f9;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                color: #333;
            }
            .container {
                text-align: center;
                background: #ffffff;
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
                max-width: 600px;
                margin: auto;
            }
            h1 {
                font-size: 2.5rem;
                color: #2c3e50;
            }
            h2 {
                font-size: 1.5rem;
                color: #2980b9;
            }
            p {
                font-size: 1rem;
                line-height: 1.5;
                margin-top: 10px;
            }
            footer {
                margin-top: 20px;
                font-size: 0.8rem;
                color: #7f8c8d;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Bienvenue sur S.O.P.H.I.A</h1>
            <h2>Segmentation Optimisée Pour l'Harmonie Intelligente des Automobiles</h2>
            <p>
                Bienvenue dans notre application de segmentation d'images, conçue pour améliorer 
                la sécurité et la navigation des véhicules autonomes. Grâce à des technologies avancées 
                de vision par ordinateur, nous analysons vos images pour fournir des prédictions détaillées, 
                tout en garantissant une confidentialité totale : <strong>aucune image n'est stockée</strong> sur notre application.
            </p>
            <footer>Merci d'utiliser notre service. Ensemble, façonnons l'avenir de la conduite autonome.</footer>
        </div>
    </body>
    </html>
    """
