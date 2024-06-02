import io
import os
import sys
import numpy as np
from uvicorn import run
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
import apis_config as config
from functions import predict

app = config.app

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware to add security and caching headers
class CustomHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers['Cache-Control'] = 'max-age=3600, must-revalidate'
        response.headers['Pragma'] = 'cache'
        response.headers['Expires'] = 'Wed, 01 Jan 2025 12:00:00 GMT'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

app.add_middleware(CustomHeadersMiddleware)

# HTTPS Redirect Middleware for security
app.add_middleware(HTTPSRedirectMiddleware)

class URLInput(BaseModel):
    url: str
    model_name: str

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/doc", response_class=HTMLResponse)
async def doc(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/models/")
def get_models():
    model_files = [f for f in os.listdir("/app/models") if f.endswith('.keras')]
    return {"models": model_files}

@app.post('/predict/')
async def predict_route(url_input: URLInput):
    try:
        model_path = os.path.join("models", url_input.model_name)
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found")
        
        prediction = await predict(url_input.url, url_input.model_name)
        return prediction

    except Exception as e:
        print(str(e))
        sys.stdout.flush()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    run(app, host="0.0.0.0", port=port)
