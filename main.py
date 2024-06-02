import os
import sys
from uvicorn import run
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from pydantic import BaseModel
import apis_config as config
from functions import predict
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

app = config.app

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLInput(BaseModel):
    url: str
    model_name: str

app.mount("/static", StaticFiles(directory="static", html=True), name="static")

@app.get("/favicon.ico")
async def get_favicon():
    headers = {
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "max-age=3600"  # Adjust cache duration as needed
    }
    
    return FileResponse("static/favicon.ico", headers=headers)

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    headers={
        "Cache-Control": f"max-age=3600", 
        "Content-Type": "text/html; charset=utf-8", 
        "X-Content-Type-Options": "nosniff"
    }
    
    return templates.TemplateResponse("index.html", {"request": request}, headers=headers)

@app.get("/doc", response_class=HTMLResponse)
async def doc(request: Request):
    return templates.TemplateResponse("doc.html", {"request": request})

@app.get("/models/")
def get_models(response: Response):
    model_files = [f for f in os.listdir("/app/models") if f.endswith('.keras')]
    headers={
        "Cache-Control": "max-age=3600", 
        "Content-Type": "application/json,application/json; charset=utf-8", 
        "X-Content-Type-Options": "nosniff"
    }
    
    return JSONResponse(content={"models": model_files}, headers=headers)

@app.post('/predict/')
async def predict_route(url_input: URLInput):
    try:
        model_path = os.path.join("models", url_input.model_name)
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found")
        
        prediction = await predict(url_input.url, url_input.model_name)
        return JSONResponse(content={prediction}, headers={"Content-Type": "application/json; charset=utf-8", "X-Content-Type-Options": "nosniff"})

    except Exception as e:
        print(str(e))
        sys.stdout.flush()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # port = int(os.environ.get('PORT', 8080))
    # run(app, host="0.0.0.0", port=port, timeout_keep_alive=1200)
    run(app)
