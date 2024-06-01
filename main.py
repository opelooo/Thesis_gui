import io
import os
import sys
import numpy as np
from uvicorn import run
from fastapi import Request, HTTPException
import apis_config as config
from pydantic import BaseModel
from functions import predict
from starlette.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware


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


templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    print("open main menu")

    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/doc", response_class=HTMLResponse)
async def main(request: Request):
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
        # Await the prediction result
        print(f"predict_route: URL={url_input.url}, Model Name={url_input.model_name}")
        sys.stdout.flush()
        
        prediction = await predict(url_input.url, url_input.model_name)
                
        return prediction

    except Exception as e:
        print(str(e))
        sys.stdout.flush()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
	# port = int(os.environ.get('PORT', 8080))
	# run(app, host="0.0.0.0", port=port, timeout_keep_alive=1200)
    run(app)