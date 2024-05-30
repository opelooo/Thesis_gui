import io
import os
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
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/doc", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/models/")
def get_models():
    model_files = [f for f in os.listdir("/app/models") if f.endswith('.keras')]
    return {"models": model_files}

@app.post('/predict')
async def predict_route(url_input: URLInput):
    try:
        # Await the prediction result
        prediction = await predict(url_input.url)
        
        # Get the max accuracy prediction
        predicted_class = int(np.argmax(prediction))
        
        # Get the accuracy from the predicted class
        accuracy = float(prediction[0][predicted_class])
        print(predicted_class, prediction[0], np.argmax(prediction[0]))
        return {"status": "success", "predicted_class": predicted_class, "accuracy": accuracy}

    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
	# port = int(os.environ.get('PORT', 8080))
	# run(app, host="0.0.0.0", port=port, timeout_keep_alive=1200)
    run(app)