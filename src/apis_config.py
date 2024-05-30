from fastapi import FastAPI

description = """
URL Classifier API helps you classify URLs as phishing, non-phishing, inactive phishing, or malware. 🚀

## Items

You can go to /documentation for more information and how to use it.

## Users

You will be able to:

* **Submit URL and Predict** (Implemented).
"""

tags_metadata = [
    {
        "name" : "predict",
        "description": "Predict a URL"
    }
]

app = FastAPI(
    title="URL-Classifier-API",
    description=description,
    version="1.0.0",
    contact={
        "name": "Mathys Jorge Alberino Seilatu",
        "url": "https://github.com/opelooo",
        "email": "mathys.alberino@gmail.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT/",
    },
    openapi_tags=tags_metadata
)
