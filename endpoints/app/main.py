import logging
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from app.helpers.exception_handler import CustomException, http_exception_handler

# Log
logging.config.fileConfig(settings.LOGGING_CONFIG_FILE,
                          disable_existing_loggers=False)


app = FastAPI(title="Chatbot Services",
              description="LiemT")

from app.api.router import router
app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_exception_handler(CustomException, http_exception_handler)


# static
public_path = "static/public"
if not os.path.exists(public_path):
    os.makedirs(public_path, exist_ok=True)
app.mount(
    "/static", StaticFiles(directory=public_path), name="static")


# /docs
@app.get(f"/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    print('here')
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/docs/swagger-ui-bundle.js",
        swagger_css_url="/static/docs/swagger-ui.css",
        swagger_ui_parameters={
            "persistAuthorization": True
        }
    )


@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()


@app.get(f"/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc",
        redoc_js_url="/static/docs/redoc.standalone.js",
    )


if __name__ == "__main__":
    uvicorn.run(app, host=settings.BASE_HOST, port=settings.BASE_PORT, limit_concurrency=10)
