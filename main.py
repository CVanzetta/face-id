"""
Main FastAPI application entry point
"""
import sys
import logging
from pathlib import Path
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse

# Add app directory to Python path
sys.path.append(str(Path(__file__).parent))

from app.api.endpoints import app as api_app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create templates instance
templates = Jinja2Templates(directory="web/templates")

# Add web interface route
@api_app.get("/web", response_class=HTMLResponse)
async def web_interface(request: Request):
    """Serve the web interface"""
    return templates.TemplateResponse("index.html", {"request": request})

# Mount static files
api_app.mount("/static", StaticFiles(directory="web/static"), name="static")

if __name__ == "__main__":
    import yaml
    
    # Load configuration
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    api_config = config.get('api', {})
    
    uvicorn.run(
        "main:api_app",
        host=api_config.get('host', '0.0.0.0'),
        port=api_config.get('port', 8000),
        reload=api_config.get('debug', False),
        log_level="info"
    )
