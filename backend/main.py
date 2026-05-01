import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

# Add the project root to the python path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.models import engine
from backend.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for the FastAPI application.
    Executes startup and shutdown events, such as verifying database connectivity.
    """
    print("Starting up FastAPI application...")
    
    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)
    
    # Database connection check removed. Running in memory mode.
    print("Database connection check disabled. Running in memory mode.")
        
    yield
    
    # Shutdown phase
    print("Shutting down FastAPI application...")


# Initialize FastAPI app
app = FastAPI(
    title="Satellite Change Detection API",
    description="Backend API for managing and running multi-temporal satellite change detection pipelines.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS Middleware
# In a production environment, you should restrict `allow_origins` to your specific frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount outputs directory for static file serving
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Register routes under the /api/v1 prefix
app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
