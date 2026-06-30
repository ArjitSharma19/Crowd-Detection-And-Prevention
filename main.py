"""
Main entry point for the Crowd Detection & Prevention system's web application.
This script starts the FastAPI local development server using Uvicorn.
It also ensures the model storage directories are properly configured.
"""

# pyrefly: ignore [missing-import]
import uvicorn
import os

if __name__ == "__main__":
    # Display startup banner on the console
    print("--------------------------------------------------")
    print("          CROWDSHIELD AI INITIALIZATION           ")
    print("--------------------------------------------------")
    print("Starting local dashboard development server...")
    print("Web dashboard will be available at: http://127.0.0.1:8000")
    print("--------------------------------------------------")
    
    # Create models folder if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Run FastAPI app via uvicorn
    uvicorn.run("webapp.main:app", host="127.0.0.1", port=8000, reload=True)
