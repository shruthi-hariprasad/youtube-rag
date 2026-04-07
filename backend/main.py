from fastapi import FastAPI

app = FastAPI()


@app.post("/videos")
async def create_video():
    return {"message": "video endpoint coming soon"}


@app.get("/videos")
async def list_videos():
    return {"message": "list endpoint coming soon"}


@app.post("/query")
async def query():
    return {"message": "query endpoint coming soon"}
