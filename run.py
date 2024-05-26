import gradio as gr
from fastapi import FastAPI

from gradio_ui import demo

app = FastAPI()


@app.get("/check")
async def root():
    return "Aibeecara app is running at /aibeecara", 200


app = gr.mount_gradio_app(app, demo, path="/aibeecara")
