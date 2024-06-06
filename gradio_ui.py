# import concurrent.futures
# from typing import List, Union
import json
import re

import google.generativeai as genai
import gradio as gr
import requests
from IPython.display import Audio
from pydub import AudioSegment

DESCRIPTION = """
<div>
<h1 style="text-align: center;">AibeeCara English Tutor</h1>
<p>Welcome to AibeeCara! This space demonstrates our advanced AI-powered English tutor, Aibee. Aibee is designed to help users improve their English through interactive and personalized conversations.</p>
<p>Our model leverages the latest in Large Language Model (LLM) technology to provide instruction-tuned interactions. This version has an extended vocabulary to ensure a comprehensive learning experience. Feel free to explore, interact, and even duplicate this space to use it privately!</p>
<p>🔎 For more details about how to use this model with <code>transformers</code> and other technical specifics, visit the model card linked above.</p>
<p>🦕 Key Features:
<ul>
    <li>Extended vocabulary support up to 32768 tokens.</li>
    <li>Supports the latest v3 tokenizer for improved text processing.</li>
    <li>Enables function calling for advanced interactions.</li>
</ul>
</p>
</div>
"""


PLACEHOLDER = """
<div style="padding: 30px; text-align: center; display: flex; flex-direction: column; align-items: center;">
   <img src="https://i.pinimg.com/736x/6a/6b/29/6a6b29806662e0d61ce2c2fb7c1e0aca.jpg" style="width: 70%; max-width: 550px; height: auto; opacity: 0.55;  "> 
   <p style="font-size: 20px; margin-bottom: 2px; opacity: 0.65;">Ask me anything...</p>
</div>
"""


css = """
h1 {
  text-align: center;
  display: block;
}
#duplicate-button {
  margin: auto;
  color: white;
  background: #1565c0;
  border-radius: 100vh;
}
"""

url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"
api_key = "<API KEY>"
output_path = "aibee_output.mp3"

headers = {"Authorization": f"Token {api_key}", "Content-Type": "application/json"}

GOOGLE_API_KEY = "<API KEYS>"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat(history=[])


def play_audio(file_path):
    return Audio(file_path, autoplay=True)


def text_to_speech(text):
    payload = {"text": text[-1][-1]}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        # data = bytes.decode(response.content)
        # audio_data = AudioSegment.from_mp3(data)
        return output_path
        # play_audio(output_path)
    else:
        print(f"Error: {response.status_code} - {response.text}")


# def update_audio(text):
#     audio_component.value = output_path


def gemini_flash(
    message: str,
    history: list,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
) -> str:
    print(f"Message: {message}")
    response = chat.send_message(message)
    # text_to_speech(response.text)
    return response.text


chatbot = gr.Chatbot(height=450, placeholder=PLACEHOLDER, label="Aibeecara Assistant")

with gr.Blocks(fill_height=True, css=css) as demo:
    gr.Markdown(DESCRIPTION)
    input_audio_component = gr.Audio(sources="microphone", label="Input Audio")
    interface = gr.ChatInterface(
        fn=gemini_flash,
        chatbot=chatbot,
        fill_height=True,
        additional_inputs_accordion=gr.Accordion(
            label="⚙️ Parameters", open=False, render=False
        ),
        additional_inputs=[
            gr.Slider(
                minimum=0.1,
                maximum=2,
                step=0.1,
                value=1.0,
                label="Temperature",
                render=False,
            ),
            gr.Slider(
                minimum=128,
                maximum=4096,
                step=1,
                value=128,
                label="Max new tokens",
                render=False,
            ),
            gr.Slider(
                minimum=0.5,
                maximum=0.9,
                step=0.05,
                value=0.9,
                label="Top P",
                render=False,
            ),
            gr.Slider(
                minimum=5, maximum=20, step=1, value=10, label="Top K", render=False
            ),
        ],
        examples=[
            ["Give me some tips on how to improve my English vocabulary."],
            ["Explain the concept of past tense in English with examples."],
            ["What are some common English phrases used in daily conversation?"],
            ["Write a friendly email inviting a friend to a study group."],
            ["Describe your favorite hobby in English and why you enjoy it."],
        ],
        cache_examples=False,
    )
    # audio_component = gr.Audio(value=output_path, label="Audio Output", autoplay=False)
    chatbot.change(
        text_to_speech,
        inputs=[chatbot],
        outputs=[gr.Audio(label="Audio Output", type="filepath", autoplay=True)],
    )
