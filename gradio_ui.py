import concurrent.futures
from typing import Dict, List, Union

import google.generativeai as genai
import gradio as gr
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

GOOGLE_API_KEY = "AIzaSyBTg3eEIsM0P124XO8LfbeGTjb3dd_Va98"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat(history=[])


def predict_custom_trained_model_sample(
    message: str,
    history: list,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    project: str = "793363955875",
    endpoint_id: str = "6923994156012404736",
    location: str = "us-east4",
    api_endpoint: str = "us-east4-aiplatform.googleapis.com",
):
    conversation = []
    for user, assistant in history:
        conversation.extend(
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        )
    conversation.append({"role": "user", "content": message})
    instances = [
        {
            "inputs": message,
        }
    ]
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    instances = instances if isinstance(instances, list) else [instances]
    instances = [
        json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    ]
    parameters_dict = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    predictions = response.predictions
    return predictions[0]


def gemini_flash(
    message: str,
    history: list,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
) -> str:
    """Processes a message using the Gemini Flash model, handling concurrent requests."""
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future = executor.submit(
            gemini_flash_async,
            message,
            history,
            temperature,
            top_p,
            top_k,
            max_new_tokens,
        )
        return future.result()


def gemini_flash_async(
    message: str,
    history: list,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
) -> str:
    """Sends a message to the Gemini Flash model asynchronously."""
    response = chat.send_message(message)  # Use the provided chat session
    return response.text


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


def greet(message: str, history: list, temperature: float, max_new_tokens: int) -> str:

    return message


chatbot = gr.Chatbot(height=450, placeholder=PLACEHOLDER, label="Aibeecara Assistant")

with gr.Blocks(fill_height=True, css=css) as demo:
    gr.Markdown(DESCRIPTION)
    gr.ChatInterface(
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
