import gradio as gr
from typing import Dict, List, Union
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value


def predict_custom_trained_model_sample(
    project: str,
    endpoint_id: str,
    instances: Union[Dict, List[Dict]],
    location: str = "us-east4",
    api_endpoint: str = "mistral-aibeecara",
):
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    instances = instances if isinstance(instances, list) else [instances]
    instances = [
        json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    ]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    predictions = response.predictions
    for prediction in predictions:
        print(" prediction:", dict(prediction))


def greet(text: str) -> str:

    return text


# predict_custom_trained_model_sample(
#     project="793363955875",
#     endpoint_id="6923994156012404736",
#     location="us-east4",
#     instances={
#   "instances" : [
#     {
#       "inputs": "How would the Future of AI in 10 Years look?",
#       "parameters": {
#         "max_new_tokens": 128,
#         "temperature": 1.0,
#         "top_p": 0.9,
#         "top_k": 10
#       }
#     }
#   ]
# }
# )

demo = gr.Interface(
    fn=greet,
    inputs=gr.components.Textbox(label='Input'),
    outputs=gr.components.Textbox(label='Output'),
    allow_flagging='never'
)
