import base64
import os

import requests
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# OpenAI API Key
api_key = os.environ["OPENAI_API_KEY"]


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def _get_image_to_text(image_path):
    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What’s in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    print(response.json())

    return response.json()["choices"][0]["message"]["content"]


def get_images_to_texts(image_path_list: list):
    assert all([os.path.exists(_) for _ in image_path_list])
    return [_get_image_to_text(_) for _ in image_path_list]
