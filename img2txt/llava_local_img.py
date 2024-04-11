import os
from PIL import Image
from transformers import pipeline

model_id = "llava-hf/llava-1.5-7b-hf"

global pipe


def load_pipeline():
    """## Load the model using `pipeline`

        We will leverage the `image-to-text` pipeline from transformers !
        """
    global pipe
    pipe = pipeline("image-to-text", model=model_id)


load_pipeline()


def _get_image_to_text(image_path):
    image = Image.open(image_path)

    """
    It is important to prompt the model wth a specific format, which is:
    ```bash
    USER: <image>\n<prompt>\nASSISTANT:
    ```
    """

    prompt = "USER: <image>\nWhat’s in this image?\nASSISTANT:"

    outputs = pipe(
        image,
        prompt=prompt,
        generate_kwargs={"max_new_tokens": 300}
    )

    print(outputs[0]["generated_text"])
    return outputs[0]["generated_text"]


def get_images_to_texts(image_path_list: list):
    assert all([os.path.exists(_) for _ in image_path_list])
    return [_get_image_to_text(_) for _ in image_path_list]
