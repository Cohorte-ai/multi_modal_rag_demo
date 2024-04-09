import logging

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import os
import shutil
from dotenv import load_dotenv
from pypdf import PdfReader

from img2txt.openai_local_img import get_images_to_texts

# Load variables from .env file
load_dotenv()
logging.getLogger().setLevel(logging.INFO)

CHROMA_PATH = "chroma"
DATA_PATH = "data/raw"
glob_pdf_pattern = "*.pdf"
extracted_img_folder = os.path.join(os.path.dirname(__file__), "data/extracted_imgs/")
if not os.path.exists(extracted_img_folder):
    os.makedirs(extracted_img_folder)


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks_from_text = split_text(documents)

    pdf_paths = [i.metadata["source"] for i in documents]
    chunks_from_images = get_texts_from_images(pdf_paths)

    chunks = chunks_from_text + chunks_from_images
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob=glob_pdf_pattern)
    documents = loader.load()
    print(f"{len(documents)} documents loaded successfully!")
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


def get_texts_from_images(pdf_paths: list):
    image_path_list = []
    image_metadata_list = []
    for each_pdf_path in pdf_paths:
        pdf_obj = PdfReader(each_pdf_path)
        for page in pdf_obj.pages:
            for img_idx, image in enumerate(page.images):
                extracted_img_path = os.path.join(extracted_img_folder, image.name)
                with open(extracted_img_path, "wb") as fp:
                    fp.write(image.data)
                    image_path_list.append(extracted_img_path)
                    image_metadata_list.append(f"{each_pdf_path} | PageNum: {page.page_number} | img_idx: {img_idx}")

    logging.info(f"Number of images {len(image_path_list)} extracted")
    img_descriptions = get_images_to_texts(image_path_list)

    chunks = [Document(page_content=img_descriptions[i], metadata={"source": f"{image_metadata_list[i]}"}) for i in
              range(len(img_descriptions))]

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
