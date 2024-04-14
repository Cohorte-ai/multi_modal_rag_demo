import streamlit as st
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Please provide an answer to the question, using only the context provided below. Ensure your response is fully informed by and directly relevant to this specific context, without exceeding its scope or introducing external information:

{context}

---

Based on the context given, please respond to the following question. Your answer should:
- Be comprehensive and conclusive, without being prematurely truncated.
- Utilize bulleted lists for clarity and organization, whenever possible, instead of consolidating all information into a single paragraph.

Question: {question}
"""

APP_TITLE = "Multimodal RAG for PDFs"
SVG_ICON_PATH = 'resources/static/cohorte_logo_content.svg'

import base64

# Read the SVG file and encode it
with open(SVG_ICON_PATH, 'rb') as file:
    data = file.read()
    b64 = base64.b64encode(data).decode("utf-8")
    image_html = f'<img src="data:image/svg+xml;base64,{b64}" style="width:50px; height:auto;"/>'

# Combine the image HTML and the Title in Markdown
combined_html_title = f"""
<div style='display: flex; align-items: center;'>
    {image_html}
    <h1 style='margin-left: 20px;'>{APP_TITLE}</h1>
</div>
"""


def main():
    # Display the combined SVG and Title
    st.markdown(combined_html_title, unsafe_allow_html=True)

    # User input for query text
    query_text = st.text_input("Enter your query:")

    if st.button("Submit"):
        # Prepare the DB
        embedding_function = OpenAIEmbeddings()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Search the DB
        results = db.similarity_search_with_relevance_scores(query_text, k=3)
        if len(results) == 0 or results[0][1] < 0.7:
            st.error(f"Unable to find matching results.{results}")
            return

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        # Display context and query
        # st.subheader("Context:")
        # st.text(context_text)
        # st.subheader("Query:")
        # st.text(query_text)

        # Initialize ChatOpenAI model
        model = ChatOpenAI()

        # Get response
        response_text = model.predict(prompt)

        # Display response and sources
        sources = [doc.metadata.get("source", None) for doc, _score in results]
        st.subheader("Response:")
        st.write(response_text)
        st.subheader("Sources:")
        st.write("\n\n".join(filter(None, sources)))  # Remove None values and join sources with comma


if __name__ == "__main__":
    main()
