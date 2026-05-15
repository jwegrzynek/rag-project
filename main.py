import torch
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from openai import OpenAI
from time import time

load_dotenv()

book_path = "books/pan-tadeusz.txt"
# embedding_model_name = "intfloat/multilingual-e5-small"
embedding_model_name = "BAAI/bge-m3"

# === 1. LOADING EMBEDDING MODEL =======================================================================================

embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    },
    encode_kwargs={
        "normalize_embeddings": True
    }
)


# === 2. LOADING DOCUMENTS AND CREATING VECTOR DATABASE ================================================================

def load_documents(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # More sophisticated splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,  # Optimal for most embeddings
        chunk_overlap=70,
        separators=["\n\n", "\n", ".", " ", ""]  # Try to keep paragraphs together
    )
    splitted_text = text_splitter.create_documents([text])

    for i, doc in enumerate(splitted_text):
        doc.metadata = {
            "source": path,
            "chunk_id": i,
            "length": len(doc.page_content)
        }

    return splitted_text


persist_directory = "chroma_db/" + book_path.split('/')[1].split('.')[0] + "/" + embedding_model_name.split('/')[1]

if not os.path.exists(persist_directory):

    start = time()
    documents = load_documents(book_path)
    stop = time()
    print(f"Załadowanie książki, podział na chunki i uzyskanie metadanych: {stop - start}")

    start = time()
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}  # Better similarity metric
    )
    stop = time()
    print(f"Utworzenie bazy danych: {stop - start}")
else:
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )


# === 3. CONNECTING TO DEEPSEEK AND FUNCTION TO GENERATE ANSWER ========================================================

api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)


def generate_answer(prompt: str):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "Jesteś pomocnym asystentem, który odpowiada po polsku."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,  # default in deepseek
        max_tokens=200
    )
    return response.choices[0].message.content


# === 4. WRAPPER OF MODEL ==============================================================================================

def custom_llm(inputs, *, stop=None, callbacks=None):
    if hasattr(inputs, "to_string"):  # jeśli to StringPromptValue
        query = inputs.to_string()
    else:  # jeśli jednak dict
        query = inputs["query"]
    return generate_answer(query)


llm_runnable = RunnableLambda(custom_llm)

# === 5. CREATING PROMPT AND LANGCHAIN PIPELINE ========================================================================

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Odpowiedz na pytanie UŻYWAJĄC WYŁĄCZNIE poniższych fragmentów.
Jeśli fragmenty nie zawierają odpowiedzi, powiedz "Nie wiem" i nie zmyślaj.

Zasady:
1. Odpowiedź musi być oparta TYLKO na dostarczonych fragmentach.
2. Bądź precyzyjny – podawaj liczby, nazwy, daty jeśli są w tekście.
3. Jeśli potrzebujesz połączyć informacje z wielu fragmentów, zrób to logicznie.

Fragmenty: {context}

Pytanie: {question}

Odpowiedź (zwięzła i oparta na faktach):"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm_runnable,
    retriever=vectordb.as_retriever(search_kwargs={"k": 10}),
    return_source_documents=True,
    input_key="query",
    chain_type_kwargs={"prompt": custom_prompt}
)

# === 6. SCRIPT LOOP ===================================================================================================

conversation_file = str(
    "conversations/" + book_path.split('/')[1].split('.')[0] + "-" + embedding_model_name.split('/')[1]) + '.txt'

with open(conversation_file, "a") as f:
    while True:
        question = input("\n❔ Zadaj pytanie o lekturę (lub wpisz 'exit'): ")

        if question.lower() == "exit":
            f.close()
            break

        f.write("Pytanie:\n" + question)

        # start = time()
        result = qa_chain.invoke({"query": question})
        # stop = time()
        # print(f"Czas potrzebny na uzyskanie odpowiedzi: {stop - start}")

        f.write(
            "\n\nOdpowiedź:\n" +
            str(result['result']) +
            "\n\n==============================================================\n\n"
        )

        print("\n📘 Odpowiedź:\n", result['result'])
        print("\n==============================================================")
