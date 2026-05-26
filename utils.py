from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.core.prompts import ChatPromptTemplate

def load_and_split_document(file_path):
    """Загружает текстовый файл и разбивает его на фрагменты."""
    try:
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        return text_splitter.split_documents(documents)
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    except Exception as e:
        raise Exception(f"Ошибка при чтении файла: {e}")

def create_knowledge_base(file_path):
    """Создает базу знаний (векторное хранилище) из файла."""
    documents = load_and_split_document(file_path)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

def create_chat_chain(vectorstore, openai_api_key):
    """Создает цепочку (chain) для ответов на вопросы."""
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Ответь на вопрос, опираясь на следующий контекст: {context}"),
        ("human", "{input}"),
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever()
    chain = create_retrieval_chain(retriever, document_chain)
    return chain
