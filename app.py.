from flask import Flask, request, jsonify, render_template
from utils import load_and_split_document
from models import LocalLLM, create_vector_store
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Инициализация модели и цепочек один раз при запуске
local_llm = LocalLLM()
vector_store = None
qa_chain = None

# Улучшенный промпт для более точных ответов
prompt_template = """Ты — полезный ИИ-ассистент. Отвечай на основе предоставленного контекста.
Если ответа в контексте нет, скажи: «Я не знаю на основе загруженных документов».

Контекст:
{context}

Вопрос: {question}
Ответ:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global vector_store, qa_chain
    if 'file' not in request.files:
        logger.error("Файл не найден в запросе")
        return jsonify({"error": "Файл не найден"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Не выбран файл"}), 400

    try:
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)
        logger.info(f"Файл сохранён: {file_path}")

        # Загружаем и обрабатываем документ
        chunks = load_and_split_document(file_path)
        vector_store = create_vector_store(chunks)
        qa_chain = RetrievalQA.from_chain_type(
            llm=local_llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 5}),  # Берём 5 наиболее релевантных чанков
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        return jsonify({"message": "Файл успешно загружен и обработан!"})
    except Exception as e:
        logger.error(f"Ошибка при загрузке файла: {e}")
        return jsonify({"error": f"Ошибка при обработке файла: {str(e)}"}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    if qa_chain is None:
        return jsonify({"error": "Сначала загрузите документ"}), 400

    question = request.json.get('question', '').strip()
    if not question:
        return jsonify({"error": "Вопрос не может быть пустым"}), 400

    try:
        result = qa_chain({"query": question})
        answer = result["result"]
        source_docs = result.get("source_documents", [])
        sources = [doc.metadata.get("page", "N/A") for doc in source_docs]
        return jsonify({
            "answer": answer,
            "sources": sources
        })
    except Exception as e:
        logger.error(f"Ошибка при генерации ответа: {e}")
        return jsonify({"error": "Произошла ошибка при генерации ответа"}), 500

if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=False, host='127.0.0.1', port=5000)  # Отключён debug для стабильности
