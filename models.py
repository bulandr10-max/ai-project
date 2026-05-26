from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
import torch
import logging

logger = logging.getLogger(__name__)

class LocalLLM:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        try:
            # Используем более экономный режим для слабых ПК: без float16, если возникают ошибки
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.pipe = pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            logger.info("Модель успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели {model_name}: {e}")
            raise

    def invoke(self, prompt, max_length=512):
        try:
            generation_config = {
                "max_new_tokens": max_length,
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            }
            result = self.pipe(prompt, **generation_config)
            return result[0]["generated_text"]
        except Exception as e:
            logger.error(f"Ошибка генерации текста: {e}")
            return "Произошла ошибка при генерации ответа."

def create_vector_store(chunks):
    try:
        # Используем легковесную, но качественную модель для эмбеддингов
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_store = FAISS.from_documents(chunks, embeddings)
        logger.info("Векторное хранилище успешно создано")
        return vector_store
    except Exception as e:
        logger.error(f"Ошибка создания векторного хранилища: {e}")
        raise
