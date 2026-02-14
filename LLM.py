from llama_cpp import Llama
import multiprocessing
import os
from huggingface_hub import hf_hub_download



model_path = 'model/qwen2.5-7b-instruct-q4_k_m.gguf' #  qwen2.5-7b-instruct-q4_k_m.gguf Qwen2.5-3B-Instruct-Q4_K_M.gguf
n_threads = max(1, multiprocessing.cpu_count()-2)
#n_threads = 6

system_prompt = """Ты технический специалист, который отвечает на вопросы касаемо дуокументации.
                    Твоя главная задача  - отвечать на вопросы ТОЛЬКО с использованием контекста.
                    Если в контексте нет информации для ответа, то ты отвечаешь : "В документации отсутствует информация о вашем запросе."
                    НЕ придумывай факты для ответа. НЕ используй свои знания, которых нет в контексте. Давай ответы на русском языке
                """

class LLMEngine():
    def __init__(self):

        self._ensure_model_exists()

        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=n_threads,
            n_batch=512,
            flash_attn=True,
            n_gpu_layers=0,
            verbose=False
        )

    def _ensure_model_exists(self):

        if not os.path.exists(model_path):
            print('Модель не найдена')
            
            os.makedirs('model', exist_ok=True)

            hf_hub_download(
                repo_id="paultimothymooney/Qwen2.5-7B-Instruct-Q4_K_M-GGUF", # paultimothymooney/Qwen2.5-7B-Instruct-Q4_K_M-GGUF  bartowski/Qwen2.5-3B-Instruct-GGUF
                filename="qwen2.5-7b-instruct-q4_k_m.gguf",
                local_dir="./model",
                local_dir_use_symlinks=False)
            
            print("Модель скачена")

        else:
            print('Модель найдена')

    def generate_response(self,  quastion, context):
        user_content = f"""
                        Контекс (Из документации - manual.pdf):
                        {context}

                        Вопрос пользователя: 
                        {quastion}
                        """
        
        response = self.llm.create_chat_completion(
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content' :user_content}
            ],
            temperature=0.1, # выдает строгие ответы из векторной базы данных
            top_p=0.9,
            max_tokens=512,
            stop=["<|im_end|>", "Вопрос:", "Контекст:"] # стоп-слова
        )

        answer = response['choices'][0]['message']['content']
        statistic = response['usage']

        return {"answer": answer,
                 "completion_tokens": statistic["completion_tokens"],
                 "prompt_tokens": statistic["prompt_tokens"]}

