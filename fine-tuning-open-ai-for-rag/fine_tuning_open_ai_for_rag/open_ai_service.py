from openai import OpenAI
from tenacity import retry, wait_exponential

class OpenAiService:
    def __init__(self, open_api_client):
        self.open_api_client = open_api_client
    
    # Function to get prompt messages
    def get_prompt(row):
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"""Answer the following Question based on the Context only. Only answer from the Context. If you don't know the answer, say 'I don't know'.
                    Question: {row.question}\n\n
                    Context: {row.context}\n\n
                    Answer:\n""",
            },
        ]
        
    # Main function to answer question
    def answer_question(self, row, prompt_func=get_prompt, model="gpt-4o-mini"):
        messages = prompt_func(row)
        response = self.api_call(messages, model)
        
        return response.choices[0].message.content
        
    # Function with tenacity for retries
    @retry(wait=wait_exponential(multiplier=1, min=2, max=6))
    def api_call(self, messages, model):
        return self.open_api_client.chat.completions.create(
            model = model,
            messages = messages,
            stop=["\n\n"],
            max_tokens = 100,
            temperature = 0.0,
        )
    
    def create(self, model_id):
        return self.open_api_client.chat.completions.create(
            model = model_id,
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi, how can I help you today?"},
                {
                    "role": "user",
                    "content": "Can you answer the following question based on the given context? If not, say, I don't know:\n\nQuestion: What is the capital of France?\n\nContext: The capital of Mars is Gaia. Answer:",
                }]
            )