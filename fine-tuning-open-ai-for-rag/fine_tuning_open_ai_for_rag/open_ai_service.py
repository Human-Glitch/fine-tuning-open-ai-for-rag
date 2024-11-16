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
    
    # Function with tenacity for retries
    @retry(wait=wait_exponential(multiplier=1, min=2, max=6))
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
        
    # Function with tenacity for retries
    @retry(wait=wait_exponential(multiplier=1, min=2, max=6))
    def create2(self, model_id):
        return self.open_api_client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Can you answer the following question based on the given context? If not, say, I don't know:\n\nQuestion: What is the capital of France?\n\nContext: The capital of Mars is Gaia. Answer:",
                },
                {
                    "role": "assistant",
                    "content": "I don't know",
                },
                {
                    "role": "user",
                    "content": "Question: Where did Maharana Pratap die?\n\nContext: Rana Pratap's defiance of the mighty Mughal empire, almost alone and unaided by the other Rajput states, constitute a glorious saga of Rajput valour and the spirit of self sacrifice for cherished principles. Rana Pratap's methods of guerrilla warfare was later elaborated further by Malik Ambar, the Deccani general, and by Emperor Shivaji.\nAnswer:",
                },
                {
                    "role": "assistant",
                    "content": "I don't know",
                },
                {
                    "role": "user",
                    "content": "Question: Who did Rana Pratap fight against?\n\nContext: In stark contrast to other Rajput rulers who accommodated and formed alliances with the various Muslim dynasties in the subcontinent, by the time Pratap ascended to the throne, Mewar was going through a long standing conflict with the Mughals which started with the defeat of his grandfather Rana Sanga in the Battle of Khanwa in 1527 and continued with the defeat of his father Udai Singh II in Siege of Chittorgarh in 1568. Pratap Singh, gained distinction for his refusal to form any political alliance with the Mughal Empire and his resistance to Muslim domination. The conflicts between Pratap Singh and Akbar led to the Battle of Haldighati. Answer:",
                },
                {
                    "role": "assistant",
                    "content": "Akbar",
                },
                {
                    "role": "user",
                    "content": "Question: Which state is Chittorgarh in?\n\nContext: Chittorgarh, located in the southern part of the state of Rajasthan, 233 km (144.8 mi) from Ajmer, midway between Delhi and Mumbai on the National Highway 8 (India) in the road network of Golden Quadrilateral. Chittorgarh is situated where National Highways No. 76 & 79 intersect. Answer:",
                },
            ])