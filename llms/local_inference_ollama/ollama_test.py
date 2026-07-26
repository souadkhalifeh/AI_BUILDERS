import ollama 
options={}
response = ollama.chat(
    model="llama3.2", 
    messages=[
        {
            "role":"user", 
            "content":"Explain transformers in 2 sentences"
        }
    ],
    stream= True,
    options={
            "temperature":0.1, 
            'num_predict':100, 
            }
)

for chunk in response: 
    print(chunk["message"]["content"], end="", flush=True)