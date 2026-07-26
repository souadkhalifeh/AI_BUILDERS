import ollama 
options={}
messages = [
    {
        "role":"system",
        "content":"You are a senior AI Engineer, preparing an educational content for high school students and first year university students."
    }
]
while True: 
    question = input("You:")
    
    messages.append(
         {
           "role":"user", 
            "content":question
           }
    )
    response = ollama.chat(
    model="llama3.2", 
    messages=messages, 
    stream=True, 
    options={
        "temperature":1.0
    }
)
    assistant_message=""
    for chunk in response: 
        token = chunk["message"]["content"]
        print(token, end="", flush=True)
        assistant_message += token
    messages.append(
        {
                     "role":"assistant",
                     "content":"assistant_message"
                     })
    print()