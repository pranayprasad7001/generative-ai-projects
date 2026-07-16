import requests
import json
import gradio as gr 

url = "http://localhost:11434/api/generate"

headers = {
    "Content-Type": "application/json",
}

# Global history list
history = []

def generate_response(prompt):
    # 1. Append the user's input with a clear label
    history.append(f"User: {prompt}")
    
    # 2. Join the history into a single text block
    final_prompt = "\n".join(history)
    
    body = {
        "model": "amy",
        "prompt": final_prompt,
        "stream": False
    }

    response = requests.post(url, headers=headers, data=json.dumps(body))

    if response.status_code == 200:
        response_data = json.loads(response.text)
        actual_response = response_data['response']
        
        # 3. Append the model's generated answer to the history
        history.append(f"Amy: {actual_response}")
        
        return actual_response
    else:
        error_msg = f"Error: {response.text}"
        print(error_msg)
        return error_msg

# Simple Interface
interface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=4, placeholder="Ask me anything about code..."),
    outputs=gr.Textbox(lines=4),
    title="Amy - Multi-language Code Assistant",
    description="Baseline version using requests and gr.Interface"
)

if __name__ == "__main__":
    interface.launch()