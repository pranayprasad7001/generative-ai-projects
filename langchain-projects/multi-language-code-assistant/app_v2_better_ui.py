import gradio as gr
from ollama import chat
from ollama import list as list_local_models


# Dynamic Model Fetching
def get_installed_models():
    """Fetches local Ollama models. Falls back to defaults if Ollama is offline."""
    try:
        model_list = list_local_models()
        names = [m["name"] for m in model_list.get("models", [])]
        return names if names else ["amy:latest"]
    except Exception:
        return ["amy:latest", "llama3:latest", "mistral:latest"]


# Chat Processing Function
def respond(user_message, history, model, temperature):
    """
    Handles the chat turn and sanitizes Gradio's history format for Ollama.
    """
    if not user_message.strip():
        return history, ""

    # Append the user's message to the Gradio conversation history
    history.append({"role": "user", "content": user_message})

    # Gradio converts content into lists (e.g., [{'text': 'hi', 'type': 'text'}])
    # Ollama strictly requires a plain string. We parse it here.
    ollama_history = []
    for msg in history:
        raw_content = msg["content"]
        
        if isinstance(raw_content, list):
            # Extract the actual text from Gradio's dictionary list
            parsed_text = "".join(
                item.get("text", "") for item in raw_content if isinstance(item, dict)
            )
            ollama_history.append({"role": msg["role"], "content": parsed_text})
        else:
            # If it's already a string, just pass it through
            ollama_history.append({"role": msg["role"], "content": str(raw_content)})


    try:
        # Call Ollama directly with the SANITIZED history array
        response = chat(
            model=model,
            messages=ollama_history,
            options={"temperature": temperature}
        )
        assistant_response = response["message"]["content"]
        
        # Append the assistant's reply to the Gradio history
        history.append({"role": "assistant", "content": assistant_response})

    except Exception as e:
        error_msg = f"❌ **Ollama Connection Error:**\n\n{e}\n\n*Make sure your Ollama service is running locally (`ollama run {model}`).*"
        history.append({"role": "assistant", "content": error_msg})

    return history, ""


# Interface Layout 
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
)

# App-level settings like theme and title are now initialized here
with gr.Blocks(theme=custom_theme, title="Amy Code Assistant") as demo:
    gr.Markdown("# 🤖 Amy - Multi-language Code Assistant")
    gr.Markdown("Review, debug, explain, and write code using your local Ollama models.")
    
    with gr.Row():
        # Left Sidebar
        with gr.Column(scale=1, min_width=250):
            gr.Markdown("### ⚙️ System Settings")
            
            model_selector = gr.Dropdown(
                choices=get_installed_models(),
                value=get_installed_models()[0],
                label="Select Local Model",
                interactive=True,
            )
            
            temp_slider = gr.Slider(
                minimum=0.0,
                maximum=1.2,
                value=0.7,
                step=0.1,
                label="Temperature (Creativity)",
            )
            
            clear_btn = gr.Button("🗑️ Clear Conversation", variant="secondary")
            
        # Right Main Panel
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                label="Assistant Window",
                height=550,
                buttons=["copy"],     
            )
            
            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="Ask anything about programming...",
                    lines=2,
                    scale=8,
                    show_label=False,
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)

    # Event Handlers
    user_input.submit(
        fn=respond,
        inputs=[user_input, chatbot, model_selector, temp_slider],
        outputs=[chatbot, user_input]
    )
    
    submit_btn.click(
        fn=respond,
        inputs=[user_input, chatbot, model_selector, temp_slider],
        outputs=[chatbot, user_input]
    )
    
    clear_btn.click(fn=lambda: [], inputs=None, outputs=chatbot)


if __name__ == "__main__":
    demo.launch()