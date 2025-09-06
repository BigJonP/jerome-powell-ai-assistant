import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from functools import lru_cache
import logging

MODEL_NAME = "BoostedJonP/powell-phi3-mini"

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
logger.info("Starting Jerome Powell AI Assistant...")


@lru_cache(maxsize=1)
def load_model():
    """Load the fine-tuned Jerome Powell model"""
    logger.info(f"Loading model: {MODEL_NAME}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir="/tmp/model_cache",
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",
            use_cache=True,
            cache_dir="/tmp/model_cache",
        )
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None

    model.generation_config.use_cache = True
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


model, tokenizer = load_model()


def generate_powell_response(question, max_length=256, num_beams=3, temperature=0.3):
    """Generate a response in Jerome Powell's style"""

    if model is None or tokenizer is None:
        return "❌ Model failed to load. Please refresh the page and try again."

    if not question.strip():
        return (
            "Please ask a question about monetary policy, economics, or Federal Reserve operations."
        )

    prompt = f"Question: {question.strip()}\nAnswer:"

    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=False,
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            generation_config = {
                "max_new_tokens": max_length,
                "num_beams": num_beams,
                "early_stopping": True,
                "do_sample": True,
                "temperature": temperature,
                "repetition_penalty": 1.1,
                "use_cache": True,
                "output_scores": False,
                "return_dict_in_generate": False,
            }

            outputs = model.generate(**inputs, **generation_config)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "Answer:" in response:
            answer = response.split("Answer:")[-1].strip()
        else:
            answer = response[len(prompt) :].strip()

        return (
            answer
            if answer
            else "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
        )

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"❌ Error generating response: {str(e)}"


custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    max-width: 1200px;
    margin: 0 auto;
}

.header-text {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.5rem;
    font-weight: bold;
    margin-bottom: 0.5rem;
}

.subtitle-text {
    text-align: center;
    color: #666;
    font-size: 1.2rem;
    margin-bottom: 2rem;
}

.example-box {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #667eea;
    margin: 1rem 0;
}

.footer-text {
    text-align: center;
    color: #666;
    font-size: 0.9rem;
    margin-top: 2rem;
    padding: 1rem;
    border-top: 1px solid #eee;
}

/* Make buttons more prominent */
.primary-button {
    background: linear-gradient(45deg, #667eea, #764ba2) !important;
    border: none !important;
    color: white !important;
}

/* Response styling */
.response-box {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 4px solid #28a745;
    font-size: 1.1rem;
    line-height: 1.6;
}
"""


def create_interface():
    with gr.Blocks(
        css=custom_css,
        title="Jerome Powell AI Assistant | Federal Reserve Q&A",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.HTML(
            """
        <div style="display: flex; align-items: center; justify-content: center; gap: 2rem; margin-bottom: 2rem;">
            <div style="flex: 1; text-align: center;">
                <div class="header-text">🏦 Jerome Powell AI Assistant</div>
                <div class="subtitle-text">
                    Fine-tuned Phi3-Mini model trained on Federal Reserve Chairman Jerome Powell's Q&A sessions
                </div>
            </div>
            <div style="flex-shrink: 0;">
                <img src="https://storage.googleapis.com/kaggle-datasets-images/8130068/12853913/f5d1487cf839f69edc1fcde3d30a583a/dataset-cover.jpeg?t=2025-08-24-13-28-39" 
                     alt="Jerome Powell" 
                     style="width: 200px; height: 200px; border-radius: 50%; object-fit: cover; border: 4px solid #667eea; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            </div>
        </div>
        """
        )

        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="💬 Ask about monetary policy, economics, or Federal Reserve operations",
                    placeholder="e.g., What factors influence Federal Reserve interest rate decisions?",
                    lines=3,
                    max_lines=5,
                )

                with gr.Row():
                    submit_btn = gr.Button("🎯 Ask Jerome Powell AI", variant="primary", scale=2)
                    clear_btn = gr.Button("🔄 Clear", scale=1)

        with gr.Accordion("⚙️ Advanced Settings", open=False):
            with gr.Row():
                max_length = gr.Slider(
                    minimum=64,
                    maximum=512,
                    value=256,
                    step=32,
                    label="Max Response Length",
                    info="Longer responses may be more detailed but take more time",
                )
                num_beams = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=3,
                    step=1,
                    label="Number of Beams",
                    info="Higher values = better quality but slower generation (beam search)",
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.3,
                    step=0.1,
                    label="Temperature",
                    info="Higher values = more creative/random responses (affects beam search diversity)",
                )

        response_output = gr.Textbox(
            label="💼 Jerome Powell AI Response",
            lines=8,
            max_lines=15,
            show_copy_button=True,
            container=True,
        )

        gr.HTML(
            """
        <div class="footer-text">
            <h3>📊 Model Information</h3>
            <p>
            <strong>Base Model:</strong> Microsoft Phi3-Mini<br>
            <strong>Fine-tuning:</strong> Specialized on Jerome Powell Q&A data<br>
            <strong>Model Hub:</strong> <a href="https://huggingface.co/BoostedJonP/powell-phi3-mini" target="_blank">BoostedJonP/powell-phi3-mini</a><br>
            <strong>Dataset:</strong> <a href="https://huggingface.co/datasets/BoostedJonP/JeromePowell-SFT" target="_blank">BoostedJonP/JeromePowell-SFT</a><br>
            <strong>Repo:</strong> <a href="https://github.com/BigJonP/powell-phi3-sft" target="_blank">BigJonP/powell-phi3-sft</a><br>
            <strong>Author:</strong> <a href="https://github.com/BigJonPP" target="_blank">Jonathan Paserman</a>
            </p>
            
            <p><em>⚠️ Disclaimer: This AI model provides educational insights based on training data and should not be considered as official Federal Reserve communication or financial advice. Always consult official Fed sources for authoritative information.</em></p>
        </div>
        """
        )

        submit_btn.click(
            fn=generate_powell_response,
            inputs=[question_input, max_length, num_beams, temperature],
            outputs=response_output,
            show_progress=True,
        )

        clear_btn.click(lambda: ("", ""), outputs=[question_input, response_output])

        question_input.submit(
            fn=generate_powell_response,
            inputs=[question_input, max_length, num_beams, temperature],
            outputs=response_output,
            show_progress=True,
        )

    return demo


demo = create_interface()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
