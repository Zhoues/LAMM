from transformers import AutoModel, AutoTokenizer
from copy import deepcopy
import gradio as gr
import mdtex2html
from model.openlamm import LAMMPEFTModel
import torch
import json
import os

from conversations import conv_templates
SYS_MSG = """
You are an AI visual assistant specialized in the domain of Minecraft, capable of analyzing a single image from the game, and engaging in a chat between a curious human and an artificial intelligence assistant. The assistant provides detailed, polite, and insightful answers to the human's queries about the image.
"""


# init the model
args = {
    'model': 'openllama_peft',
    'encoder_ckpt_path': '../model_zoo/mineclip_ckpt/mineclip_image_encoder_vit-B_196tokens.pth',
    'llm_ckpt_path': '../model_zoo/vicuna_ckpt/13b_v1.5/',
    'delta_ckpt_path': '../ckpt/Mine_36k_simple_reply_Vicuna_13b_v1.5/pytorch_model.pt',
    'conv_mode': 'vicuna_v1_5',
    'task_type': 'minecraft',

    'stage': 2,
    'max_tgt_len': 128,
    'lora_r': 32,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
    'lora_target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    'vision_type': 'image',
    'vision_feature_type': 'local',
    'num_vision_token': 196,
    'encoder_pretrain': 'mineclip',
    'system_header': True,
}

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
model = LAMMPEFTModel(**args)
delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=torch.device(device))
model.load_state_dict(delta_ckpt, strict=False)
model = model.eval().half().cuda()
print(f'[!] init the 13b model over ...')

"""Override Chatbot.postprocess"""


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    if text.endswith("##"):
        text = text[:-2]
    return text


def re_predict(
    input, 
    image_path, 
    chatbot, 
    max_length, 
    top_p, 
    temperature, 
    history, 
    modality_cache, 
):
    # drop the latest query and answers and generate again
    q, a = history.pop()
    chatbot.pop()
    return predict(q, image_path, chatbot, max_length, top_p, temperature, history, modality_cache)

def generate_conversation_text(args, input, history):
    """get all conversation text

    :param args args: input args
    :param str question: current input from user
    :param list history: history of conversation, [(q, a)]
    """
    assert input is not None or len(input) > 0, "input is empty!"
    conv = conv_templates[args['conv_mode']]
    SYS_MSG = conversation_dict[args['task_type']]
    prompts = ''
    prompts += SYS_MSG
    if len(history) > 0:
        print("{} Q&A found in history...".format(len(history)))
    for q, a in history:
        prompts += "{} {}: {}\n{} {}: {}\n".format(conv.sep, conv.roles[0], q.replace('<image>', '').replace('\n', ''), conv.sep, conv.roles[1], a)
    prompts += "{} {}: {}\n".format(conv.sep, conv.roles[0], input)
    return prompts


def predict(
    input, 
    image_path, 
    chatbot, 
    max_length, 
    top_p, 
    temperature, 
    history, 
    modality_cache, 
):
    if image_path is None:      # 
        return [(input, "There is no input data provided! Please upload your data and start the conversation.")]
    else:
        print(f'[!] image path: {image_path}\n')        # [!] audio path: {audio_path}\n[!] video path: {video_path}\n[!] thermal path: {thermal_path}')

    prompt_text = generate_conversation_text(args, input, history)

    response = model.generate({
        'prompt': [prompt_text] if not isinstance(prompt_text, list) else prompt_text,
        'image_paths': [image_path] if image_path else [],
        'top_p': top_p,
        'temperature': temperature,
        'max_tgt_len': max_length,
        'modality_embeds': modality_cache
    })
    if isinstance(response, list):
        response = response[0]
    chatbot.append((parse_text(input), parse_text(response)))
    history.append((input, response))
    return chatbot, history, modality_cache


def reset_user_input():
    return gr.update(value='')

def reset_dialog():
    return [], []

def reset_state():
    return None, [], [], []


with gr.Blocks(scale=4) as demo:
    # gr.Image("./images/lamm_title.png", show_label=False, height=50)
    gr.HTML(
        """
        <p>
        <p align="center">
            <font size='4'>
            <a href="https://openlamm.github.io/" target="_blank">🏠 Home Page</a> • <a href="https://github.com/OpenLAMM/LAMM" target="_blank">🌏 Github</a> • <a href="https://arxiv.org/pdf/2306.06687.pdf" target="_blank">📰 Paper</a> • <a href="https://www.youtube.com/watch?v=M7XlIe8hhPk" target="_blank">▶️ YouTube </a> • <a href="https://www.bilibili.com/video/BV1kN411D7kt/?share_source=copy_web&vd_source=ab4c734425ed0114898300f2c037ac0b" target="_blank"> 📺 Bilibili</a> • <a href="https://opendatalab.com/LAMM" target="_blank">📀 Data</a> • <a href="https://huggingface.co/openlamm" target="_blank">📦 LAMM Models</a>
            </font>
        </p>
        </p>
        """
    )
    # gr.HTML("""<h1>LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark</h1>""")
    # gr.Markdown(
    # """
    # <p>
    
    # <a href="https://arxiv.org/pdf/2306.06687.pdf" target="_blank"><img src="https://img.shields.io/badge/arxiv-PDF-red"/></a> 
    
    # <a href="https://openlamm.github.io" target="_blank"><img src="https://img.shields.io/badge/LAMM-HomePage-blue"/></a> 

    # <a href="https://opendatalab.com/LAMM" target="_blank"><img src="https://img.shields.io/badge/LAMM-Dataset-green"/></a> 
    
    # <a href="https://www.youtube.com/watch?v=M7XlIe8hhPk" target="_blank"><img src="https://img.shields.io/badge/video-Youtube-red"/></a>
    
    # <a href="https://www.bilibili.com/video/BV1kN411D7kt/?share_source=copy_web&vd_source=ab4c734425ed0114898300f2c037ac0b" target="_blank"><img src="https://img.shields.io/badge/video-Bilibili-blue"/></a>
    
    # <a href="https://github.com/OpenLAMM/LAMM" target="_blank"><img src="https://img.shields.io/badge/Repo-Github-white"/></a> 

    # <a href="https://huggingface.co/openlamm" target="_blank"><img src="https://img.shields.io/badge/Models-huggingface-yellow"/></a> 
    
    # <img src="https://img.shields.io/github/stars/OpenLAMM/LAMM.svg?style=social&label=Star"/>
    # </p>
    # Drop your image & Start talking with LAMM models.
    # """)

    with gr.Row(scale=1):
        with gr.Column(scale=1):
            image_path = gr.Image(type="filepath", label="Image", value=None).style(height=600)
    
        chatbot = gr.Chatbot(scale=1).style(height=600)

    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(container=False)
            with gr.Column(min_width=32, scale=1):
                with gr.Row(scale=1):
                    submitBtn = gr.Button("Submit", variant="primary")
                with gr.Row(scale=1):
                    resubmitBtn = gr.Button("Resubmit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 600, value=256, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.01, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.9, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])
    modality_cache = gr.State([])

    submitBtn.click(
        predict, [
            user_input, 
            image_path, 
            chatbot, 
            max_length, 
            top_p, 
            temperature, 
            history, 
            modality_cache,
        ], [
            chatbot, 
            history,
            modality_cache
        ],
        show_progress=True
    )

    resubmitBtn.click(
        re_predict, [
            user_input, 
            image_path, 
            chatbot, 
            max_length, 
            top_p, 
            temperature, 
            history, 
            modality_cache,
        ], [
            chatbot, 
            history,
            modality_cache
        ],
        show_progress=True
    )

    submitBtn.click(reset_user_input, [], [user_input])
    emptyBtn.click(reset_state, outputs=[
        image_path,
        chatbot, 
        history, 
        modality_cache
    ], show_progress=True)
    
import socket
print(socket.gethostbyname(socket.gethostname()))
demo.queue().launch(server_name="0.0.0.0", server_port=25510)