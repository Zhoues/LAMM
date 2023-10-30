import os
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from copy import deepcopy
import time
from model.openlamm import LAMMPEFTModel
import torch
import json
import argparse
from conversations import conv_templates, conversation_dict
from tqdm import tqdm
from bigmodelvis import Visualization

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid

app = Flask(__name__)
CORS(app)

INPUT_KEYS = ['image_path', 'images', 'pcl_path']
SYS_MSG = """
You are an AI visual assistant specialized in the domain of Minecraft, capable of analyzing a single image from the game, and engaging in a chat between a curious human and an artificial intelligence assistant. The assistant provides detailed, polite, and insightful answers to the human's queries about the image.
"""

def generate_conversation_text(args, input, history):
    """get all conversation text

    :param args args: input args
    :param str question: current input from user
    :param list history: history of conversation, [(q, a)]
    """
    assert input is not None or len(input) > 0, "input is empty!"
    conv = conv_templates[args.conv_mode]
    SYS_MSG = conversation_dict[args.task_type]
    prompts = ''
    prompts += SYS_MSG
    if len(history) > 0:
        print("{} Q&A found in history...".format(len(history)))
    for q, a in history:
        prompts += "{} {}: {}\n{} {}: {}\n".format(conv.sep, conv.roles[0], q.replace('<image>', '').replace('\n', ''), conv.sep, conv.roles[1], a)
    prompts += "{} {}: {}\n".format(conv.sep, conv.roles[0], input)
    return prompts
    


def predict(
    args,
    model,
    input,
    images, 
    image_path, 
    pcl_path,
    chatbot, 
    max_length, 
    top_p, 
    temperature, 
    history, 
    modality_cache, 
    show_prompt=False
):
    if image_path is None and pcl_path is None and images is None:
        return [(input, "There is no input data provided! Please upload your data and start the conversation.")]
    else:
        pass

    start = time.time()
    prompt_text = generate_conversation_text(args, input, history)
    if show_prompt:
        print(f'[!] prompt text: \n\t{prompt_text}', flush=True)
    response = model.generate({
        'prompt': [prompt_text] if not isinstance(prompt_text, list) else prompt_text,
        'image_paths': [image_path] if image_path else [],
        'pcl_paths': [pcl_path] if pcl_path else [],
        'images': [images] if images else [],
        'top_p': top_p,
        'temperature': temperature,
        'max_tgt_len': max_length,
        'modality_embeds': modality_cache
    }, is_web_server=True)
    # chatbot.append((parse_text(input), parse_text(response)))
    history.append((input, response))
    return chatbot, history, modality_cache, time.time() - start

def make_input_dict(args, vision_path):
    
    input_dict = dict()
    for key in INPUT_KEYS:
        if key.split('_')[0] == args.vision_type:
            input_dict[key] = vision_path
        else:
            input_dict[key] = None
    return input_dict


# WARNING: Remenber to update [delta_ckpt_path], [llm_ckpt_path], [conv_mode], [task_type] and [vision_root_path]
# 'conv_mode': 'vicuna_v1_1',
# 'conv_mode': 'default',
model_arg_dict = {
    'answer_file': '../answers/answer.txt',
    'conv_mode': 'llama2',
    'delta_ckpt_path': '../ckpt/Mine_52k_LLaMA2_7b_chat_epoch_3_mlp2x_gelu/pytorch_model.pt',
    'detail_log': False,
    'encoder_ckpt_path': '../model_zoo/mineclip_ckpt/mineclip_image_encoder_vit-B_196tokens.pth',
    'encoder_pretrain': 'mineclip',
    'force_test': False,
    'llm_ckpt_path': '../model_zoo/llama2_ckpt/7b_chat/',
    'lora_alpha': 32,
    'lora_dropout': 0.1,
    'lora_r': 32,
    'lora_target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    'max_tgt_len': 2048,
    'model': 'lamm_peft',
    'num_round': 100,
    'num_vision_token': 196,
    'question_file': 'conv.txt',
    'stage': 2,
    'task_type': 'minecraft',
    'temperature': 1.0,
    'top_p': 0.9,
    'vision_feature_type': 'local',
    'vision_output_layer': -2,
    'vision_root_path': '../images',
    'vision_type': 'image'
 }

parser = argparse.ArgumentParser()

for key, value in model_arg_dict.items():
    parser.add_argument(f'--{key}', default=value)

args = parser.parse_args()

# load model
model = LAMMPEFTModel(**args.__dict__)
delta_ckpt = torch.load(args.delta_ckpt_path, map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
print(f'[!] merging LoRA weights ...')
model = model.eval().half().cuda()
Visualization(model).structure_graph()
print(f'[!] init the LLM over ...')

if os.path.isfile(args.delta_ckpt_path):
        print("[!] Loading delta checkpoint: {}...".format(args.delta_ckpt_path))
        delta_ckpt = torch.load(args.delta_ckpt_path, map_location=torch.device('cpu'))
        model.load_state_dict(delta_ckpt, strict=False)
elif args.force_test:
    print("[!] Loading vicuna checkpoint: {}... while {} not found!".format(args.llm_ckpt_path, args.delta_ckpt_path))
else:
    raise ValueError("delta checkpoint not exists!")

model= model.eval().half().cuda()


UPLOAD_FOLDER = model_arg_dict['vision_root_path']
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/image', methods=['POST'])
def image():
    history = []

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    unique_id = str(uuid.uuid4())
    filename = secure_filename(unique_id + '.jpg')
    vision_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(vision_path)
    input_dict = make_input_dict(args, vision_path)

    input_text = request.form.get('text')

    _, history, _, item_time = predict(
        args=args,
        model=model,
        input=input_text,
        **input_dict,
        chatbot=[],
        max_length=args.max_tgt_len,
        top_p=args.top_p,
        temperature=args.temperature,
        history=history,
        modality_cache=[],
    )

    is_del = input_text = request.form.get('is_del')
    if is_del is not None and int(is_del) == 1:
        os.remove(vision_path)

    output = history[-1][1][0].split("\n##")[0]
    
    return jsonify({'result': 1, 'answer': output})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=25542)
