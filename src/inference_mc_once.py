import os
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from copy import deepcopy
import time
from model.openlamm import LAMMPEFTModel
import torch
import json
import argparse
from conversations import conv_templates
from tqdm import tqdm
from bigmodelvis import Visualization

INPUT_KEYS = ['image_path', 'images', 'pcl_path']
SYS_MSG = """
You are an AI visual assistant specialized in the domain of Minecraft, capable of analyzing a single image from the game, and engaging in a chat between a curious human and an artificial intelligence assistant. The assistant provides detailed, polite, and insightful answers to the human's queries about the image.
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lamm_peft')
    parser.add_argument('--vision_type', type=str, default='pcl', choices=('image', 'pcl'))
    parser.add_argument('--encoder_pretrain', type=str, default='epcl')
    parser.add_argument('--encoder_ckpt_path', type=str, 
                        help="path of vision pretrained model; CLIP use default path in cache")
    parser.add_argument('--llm_ckpt_path', type=str, default='../model_zoo/vicuna_ckpt/13b_v0')
    parser.add_argument('--delta_ckpt_path', type=str, default='../model_zoo/pandagpt_ckpt/13b/pytorch_model.pt')
    parser.add_argument('--force_test', action='store_true', help='whether to force test mode, ignore file missing')
    parser.add_argument('--stage', type=int, default=2, help='has no function in testing')
    # Architecture configurations
    # Lora configurations
    parser.add_argument('--lora_r', type=int, default=32)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--lora_target_modules', nargs='+', default=['q_proj', 'k_proj', 'v_proj', 'o_proj'])
    # Embedding configurations
    parser.add_argument('--vision_feature_type', type=str, default='local', choices=('local', 'global'))
    parser.add_argument('--vision_output_layer', type=int, default=-1, choices=(-1, -2), help='the layer to output visual features; -1 means global from last layer')
    parser.add_argument('--num_vision_token', type=int, default=256) # the maximum sequence length
    # Data configurations
    parser.add_argument('--max_tgt_len', type=int, default=1024, help='max length of generated tokens')
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--conv_mode', type=str, default='simple')
    parser.add_argument('--task_type', type=str, default='conversation')
    parser.add_argument('--num_round', '-N', type=int, default=100, help='number of rounds of conversation')
    parser.add_argument('--question_file', type=str, default='conv.txt', help='conversation file')
    parser.add_argument('--vision_root_path', type=str, default='', help='image directory')
    parser.add_argument('--answer_file', type=str, default='../answers/answer.txt', help='answer file')
    parser.add_argument('--detail_log', action='store_true', help='whether to log detail conversation')
    args = parser.parse_args()
    
    if args.vision_feature_type == 'local':
        args.vision_output_layer = -2
        # args.num_vision_token = 256
    elif args.vision_feature_type == 'global':
        args.vision_output_layer = -1
        args.num_vision_token = 1
    else:
        raise NotImplementedError('NOT implement vision feature type: {}'.format(args.vision_feature_type))
    # make sure input
    assert len(args.vision_root_path) == 0 or os.path.exists(args.vision_root_path), "vision root directory not exists!"
    assert os.path.exists(args.delta_ckpt_path) or args.force_test, "delta checkpoint not exists and it's required!"
    assert os.path.exists(args.llm_ckpt_path), "vicuna checkpoint not exists!"
    assert args.encoder_pretrain == 'clip' or os.path.exists(args.encoder_ckpt_path), "vision checkpoint not exists!"
    args.max_tgt_len = args.max_tgt_len - 1 + args.num_vision_token
    if not os.path.isdir(os.path.dirname(args.answer_file)):
        os.makedirs(os.path.dirname(args.answer_file), exist_ok=True)
    args.max_tgt_len = max(args.max_tgt_len - 1 + args.num_vision_token, 2048)
    print(json.dumps(vars(args), indent=4, sort_keys=True))
    return args


def generate_conversation_text(args, input, history):
    """get all conversation text

    :param args args: input args
    :param str question: current input from user
    :param list history: history of conversation, [(q, a)]
    """
    assert input is not None or len(input) > 0, "input is empty!"
    conv = conv_templates[args.conv_mode]
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
    })
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

def main(args):
    model = LAMMPEFTModel(**args.__dict__)
    
    if os.path.isfile(args.delta_ckpt_path):
        print("[!] Loading delta checkpoint: {}...".format(args.delta_ckpt_path))
        delta_ckpt = torch.load(args.delta_ckpt_path, map_location=torch.device('cpu'))
        model.load_state_dict(delta_ckpt, strict=False)
    elif args.force_test:
        print("[!] Loading vicuna checkpoint: {}... while {} not found!".format(args.llm_ckpt_path, args.delta_ckpt_path))
    else:
        raise ValueError("delta checkpoint not exists!")

    model = model = model.eval().half().cuda()
    history = []
    vision_path = ""
    print("Input a file path: ", flush=True)

    while not os.path.isfile(vision_path):
        vision_path = input("Vision path: ")
        if not os.path.isfile(vision_path):
            print(f'{vision_path} not found!')
    print(f'[!] vision content path: {vision_path}', flush=True)
    input_dict = make_input_dict(args, vision_path)

    input_text = input()
    print(f'[!] ### Human: {input_text}', flush=True)

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
    import ipdb; ipdb.set_trace()
    print('[!] ### Assistant: {}'.format(history[-1][1][0].split("\n##")[0]))
    

if __name__ == '__main__':
    args = parse_args()
    print(json.dumps(vars(args), indent=4, sort_keys=True))
    main(args)
