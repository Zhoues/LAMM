from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer, pipeline

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid

app = Flask(__name__)
CORS(app)

model_name_or_path = "../model_zoo/llama2_ckpt/70b_chat"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,device_map="auto",trust_remote_code=False,revision="main")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

@app.route('/text', methods=['POST'])
def text():

    input_text = request.form.get('text')
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    generate_ids = model.generate(inputs=inputs.input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=4096)
    generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    output = generated_text.replace(input_text, '').strip()

    return jsonify({'result': 1, 'answer': output})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=25542)