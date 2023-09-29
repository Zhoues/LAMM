from transformers import AutoTokenizer, LlamaForCausalLM

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid

app = Flask(__name__)
CORS(app)


PATH_TO_CONVERTED_WEIGHTS = "../model_zoo/vicuna_ckpt/13b_v1.5_16k"
PATH_TO_CONVERTED_TOKENIZER= "../model_zoo/vicuna_ckpt/13b_v1.5_16k"

tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
model = model.eval().half().cuda()

@app.route('/text', methods=['POST'])
def text():

    input_text = request.form.get('text')
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = inputs.to(model.device)

    generate_ids = model.generate(inputs.input_ids, max_length=4096)
    generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    output = generated_text.replace(input_text, '').strip()

    return jsonify({'result': 1, 'answer': output})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=25542)