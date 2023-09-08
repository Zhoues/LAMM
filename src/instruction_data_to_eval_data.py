import json

def transform_json(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    transformed_data = []
    for item in data:
        conversations = item['conversations']
        for i in range(0, len(conversations), 2):
            new_item = {}
            new_item['query'] = conversations[i]['value']
            new_item['sentences'] = [conversations[i+1]['value']]
            new_item['id'] = item['id']
            new_item['image'] = "minecraft_images/" + item['image']
            new_item['src_image'] = item['src_image']
            transformed_data.append(new_item)

    with open(output_file, 'w') as f:
        json.dump(transformed_data, f, ensure_ascii=False, indent=4)

transform_json('../datasets/2D_Instruct/LAMM_MineClip_36k/LAMM_MineClip_36k_instruct_simple_reply.json', 
               '../datasets/2D_Benchmark/meta_file/minecraft_minecraft.json')