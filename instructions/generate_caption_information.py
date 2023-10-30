import re
import os
import pickle
import pdb
from collections import Counter
import json
import random
from tqdm import tqdm

from utils import *

instruction_data_list = []
data_path = "/mnt/petrelfs/share_data/liuqichang/finaldata_20/mob_observation_final_20"

# mob_observation
observation_path = os.path.join(data_path,"lidar_blockname")
for filepath, dirnames, filenames in os.walk(observation_path):
    for filename in tqdm(filenames):
        # get rays_block_name_list and id
        file_path = os.path.join(filepath,filename)
        block_file = open(file_path,'rb')
        rays_block_name_list = pickle.load(block_file).tolist()
        id = filename[:-4]

        # get rays_entity_name_list
        entity_filename = filename
        entity_file_path = os.path.join(os.path.join(data_path,"lidar_entityname"), entity_filename)
        entity_file = open(entity_file_path,'rb')
        rays_entity_name_list = pickle.load(entity_file).tolist()

        # get location_stats
        stats_filename = id + '.txt'
        stats_filepath = os.path.join(os.path.join(data_path,"plain_observation"), stats_filename)
        location_stats = extract_location_stats(stats_filepath)

        # Multi-Round QA
        conversations = []
        conversations.extend(create_conversation_about_caption_one_round(rays_block_name_list, rays_entity_name_list, location_stats))
        conversations.extend(create_conversation_about_light_one_round(location_stats))
        conversations.extend(create_conversation_about_block_one_round(rays_block_name_list))
        conversations.extend(create_conversation_about_creature_one_round(rays_entity_name_list))
        conversations.extend(create_conversation_about_time_one_round(location_stats))
        conversations.extend(create_conversation_about_weather_one_round(location_stats))
        conversations.extend(create_conversation_about_biome_one_round(location_stats))
        instruction_data = create_instruction_data(id, "jpg", conversations)
        instruction_data_list.append(instruction_data)

        # One-Round QA (except Block & Creature)
        conversations = []
        conversations.extend(create_conversation_about_light_one_round(location_stats))
        instruction_data = create_instruction_data(id, "jpg", conversations)
        instruction_data_list.append(instruction_data)

        conversations = []
        conversations.extend(create_conversation_about_time_one_round(location_stats))
        instruction_data = create_instruction_data(id, "jpg", conversations)
        instruction_data_list.append(instruction_data)

        conversations = []
        conversations.extend(create_conversation_about_weather_one_round(location_stats))
        instruction_data = create_instruction_data(id, "jpg", conversations)
        instruction_data_list.append(instruction_data)

        conversations = []
        conversations.extend(create_conversation_about_biome_one_round(location_stats))
        instruction_data = create_instruction_data(id, "jpg", conversations)
        instruction_data_list.append(instruction_data)
        
        # One-Round QA (for Block & Creature)
        conversations = []
        conversations.extend(create_conversation_about_block_one_round(rays_block_name_list))
        instruction_data = create_instruction_data(id, "jpg", conversations)
        instruction_data_list.append(instruction_data)

        conversations = []
        conversations.extend(create_conversation_about_creature_one_round(rays_entity_name_list))
        instruction_data = create_instruction_data(id, "jpg", conversations)
        instruction_data_list.append(instruction_data)

        ## One-Round QA (for Caption)
        conversations = []
        conversations.extend(create_conversation_about_caption_one_round(rays_block_name_list, rays_entity_name_list, location_stats))
        instruction_data = create_instruction_data(id, "jpg", conversations)
        instruction_data_list.append(instruction_data)

        conversations = []
        conversations.extend(create_conversation_about_caption_one_round(rays_block_name_list, rays_entity_name_list, location_stats))
        instruction_data = create_instruction_data(id, "jpg", conversations)
        instruction_data_list.append(instruction_data)

        conversations = []
        conversations.extend(create_conversation_about_caption_one_round(rays_block_name_list, rays_entity_name_list, location_stats))
        instruction_data = create_instruction_data(id, "jpg", conversations)
        instruction_data_list.append(instruction_data)


        
# dump as json
json_data = json.dumps(instruction_data_list)
with open('datasets/2D_Instruct/Mine_52k/Mine_52k_instruct_caption.json','w') as file:
    file.write(json_data)