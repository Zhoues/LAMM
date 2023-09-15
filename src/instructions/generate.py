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

#biome_observation
# for filepath,dirnames,filenames in os.walk(r'/mnt/petrelfs/share_data/liuqichang/final_data2/biome_observation/lidar_blockname'):
#     for filename in filenames:
#         print(filename)

#         # get rays_block_name_list and id
#         file_path = os.path.join(filepath,filename)
#         block_file = open(file_path,'rb')
#         rays_block_name_list = pickle.load(block_file).tolist()
#         id = filename[:-4]

#         # get rays_entity_name_list
#         entity_filename = filename
#         entity_file_path = os.path.join('/mnt/petrelfs/share_data/liuqichang/final_data2/biome_observation/lidar_entityname', entity_filename)
#         entity_file = open(entity_file_path,'rb')
#         rays_entity_name_list = pickle.load(entity_file).tolist()

#         # get location_stats
#         stats_filename = id + '.txt'
#         stats_filepath = os.path.join('/mnt/petrelfs/share_data/liuqichang/final_data2/biome_observation/plain_observation', stats_filename)
#         location_stats = extract_location_stats(stats_filepath)
        
#         conversations = []
#         conversations.extend(create_conversation_about_light_one_round(location_stats, True))
#         conversations.extend(create_conversation_about_block_one_round(rays_block_name_list, False))
#         conversations.extend(create_conversation_about_creature_one_round(rays_entity_name_list, False))
#         conversations.extend(create_conversation_about_time_one_round(location_stats, False))
#         conversations.extend(create_conversation_about_weather_one_round(location_stats, False))
#         conversations.extend(create_conversation_about_biome_one_round(location_stats, False))
#         instruction_data = create_instruction_data(id, "jpg", conversations)
#         instruction_data_list.append(instruction_data)
#         # pdb.set_trace()

#         conversations = []
#         conversations.extend(create_conversation_about_block_one_round(rays_block_name_list, True))
#         conversations.extend(create_conversation_about_creature_one_round(rays_entity_name_list, False))
#         instruction_data = create_instruction_data(id, "jpg", conversations)
#         instruction_data_list.append(instruction_data)

#         conversations = []
#         conversations.extend(create_conversation_about_block_one_round(rays_block_name_list, True))
#         conversations.extend(create_conversation_about_creature_one_round(rays_entity_name_list, False))
#         instruction_data = create_instruction_data(id, "jpg", conversations)
#         instruction_data_list.append(instruction_data)

#         conversations = []
#         conversations.extend(create_conversation_about_block_one_round(rays_block_name_list, True))
#         conversations.extend(create_conversation_about_creature_one_round(rays_entity_name_list, False))
#         instruction_data = create_instruction_data(id, "jpg", conversations)
#         instruction_data_list.append(instruction_data)


# mob_observation
for filepath, dirnames, filenames in os.walk(r'/mnt/petrelfs/share_data/liuqichang/trial_data_mob0907/mob_observation_50/lidar_blockname'):
    for filename in tqdm(filenames):
        # get rays_block_name_list and id
        file_path = os.path.join(filepath,filename)
        block_file = open(file_path,'rb')
        rays_block_name_list = pickle.load(block_file).tolist()
        id = filename[:-4]

        # get rays_entity_name_list
        entity_filename = filename
        entity_file_path = os.path.join('/mnt/petrelfs/share_data/liuqichang/trial_data_mob0907/mob_observation_50/lidar_entityname', entity_filename)
        entity_file = open(entity_file_path,'rb')
        rays_entity_name_list = pickle.load(entity_file).tolist()

        # get location_stats
        stats_filename = id + '.txt'
        stats_filepath = os.path.join('/mnt/petrelfs/share_data/liuqichang/trial_data_mob0907/mob_observation_50/plain_observation', stats_filename)
        location_stats = extract_location_stats(stats_filepath)

        # Multi-Round QA
        conversations = []
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
        conversations.extend(create_conversation_about_block_one_round(rays_block_name_list))
        instruction_data = create_instruction_data(id, "jpg", conversations)
        instruction_data_list.append(instruction_data)

        conversations = []
        conversations.extend(create_conversation_about_block_one_round(rays_block_name_list))
        instruction_data = create_instruction_data(id, "jpg", conversations)
        instruction_data_list.append(instruction_data)

        conversations = []
        conversations.extend(create_conversation_about_creature_one_round(rays_entity_name_list))
        instruction_data = create_instruction_data(id, "jpg", conversations)
        instruction_data_list.append(instruction_data)

        conversations = []
        conversations.extend(create_conversation_about_creature_one_round(rays_entity_name_list))
        instruction_data = create_instruction_data(id, "jpg", conversations)
        instruction_data_list.append(instruction_data)

        
# dump as json
json_data = json.dumps(instruction_data_list)
with open('../datasets/2D_Instruct/Mine_36k/Mine_36k_instruct.json','w') as file:
    file.write(json_data)