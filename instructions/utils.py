import re
import os
import pickle
import pdb
from collections import Counter
import json
import random

from QA_MSG import *

def remove_duplicates_order_by_frequency(input_list):
    counter = Counter(input_list)
    return [item for item, count in counter.most_common()]

def create_conversation(human: str, gpt: str, first_time=False):
    human_dict = {
        "from": "human",
        "value": "<image>\n You have got an image of Minecraft from a first-person perspective. " + human if first_time else human
        }
    gpt_dict = {
        "from": "gpt",
        "value": gpt
        }
    return human_dict, gpt_dict

def create_instruction_data(id, format, conversations):
    instruction_data_creature_dict = {
        "id": f"{id}",
        "image": f"{id}.{format}",
        "conversations": conversations,
        "task_type": "minecraft",
        "src_image": "minecraft"
    }
    return instruction_data_creature_dict

def create_conversation_about_caption_one_round(rays_block_name_list, rays_entity_name_list, location_stats, first_time=False, random_id=None):

    random_id = random.randint(0, len(Caption_QA) - 1) if random_id is None else random_id

    block_answer = create_conversation_about_block_one_round(rays_block_name_list, first_time=first_time, random_id=random_id)[1]["value"]
    creature_answer = create_conversation_about_creature_one_round(rays_entity_name_list, first_time=first_time, random_id=random_id)[1]["value"]
    biome_answer = create_conversation_about_biome_one_round(location_stats, first_time=first_time, random_id=random_id)[1]["value"]
    brightness_answer = create_conversation_about_light_one_round(location_stats, first_time=first_time, random_id=random_id)[1]["value"]
    datetime_answer = create_conversation_about_time_one_round(location_stats, first_time=first_time, random_id=random_id)[1]["value"]
    weather_answer = create_conversation_about_weather_one_round(location_stats, first_time=first_time, random_id=random_id)[1]["value"]

    human_dict, gpt_dict = create_conversation(human=Caption_QA[random_id]['Q'],
                                                gpt=Caption_QA[random_id]['A'].format(block_answer=block_answer, creature_answer=creature_answer,
                                                biome_answer=biome_answer, brightness_answer=brightness_answer, datetime_answer=datetime_answer, weather_answer=weather_answer),
                                                first_time=first_time)

    return [human_dict, gpt_dict]


def create_conversation_about_block_one_round(rays_block_name_list, first_time=False, random_id=None):
    
    blocks = remove_duplicates_order_by_frequency(rays_block_name_list) 
    blocks = [block for block in blocks if block != 'air']
    random.shuffle(blocks)
    blocks_string = ', '.join(blocks)

    random_id = random.randint(0, len(Block_QA) - 1) if random_id is None else random_id

    human_dict, gpt_dict = create_conversation(human=Block_QA[random_id]['Q'],
                                                gpt=Block_QA[random_id]['A'].format(blocks_string),
                                                first_time=first_time)
    return [human_dict, gpt_dict]
    

def create_conversation_about_creature_one_round(rays_entity_name_list, first_time=False, random_id=None):
    
    creatures = remove_duplicates_order_by_frequency(rays_entity_name_list) 
    creatures = [creature for creature in creatures if creature != 'null']
    random.shuffle(creatures)
    creatures_string = ', '.join(creatures)

    if len(creatures_string) != 0:  
        random_id = random.randint(0, len(Creature_QA) - 1) if random_id is None else random_id
        human_dict, gpt_dict = create_conversation(human=Creature_QA[random_id]['Q'],
                                                gpt=Creature_QA[random_id]['A'].format(creatures_string),
                                                first_time=first_time)        
    else:
        random_id = random.randint(0, len(No_Creature_QA) - 1) if random_id is None else random_id
        human_dict, gpt_dict = create_conversation(human=No_Creature_QA[random_id]['Q'],
                                                gpt=No_Creature_QA[random_id]['A'],
                                                first_time=first_time)   
    return [human_dict, gpt_dict]

def create_conversation_about_light_one_round(location_stats, first_time=False, random_id=None):
    if location_stats['light_level'] <= 3:
        random_id = random.randint(0, len(Not_Bright_QA) - 1) if random_id is None else random_id
        human_dict, gpt_dict = create_conversation(human=Not_Bright_QA[random_id]['Q'],
                                                gpt=Not_Bright_QA[random_id]['A'],
                                                first_time=first_time)
    else:
        random_id = random.randint(0, len(Bright_QA) - 1) if random_id is None else random_id
        human_dict, gpt_dict = create_conversation(human=Bright_QA[random_id]['Q'],
                                                gpt=Bright_QA[random_id]['A'],
                                                first_time=first_time)
    return [human_dict, gpt_dict]

def create_conversation_about_time_one_round(location_stats, first_time=False, random_id=None):
    if not location_stats['can_see_sky']:
        random_id = random.randint(0, len(Not_Determine_Time_QA) - 1) if random_id is None else random_id
        human_dict, gpt_dict = create_conversation(human=Not_Determine_Time_QA[random_id]['Q'],
                                                gpt=Not_Determine_Time_QA[random_id]['A'],
                                                first_time=first_time)
    elif location_stats['sun_brightness'] + location_stats['sky_light_level'] > 0.6:
        random_id = random.randint(0, len(Datetime_QA) - 1) if random_id is None else random_id
        human_dict, gpt_dict = create_conversation(human=Datetime_QA[random_id]['Q'],
                                                gpt=Datetime_QA[random_id]['A'],
                                                first_time=first_time)
    else:
        random_id = random.randint(0, len(Nighttime_QA) - 1) if random_id is None else random_id
        human_dict, gpt_dict = create_conversation(human=Nighttime_QA[random_id]['Q'],
                                                gpt=Nighttime_QA[random_id]['A'],
                                                first_time=first_time)
    return [human_dict, gpt_dict]


def create_conversation_about_weather_one_round(location_stats, first_time=False, random_id=None):
    if not location_stats['can_see_sky']:
        random_id = random.randint(0, len(Not_Determine_Weather_QA) - 1) if random_id is None else random_id
        human_dict, gpt_dict = create_conversation(human=Not_Determine_Weather_QA[random_id]['Q'],
                                                gpt=Not_Determine_Weather_QA[random_id]['A'],
                                                first_time=first_time)
    elif not location_stats['is_raining']:
        random_id = random.randint(0, len(Sunny_QA) - 1) if random_id is None else random_id
        human_dict, gpt_dict = create_conversation(human=Sunny_QA[random_id]['Q'],
                                                gpt=Sunny_QA[random_id]['A'],
                                                first_time=first_time)
    else:
        random_id = random.randint(0, len(Rainy_QA) - 1) if random_id is None else random_id
        human_dict, gpt_dict = create_conversation(human=Rainy_QA[random_id]['Q'],
                                                gpt=Rainy_QA[random_id]['A'],
                                                first_time=first_time)
    return [human_dict, gpt_dict]


def create_conversation_about_biome_one_round(location_stats, first_time=False, random_id=None):
    biome_name = str(location_stats['biome_name'])

    random_id = random.randint(0, len(Biome_QA) - 1) if random_id is None else random_id

    human_dict, gpt_dict = create_conversation(human=Biome_QA[random_id]['Q'],
                                                gpt=Biome_QA[random_id]['A'].format(biome_dict[biome_name]),
                                                first_time=first_time)  
    return [human_dict, gpt_dict]



def extract_location_stats(filename):
    location_stats = {}
    with open(filename, 'r') as file:
        content = file.read()

    # Match Biome ID
    match = re.search(r"Biome Name: ([\w]+)", content)
    if match:
        location_stats['biome_name'] = str(match.group(1))
        # location_stats['biome_name'] = biome_dict[location_stats['biome_name']]

    # Match Rainfall
    match = re.search(r"Rainfall: \[([\d.]+)\]", content)
    if match:
        location_stats['rainfall'] = float(match.group(1))

    # Match Can See Sky
    match = re.search(r"Can See Sky: (True|False)", content)
    if match:
        location_stats['can_see_sky'] = True if match.group(1) == 'True' else False

    # Match Is Raining
    match = re.search(r"Is Raining: (True|False)", content)
    if match:
        location_stats['is_raining'] = True if match.group(1) == 'True' else False

    # Match Light Level
    match = re.search(r"Light Level: \[([\d.]+)\]", content)
    if match:
        location_stats['light_level'] = float(match.group(1))

    # Match Sky Light Level
    match = re.search(r"Sky Light Level: \[([\d.]+)\]", content)
    if match:
        location_stats['sky_light_level'] = float(match.group(1))

    # Match Sun Brightness
    match = re.search(r"Sun Brightness: \[([\d.]+)\]", content)
    if match:
        location_stats['sun_brightness'] = float(match.group(1))

    return location_stats