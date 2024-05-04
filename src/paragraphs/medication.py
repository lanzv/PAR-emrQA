import numpy as np
import logging
import re
from collections import Counter
from src.paragraphs.utils import paragraphize_by_topics, postprocess_sentence_groups

def paragraphize_medication(data, frequency_threshold, topics=None):
    sentence_groups = __split_medication_to_sentence_groups(data, new_lines = True)
    sentence_groups = postprocess_sentence_groups(data, sentence_groups)
    paragraphs, topics = paragraphize_by_topics(sentence_groups, frequency_threshold, topics)
    return paragraphs, topics


def __split_medication_to_sentence_groups(data, new_lines = True):
    """
    data ~ data.json["data"][1], where data["topics] = "medication"
    """ 
    # split by dots function
    def dot_split(block):
        dot_pattern = r'[^A-Z\.]{5}\.[\s]*\n$'
        uppercased_pattern = r'^[0-9]+[\s]*[\)\.][\s]*[A-Z]|^[A-Z]'
        # split by dots
        new_block = ''
        blocks = []
        for i, line in enumerate(block.split('\n')):
            line += '\n'
            if bool(re.search(dot_pattern, line)) and (len(block.split('\n')) > i+1 and bool(re.search(uppercased_pattern, block.split('\n')[i+1]))):
                # Split
                new_block += line
                blocks.append(new_block)
                new_block = ''
            else:
                new_block += line
        if len(new_block) != 0:
            blocks.append(new_block)
        # Remove last \n that was add via for i, line in enumerate(block.split('\n')): line += "\n"
        blocks[-1] = blocks[-1][:-1] 
        if len(blocks[-1]) == 0:
            blocks = blocks[:-1]
        return blocks
    
    def uppercased_topic_split(block):
        pattern = r'^[A-Z\s]*:'
        # split by dots
        new_block = ''
        blocks = []
        for i, line in enumerate(block.split('\n')):
            line += '\n'
            if bool(re.search(pattern, line)):
                # Split
                if len(new_block) != 0:
                    blocks.append(new_block)
                new_block = line
            else:
                new_block += line
        if len(new_block) != 0:
            blocks.append(new_block)
        # Remove last \n that was add via for i, line in enumerate(block.split('\n')): line += "\n"
        blocks[-1] = blocks[-1][:-1] 
        if len(blocks[-1]) == 0:
            blocks = blocks[:-1]
        return blocks

    # Split to sentence groups by dots
    sentence_groups = []
    for report in data["data"]:
        context_list = report["paragraphs"][0]["context"]
        # Split by dots
        sentence_groups.append(dot_split(''.join(context_list)))

    # Split by UPPERCASED TOPICS : at begginings
    sentence_groups_temp = []
    for con in sentence_groups:
        new_con = []
        for group in con:
            new_blocks = uppercased_topic_split(group)
            for block in new_blocks:
                new_con.append(block)
        sentence_groups_temp.append(new_con)
    sentence_groups = sentence_groups_temp


    # Remove '\n'
    if not new_lines:
        sentence_groups_temp = []
        for con in sentence_groups:
            new_con = []
            for group in con:
                new_group = group.replace('\n', ' ')
                if new_group[-1] == ' ':
                    new_group = new_group[:-1]
                new_con.append(new_group)
            
            sentence_groups_temp.append(new_con)
        sentence_groups = sentence_groups_temp


    counts = np.array([len(gs) for gs in sentence_groups])
    logging.info("Contexts were splited into {} sentence groups, which are {} groups on average per one report".format(np.sum(counts), np.mean(counts)))
    
    return sentence_groups