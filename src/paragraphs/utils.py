import re
import numpy as np
import logging
from collections import Counter



def postprocess_sentence_groups(data, sentence_groups):
    """
    Postprocess sentence group texts - merge such groups that contain 
    different parts of the same answer text
    Check and assert all invalid answer text (that are already invalid in the original emrQA
    - for instance there are two lines in as answer evidence which should not be based on
    the definition)

    Parameters
    ----------
    data: dict
        filtered SQuAD-like formated dataset of the emrQA subset 
        only the context is kept the same with "answer_line", "original_evidence" corresponding to this context as part of the answer object
        the normalized context (the SQuAD-like one) is stored as the "norm_context" next to "context"
    sentence_groups: list of lists of strings
        for each reprot, there is a list of passages where each ends definitaly (99%) ends with the completed 
        sentence (no incompleted sentences included) forming the whole report 
    Returns
    -------
    new_sentence_groups: list of lists of strings
        sentence groups that don't split the answer texts
    """
    new_sentence_groups = []
    for report, rep_sgs in zip(data["data"], sentence_groups):
        new_sgs = rep_sgs
        sgs_to_modify = new_sgs
        the_only_paragraph = report["paragraphs"][0]
        sg_offsets = [0]
        for sg in new_sgs:
            sg_offsets.append(sg_offsets[-1] + len(sg.split("\n")) - 1) # -1 for the one extra "\n" at the end of the last line
        
        for qa in the_only_paragraph["qas"]:
            for ans in qa["answers"]:
                for i, offset in enumerate(sg_offsets[:-1]):
                    if ans["answer_line"]-1 >= offset and ans["answer_line"]-1 < sg_offsets[i+1]:
                        if len(ans["original_evidence"].split("\n")) == 3 and (
                                    ans["original_evidence"].split("\n")[0] == ans["original_evidence"].split("\n")[1] and ans["original_evidence"].split("\n")[2] == ''
                                ):
                            original_evidence = ans["original_evidence"].split("\n")[0] + "\n"
                            if ans["answer_line"] - 1 == sg_offsets[i+1] - 1 and len(new_sgs) > i + 1:
                                assert not original_evidence in new_sgs[i+1]
                        elif len(ans["original_evidence"].split("\n")) == 4 and (
                                    ans["original_evidence"].split("\n")[1] == ans["original_evidence"].split("\n")[2] and ans["original_evidence"].split("\n")[3] == ''
                                ):
                            original_evidence = ans["original_evidence"].split("\n")[0] + "\n" + ans["original_evidence"].split("\n")[1] + "\n"
                            if ans["answer_line"] - 1 >= sg_offsets[i+1] - 2 and len(new_sgs) > i + 1:
                                assert not (ans["original_evidence"].split("\n")[1])+"\n" in new_sgs[i+1]
                        elif len(ans["original_evidence"].split("\n")) == 4 and (
                                    ans["original_evidence"].split("\n")[0] == ans["original_evidence"].split("\n")[1] and ans["original_evidence"].split("\n")[3] == ''
                                ):
                            original_evidence = ans["original_evidence"].split("\n")[0] + "\n" + ans["original_evidence"].split("\n")[2] + "\n"
                            if ans["answer_line"] - 1 >= sg_offsets[i+1] - 2 and len(new_sgs) > i + 1:
                                assert not (ans["original_evidence"].split("\n")[2])+"\n" in new_sgs[i+1]
                        else:
                            original_evidence = ans["original_evidence"]
                    
                        if ans["answer_line"] - 1 == sg_offsets[i+1] - 1 and not original_evidence in new_sgs[i] and len(new_sgs) > i + 1:
                            logging.warn("The answer {} (originally '{}') lays between two paragraphs, '{}'".format(ans["text"], ans["original_evidence"], new_sgs[i]))
                            # merge ith and (i+1)th paragraphs
                            sgs_to_modify = new_sgs.copy()
                            sgs_to_modify[i] += sgs_to_modify[i+1]
                            del sgs_to_modify[i+1]
                        if not (original_evidence in new_sgs[i] or ans["answer_line"] - 1 == sg_offsets[i+1] - 1):
                            logging.warn("The evidenc '{}' is in wrong format and it does not occure in the paragraph '{}'".format(original_evidence, new_sgs[i]))
                        break
                new_sgs = sgs_to_modify
                sg_offsets = [0]
                for sg in new_sgs:
                    sg_offsets.append(sg_offsets[-1] + len(sg.split("\n")) - 1) # -1 for the one extra "\n" at the end of the last line
        new_sentence_groups.append(new_sgs)
    return new_sentence_groups                
                


def paragraphize_by_topics(sentence_groups, frequency_threshold = 50, topics = None, topic_regex = r'(^([0-9]+[\s]*[\.\)][\s]*)?[A-Z][a-zA-Z\s\(\)]*:)'):
    """
    Consider all sentence groups, find all possible topics at the beginning 
    of each sentence group using the topic_regex, check with such topic is 
    more frequentu than frequency_threshold in the topics (or in these 
    sentence_groups topics if topics parameter is None) and merge sentence groups
    together till there is such group containing heading which fulfill the frequency_threshold condition

    Parameters
    ----------
    sentence_groups: list of lists of strings
        for each reprot, there is a list of passages where each ends definitaly (99%) ends with the completed 
        sentence (no incompleted sentences included) forming the whole report 
    frequency_threshold: int
        number of occurrences of headings that serve as paragraph separators
    topics: collections.Counter or None
        Counter of topics and their occurrences in training data
        if not set, paragraphizer will find its own Counter, otherwise it uses given data
    topic_regex: str
        regex to indentify headings at the beginnings, for instance 'Subsection Title (Part 1):'
    Returns
    -------
    paragraphs: list of lists of strings
        list of segmented paragraphs represented as a list of strings (paragrpahs) for each report
    topics: collections.Counter
        currently used topics Counter - either the passed one, or if was none passed, the newly created one
    """
    # Get frequencies of topics over all reports
    if topics == None:
        topics = []
        for context in sentence_groups:
            for sen_group in context:
                candidates = re.findall(topic_regex, sen_group)
                if len(candidates) > 0 and len(candidates[0]) > 0:
                    topics.append(candidates[0][0])
        topics = Counter(topics)

    # Group sentence groups while there is no frequent topic
    paragraphs = []
    for context in sentence_groups:
        par_con = []
        new_paragraph = ''
        for sen_group in context:
            candidates = re.findall(topic_regex, sen_group)
            if len(candidates) > 0 and len(candidates[0]) > 0 and topics[candidates[0][0]] >= frequency_threshold:
                if len(new_paragraph) > 0:
                    par_con.append(new_paragraph)
                new_paragraph = sen_group
            else:
                new_paragraph += sen_group
        if len(new_paragraph) > 0:
            par_con.append(new_paragraph)
        paragraphs.append(par_con)

    # Return topics that are considered for paragraph separations
    filtered_topics = Counter({key: value for key, value in topics.items() if value >= frequency_threshold})
    
    # measure the average length of paragraphs
    paragraphs_lengths = []
    for report in paragraphs:
        for paragraph in report:
            paragraphs_lengths.append(len(paragraph))


    counts = np.array([len(ps) for ps in paragraphs])
    logging.info("Contexts were splited into {} paragraphs, which are {} paragraphs on average per one report. There are {} unique topics with frequency threshold (greater or equal) {}. The overall paragraph average length (characters) is {}".format(np.sum(counts), np.mean(counts), len(filtered_topics), frequency_threshold, np.mean(paragraphs_lengths)))
        
    return paragraphs, topics


def prune_sentence(sent):
    """
    Authors: https://github.com/xiangyue9607/CliniRC
    
    Normalize the given sentence (replace special characters by its alternatives, etc..)

    Parameters
    ----------
    sent: str
        sentence to be pruned
    Returns
    -------
    sent: str
        pruned sentence
    """
    sent = ' '.join(sent.split())
    # replace html special tokens
    sent = sent.replace("_", " ")
    sent = sent.replace("&lt;", "<")
    sent = sent.replace("&gt;", ">")
    sent = sent.replace("&amp;", "&")
    sent = sent.replace("&quot;", "\"")
    sent = sent.replace("&nbsp;", " ")
    sent = sent.replace("&apos;", "\'")
    sent = sent.replace("_", "")
    sent = sent.replace("\"", "")

    # remove whitespaces before punctuations
    sent = re.sub(r'\s([?.!\',)"](?:\s|$))', r'\1', sent)
    # remove multiple punctuations
    sent = re.sub(r'[?.!,]+(?=[?.!,])', '', sent)
    sent = ' '.join(sent.split())
    return sent


def find_substring_offsets(string, substring):
    """
    Find all occurrences of substring in the given string and return their offsets

    Parameters
    ----------
    string: str
        string (possibly) containing the substring
    substring: str
        substring (possibly) part of the string (could be many times)
    Returns
    -------
    offsets: list of ints
        list of offsets of all substring occurrences in the string
    """
    offsets = []
    start = 0
    while True:
        start = string.find(substring, start)
        if start == -1:
            break
        offsets.append(start)
        start += 1
    return offsets