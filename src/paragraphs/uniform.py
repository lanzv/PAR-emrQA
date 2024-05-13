import numpy as np
import logging
import re
from collections import Counter
from src.paragraphs.utils import paragraphize_by_topics

def paragraphize_uniformly(data, frequency_threshold, target_average, topics = None):
    paragraphs, _ = __split_to_paragraphs_uniformly(data, target_average)
    return paragraphs, None



def __split_to_paragraphs_uniformly(data, target_average):
    """
    """
    paragraphs = []
    counts = []
    paragraphs_lengths = []
    for report in data["data"]:
        report_length = len(report["paragraphs"][0]["norm_context"])
        num_of_paragraphs = max(1, round(float(report_length)/target_average))
        real_average = round(float(report_length)/num_of_paragraphs)

        # init paragraph offsets
        paragraph_offsets = [0]
        for _ in range(num_of_paragraphs-1):
            paragraph_offsets.append(paragraph_offsets[-1] + real_average)
        paragraph_offsets.append(report_length)

        # make sure that there are no splited answers between two paragraphs
        for _ in range(2):
            for qa in report["paragraphs"][0]["qas"]:
                for answer in qa["answers"]:
                    new_paragraph_offsets = paragraph_offsets.copy()
                    for i, offset in enumerate(paragraph_offsets):
                        if answer["answer_start"] >= offset and answer["answer_start"] < paragraph_offsets[i+1]:
                            if not answer["answer_start"] + len(answer["text"]) <= paragraph_offsets[i+1]:
                                new_paragraph_offsets[i+1] = answer["answer_start"] + len(answer["text"])
                    paragraph_offsets = new_paragraph_offsets

        # create paragraphs regarding offsets
        report_paragraphs = []
        for i, j in zip(paragraph_offsets[:-1], paragraph_offsets[1:]):
            report_paragraphs.append(report["paragraphs"][0]["norm_context"][i:j])
            paragraphs_lengths.append(len(report_paragraphs[-1]))
        paragraphs.append(report_paragraphs)
        counts.append(len(report_paragraphs))

    logging.info("Contexts were splited into {} paragraphs, which are {} paragraphs on average per one report. The overall paragraph average length (characters) is {}".format(np.sum(counts), np.mean(counts), np.mean(paragraphs_lengths)))        
    return paragraphs, None
    