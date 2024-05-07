
import re
import uuid
import numpy as np
import logging
from src.paragraphs.medication import paragraphize_medication
from src.paragraphs.relations import paragraphize_relations
from src.paragraphs.uniform import paragraphize_uniformly
from src.paragraphs.utils import prune_sentence, find_substring_offsets


class Paragraphizer:
    def preprocess(data, max_answer_length=20):
        """
        filter -> preprocess -> data[i]~report -> gold answers, gold question ids
        """
        preprocessed_data = {"data": []}
        for report_id, report in enumerate(data["paragraphs"]):
            context = report["context"]
            # Prepare lenghts to be able find relative position of evidence based on line number
            context_len = np.sum([len(line) for line in context])
            line_offsets = [0]
            for line in context:
                line_offsets.append(line_offsets[-1] + len(line))

            norm_context = ' '.join((' '.join(context).split()))
            norm_context = prune_sentence(norm_context)
            preprocessed_report = {"paragraphs": [{"qas":[]}]}
            preprocessed_report["paragraphs"][0]["norm_context"] = norm_context
            preprocessed_report["paragraphs"][0]["context"] = context
            for qa in report["qas"]:
                new_answers = []
                for answer in qa["answers"]:
                    if answer["answer_entity_type"] == 'complex':
                        continue
                    evidence_line = answer["evidence_start"]
                    original_relative_position = float(line_offsets[evidence_line])/context_len
                    evidence = prune_sentence(answer["evidence"])
                    # Find the evidence in paragraph
                    if evidence:
                        evidence = ' '.join(evidence.split())
                        while evidence[-1] in [',', '.', '?', '!', '-', ' ']:
                            evidence = evidence[:-1]
                        char_pos = -1
                        temp_evidence = evidence
                        final_evidence = temp_evidence
                        num = 0
                        while char_pos == -1:
                            offsets = find_substring_offsets(norm_context, temp_evidence)
                            best_offset = -1
                            pos_score = len(norm_context)
                            for offset in offsets:
                                new_rel_pos = float(offset)/len(norm_context)
                                if abs(new_rel_pos - original_relative_position) < pos_score:
                                    pos_score = abs(new_rel_pos - original_relative_position)
                                    best_offset = offset 
                            char_pos = best_offset
                            final_evidence = temp_evidence
                            temp_evidence = ' '.join(temp_evidence.split()[:-1])
                            num += 1
                        if char_pos >= 0 and final_evidence:
                            if len(final_evidence.split()) < max_answer_length:
                                new_answers.append({"answer_start": char_pos, "text": final_evidence, "answer_line": answer["evidence_start"], "original_evidence": answer["evidence"]})
                        else:
                            continue
                if len(new_answers) == 0:
                    continue

                # Create single qas element for each parahprazed question
                for q in qa["question"]:
                    new_qa = {
                        "question": q,
                        # generate a unique uuid for each question (similar to SQuAD)
                        "id" : str(uuid.uuid1().hex),
                        "answers": new_answers
                    }
                    preprocessed_report["paragraphs"][0]["qas"].append(new_qa)

            preprocessed_report["title"] = "REPORT {}".format(report_id)
            preprocessed_data["data"].append(preprocessed_report)
        return preprocessed_data

    
    def paragraphize(data, title, frequency_threshold, topics = None, target_average=500):
        """
        get data -> paragraphize data ~ split data[i]["paragraphs"] into more paragraphs (right now should be len(data[i]["paragraphs"]) = 1)
        """
        if title == "relations":
            paragraphs, topics = paragraphize_relations(data, frequency_threshold, topics)
        elif title == "medication":
            paragraphs, topics = paragraphize_medication(data, frequency_threshold, topics)
        elif title == "uniform":
            paragraphs, _ = paragraphize_uniformly(data, frequency_threshold, target_average)
        else:
            raise Exception("{} title is not supported".format(title))


        paragraphized_data = {"data": []}
        for report_id, (pars, original_report) in enumerate(zip(paragraphs, data["data"])):
            orginal_context = original_report["paragraphs"][0]["context"]
            norm_context = original_report["paragraphs"][0]["norm_context"]
            # Init new paragraphized report json
            paragraphized_report = {"paragraphs": []}
            par_offsets = [0]
            normed_pars = []
            for new_par in pars:
                new_par = ' '.join((' '.join(new_par.split('\n')).split()))
                if title != "uniform": # otherwise all paragraphs are already based on normalized and pruned context
                    new_par = prune_sentence(new_par)
                normed_pars.append(new_par)
            for par_id, new_par in enumerate(normed_pars):
                if par_id == len(pars)-1:
                    par_mid = 1
                else:
                    beg_offset = norm_context.find(normed_pars[par_id+1], (par_offsets[-1] + len(new_par)))
                    if beg_offset == -1:
                        logging.error(normed_pars)
                        logging.error("The paragraph is missing ... paragraph: '{}', offset: '{}', context: '{}'".format(normed_pars[par_id+1], par_offsets[-1] + len(new_par), norm_context))
                        assert beg_offset != -1
                    par_mid = beg_offset - (par_offsets[-1] + len(new_par)) # for extra spaces between paragraphs etc..
                par_offsets.append(par_offsets[-1] + len(new_par) + par_mid)
                paragraphized_report["paragraphs"].append({"context": new_par, "qas": []})

            # Collect evidences and answer starts
            for qa in original_report["paragraphs"][0]["qas"]:
                new_qa_bypars = {}
                qid = qa["id"]
                question = qa["question"]
                for answer in qa["answers"]:
                    paragraph_found = False
                    for i, pruned_paragraph in enumerate(paragraphized_report["paragraphs"]):
                        if par_offsets[i] <= answer["answer_start"] and par_offsets[i+1] > answer["answer_start"]:
                            if not i in new_qa_bypars:
                                new_qa_bypars[i] = []
                            new_answer = {"text": answer["text"], "answer_start": answer["answer_start"]-par_offsets[i]}
                            new_qa_bypars[i].append(new_answer)
                            paragraph_found = True
                            if not pruned_paragraph["context"][new_answer["answer_start"]:new_answer["answer_start"]+len(new_answer["text"])] == new_answer["text"]:
                                logging.error("The evidence does not match... evidence start: '{}'".format(answer["answer_start"]) +
                                            ", ith paragraph: '{}'".format(i) + 
                                            ", ith report: '{}'".format(report_id) +
                                            ", paragraph: '{}'".format(pruned_paragraph["context"]) + 
                                            ", new answer: '{}'".format(pruned_paragraph["context"][new_answer["answer_start"]:new_answer["answer_start"]+len(new_answer["text"])]) + 
                                            ", old answer: '{}'".format(new_answer["text"]) + 
                                            ", the original context: '{}'".format(orginal_context))
                                assert pruned_paragraph["context"][new_answer["answer_start"]:new_answer["answer_start"]+len(new_answer["text"])] == new_answer["text"]
                            break
                    assert paragraph_found
                for par_num in new_qa_bypars:
                    paragraphized_report["paragraphs"][par_num]["qas"].append({
                        "id": qid,
                        "question": question,
                        "answers": new_qa_bypars[par_num]
                    })


            paragraphized_report["title"] =  original_report["title"]
            paragraphized_data["data"].append(paragraphized_report)
        return paragraphized_data, topics

