import evaluate as eval_lib    
import src.squad_v1_1_evaluation_script as evaluate

class Evaluate:

    def question_answering(gold_data, predictions):
        """
        gold data ~ the dataset of data["data"][i]~report, where len(report["paragraphs"]) = 1
        predictions ~ {"id1": "answer text", "id2": "answer text2", ...}
        prediction for the same question from different paragraph is chosen only one,depending on the min cls
        """
        if len(predictions) == 0:
            return {'exact_match': 0.0, 'f1': 0.0}
        return evaluate.evaluate(gold_data["data"], predictions)

    def paragraph_retrieval(gold_data, predictions):
        """
        gold_data ~ the paragraphized dataset of data["data"][i]~report, where len(report["paragraphs"]) = n
        predictions ~ {"id1": [3, 5, 2, ...], "id2": [2, 1, 10, 0, ..], "id3": [21, 14, 18, 3, ..]} of top paragraphs
        the given list is sorted list of the top confident paragraphs
        """
        if len(predictions) == 0:
            return {"p@1": 0.0, "p@2": 0.0, "p@3": 0.0}
        # prepare gold paragraphs
        correct1 = 0
        correct2 = 0
        correct2_contributed_ids = set() # to not contribute two times for the same question - first and second elemnts in case there are more paragraphs containing the answer
        correct3 = 0
        correct3_contributed_ids = set() # to not contribute three times for the same question - same as above
        all_qas = set()
        for report in gold_data["data"]:
            for par_id, paragraph in enumerate(report["paragraphs"]):
                for qa in paragraph["qas"]:
                    qa_id = qa["id"]
                    all_qas.add(qa_id)
                    if qa_id in predictions:
                        if par_id == predictions[qa_id][0]:
                            correct1 += 1
                            if not qa_id in correct2_contributed_ids:
                                correct2 += 1
                                correct2_contributed_ids.add(qa_id)
                            if not qa_id in correct3_contributed_ids:
                                correct3 += 1
                                correct3_contributed_ids.add(qa_id)
                        if len(predictions[qa_id]) >= 2 and par_id == predictions[qa_id][1]:
                            if not qa_id in correct2_contributed_ids:
                                correct2 += 1
                                correct2_contributed_ids.add(qa_id)
                            if not qa_id in correct3_contributed_ids:
                                correct3 += 1
                                correct3_contributed_ids.add(qa_id)
                        if len(predictions[qa_id]) >= 3 and par_id == predictions[qa_id][2]:
                            if not qa_id in correct3_contributed_ids:
                                correct3 += 1
                                correct3_contributed_ids.add(qa_id)
        precision_at1 = correct1/len(all_qas)
        precision_at2 = correct2/len(all_qas)
        precision_at3 = correct3/len(all_qas)
        return {"p@1": precision_at1, "p@2": precision_at2, "p@3": precision_at3}