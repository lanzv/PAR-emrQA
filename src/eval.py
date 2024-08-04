import evaluate as eval_lib    
import src.squad_v1_1_evaluation_script as evaluate

class Evaluate:
    """
    Static evaluation class for the PR, Oracle-QA and PRQA tasks
    """

    def question_answering(gold_data, predictions):
        """
        Call SQuAD official evaluation script to get Exact Match (EM) and F1 scores

        Parameters
        ----------
        gold_data: dict
            the preprocessed SQuAD-like emrQA subset of data["data"][i]~report, where len(report["paragraphs"]) = 1 (not paragraphized, the original one)
        predictions: dict
            predictions for the gold_data subset in the format of {"id1": "answer text", "id2": "answer text2", ...}
            each ID should consist only one predicted answer (the most confident one)
        Returns
        -------
        scores: dict
            em and f1 scores in the format of {'exact_match': exact_match, 'f1': f1}
        """
        if len(predictions) == 0:
            return {'exact_match': 0.0, 'f1': 0.0}
        return evaluate.evaluate(gold_data["data"], predictions)

    def paragraph_retrieval(gold_data, predictions):
        """
        Compute P@1, P@2 and P@3 precision scores of the paragraph retrieval task
        The script relies on the id uniqness in the gold_data, if not it could easily
        happen that P@1 > P@3 (because the script will not be working correctly)

        Parameters
        ----------
        gold_data: dict
            the paragraphized preprocessed SQuAD-like emrQA subset of data["data"][i]~report, where len(report["paragraphs"]) = n
        predictions: dict
            paragraph retrieval predictions in the format of {"id1": [3, 5, 2, ...], "id2": [2, 1, 10, 0, ..], "id3": [21, 14, 18, 3, ..]} 
            where the given list is sorted list of paragraph ids based on the paragraph's confidence given the question id
        Returns
        -------
        scores: dict
            p@1, p@2, and p@3 scores in the format of {"p@1": precision_at1, "p@2": precision_at2, "p@3": precision_at3}
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