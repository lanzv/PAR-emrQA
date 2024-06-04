import matplotlib.pyplot as plt
import matplotlib
import numpy as np

relations_map = {
"0": 226.61348077935756,
"68": 668.2282608695652,
"134": 1005.4649532710281,
"143": 1332.3188854489165,
"149": 1707.6944444444443,
"154": 2010.9299065420562,
"157": 2640.116564417178, 
"190": 3052.049645390071,
"198": 5003.941860465116
}

relations_uniform_map = {
"0": 253.16746698679472,
"68": 662.1302982731554,
"134": 1062.410579345088,
"143": 1429.7525423728814,
"149": 1866.2699115044247,
"154": 2243.494680851064,
"157": 2793.225165562914, 
"190": 3012.692857142857,
"198": 4135.068627450981
}

medication_map = {
"0": 348.69066937119675,
"66": 619.4756756756757,
"77": 955.025,
"89": 1254.7773722627737,
"100": 1534.861607142857,
"119": 1868.5271739130435,
"121": 2323.0337837837837,
"142": 2989.6434782608694,
"229": 6486.962264150943
}

medication_uniform_map = {
"0": 338.3053892215569,
"66": 621.9853211009174,
"77": 939.0083102493074,
"89": 1250.8560885608856,
"100": 1554.9633027522937,
"119": 1832.335135135135,
"121": 2275.046979865772,
"142": 2999.8407079646017,
"229": 5844.517241379311
}



def plot_scores_QAPR(scores_dict, model, experiment_title='relations'):
    if experiment_title == "relations":
        title = "Relations, heading-based"
        ft_map = relations_map
        x_min, x_max = 200, 4200
        xticks = [200, 1000, 2000, 3000, 4000]
        qa_y_min, qa_y_max = 85, 100
        pr_y_min, pr_y_max = 94, 100.1
    elif experiment_title == "medication":
        title = "Medication, heading-based"
        ft_map = medication_map
        x_min, x_max = 200, 4200
        xticks = [200, 1000, 2000, 3000, 4000]
        qa_y_min, qa_y_max = 20, 80
        pr_y_min, pr_y_max = 75, 100.1
    elif experiment_title == "relations_uniform":
        title = "Relations, naive"
        ft_map = relations_uniform_map
        x_min, x_max = 200, 4200
        xticks = [200, 1000, 2000, 3000, 4000]
        qa_y_min, qa_y_max = 85, 100
        pr_y_min, pr_y_max = 94, 100.1
    elif experiment_title == "medication_uniform":
        title = "Medication, naive"
        ft_map = medication_uniform_map
        x_min, x_max = 200, 4200
        xticks = [200, 1000, 2000, 3000, 4000]
        qa_y_min, qa_y_max = 20, 80
        pr_y_min, pr_y_max = 75, 100.1
    else:
        raise Exception()
    matplotlib.rcParams.update({'font.size': 18})

    x = []
    p1, p2, p3 = {}, {}, {}
    prqa_em, prqa_f1 = {}, {}
    qa_em, qa_f1 = {}, {}
    p1, p2, p3 = {}, {}, {}
    for seed in scores_dict[model]:
        seed_obj = scores_dict[model][seed]
        for ft in seed_obj:
            if not ft in p1:
                p1[ft] = []
            if not ft in p2:
                p2[ft] = []
            if not ft in p3:
                p3[ft] = []
            if not ft in prqa_em:
                prqa_em[ft] = []
            if not ft in prqa_f1:
                prqa_f1[ft] = []
            if not ft in qa_em:
                qa_em[ft] = []
            if not ft in qa_f1:
                qa_f1[ft] = []
            p1[ft].append(100*seed_obj[ft]["PR"]["p@1"])
            p2[ft].append(100*seed_obj[ft]["PR"]["p@2"])
            p3[ft].append(100*seed_obj[ft]["PR"]["p@3"])
            prqa_em[ft].append(seed_obj[ft]["PRQA"]["exact_match"])
            prqa_f1[ft].append(seed_obj[ft]["PRQA"]["f1"])
            qa_em[ft].append(seed_obj[ft]["QA"]["exact_match"])
            qa_f1[ft].append(seed_obj[ft]["QA"]["f1"])

    x = []
    fts = []
    for ft in p1:
        x.append(ft_map[ft])
        fts.append(ft)
    assert len(x) == len(fts)
    values = [p1, p2, p3, prqa_em, prqa_f1, qa_em, qa_f1]
    for ft in fts:
        for val in values:
            assert ft in val
            assert len(val[ft]) == 3

    plt.figure(figsize=(10,6))
    plt.plot(x, [np.mean(qa_f1[ft]) for ft in fts], 'r', label='F1')
    plt.plot(x, [np.min(qa_f1[ft]) for ft in fts], 'r--')
    plt.plot(x, [np.max(qa_f1[ft]) for ft in fts], 'r--')
    plt.plot(x, [np.mean(qa_em[ft]) for ft in fts], 'm', label='Exact Match')
    plt.plot(x, [np.min(qa_em[ft]) for ft in fts], 'm--')
    plt.plot(x, [np.max(qa_em[ft]) for ft in fts], 'm--')
    
    plt.xlabel('Average Segment Length (characters)')
    plt.ylabel('Scores')
    plt.xlim(x_min, x_max)
    plt.xticks(xticks)
    plt.ylim(qa_y_min, qa_y_max)
    plt.title('{} - {} - Oracle-QA'.format(title, model))
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(x, [np.mean(p1[ft]) for ft in fts], 'g', label='P@1')
    plt.plot(x, [np.min(p1[ft]) for ft in fts], 'g--')
    plt.plot(x, [np.max(p1[ft]) for ft in fts], 'g--')
    plt.plot(x, [np.mean(p2[ft]) for ft in fts], 'b', label='P@2')
    plt.plot(x, [np.min(p2[ft]) for ft in fts], 'b--')
    plt.plot(x, [np.max(p2[ft]) for ft in fts], 'b--')
    plt.plot(x, [np.mean(p3[ft]) for ft in fts], 'r', label='P@3')
    plt.plot(x, [np.min(p3[ft]) for ft in fts], 'r--')
    plt.plot(x, [np.max(p3[ft]) for ft in fts], 'r--')
    plt.xlabel('Average Segment Length (characters)')
    plt.ylabel('Scores')
    plt.xlim(x_min, x_max)
    plt.xticks(xticks)
    plt.ylim(pr_y_min, pr_y_max)
    plt.title('{} - {} - PR'.format(title, model))
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(x, [np.mean(prqa_f1[ft]) for ft in fts], 'r', label='F1')
    plt.plot(x, [np.min(prqa_f1[ft]) for ft in fts], 'r--')
    plt.plot(x, [np.max(prqa_f1[ft]) for ft in fts], 'r--')
    plt.plot(x, [np.mean(prqa_em[ft]) for ft in fts], 'm', label='Exact Match')
    plt.plot(x, [np.min(prqa_em[ft]) for ft in fts], 'm--')
    plt.plot(x, [np.max(prqa_em[ft]) for ft in fts], 'm--')

    plt.xlabel('Average Segment Length (characters)')
    plt.ylabel('Scores')
    plt.xlim(x_min, x_max)
    plt.xticks(xticks)
    plt.ylim(qa_y_min, qa_y_max)
    plt.title('{} - {} - PR-QA'.format(title, model))
    plt.legend()
    plt.show()




def visualize_dataset_stats(fts= [0, 68, 134, 143, 149, 154, 157, 190, 198],title = "relations"):
    import logging
    import json
    from src.paragraphizer import Paragraphizer
    import matplotlib.pyplot as plt
    import numpy as np

    matplotlib.rcParams.update({'font.size': 12})
    if title == "relations":
        dataset_title = "Relations"
    elif title == "medication": 
        dataset_title = "Medication"
    else:
        raise Exception()

    with open("./data/{}-train.json".format(title), 'r') as f:
        train = json.load(f)
    with open("./data/{}-dev.json".format(title), 'r') as f:
        dev = json.load(f)
    with open("./data/{}-test.json".format(title), 'r') as f:
        test = json.load(f)



    x = fts
    # QA-pairs
    qa_pairs_train = []
    qa_pairs_dev = []
    qa_pairs_test = []
    # histogram frequencies
    # todo generated from train_topics 
    _, train_topics = Paragraphizer.paragraphize(data = train, title=title, frequency_threshold = 0)
    # average paragraph lengths
    par_lengths_train = []
    par_lengths_dev = []
    par_lengths_test = []
    para_lengths_train = []
    para_lengths_dev = []
    para_lengths_test = []

    def get_lens_qas(dataset):
        qa_pairs = 0
        par_lengths = []
        para_lengths = []
        for report in dataset["data"]:
            for paragraph in report["paragraphs"]:
                qa_pairs += len(paragraph["qas"])
                par_lengths.append(len(paragraph["context"]))
                atleast_one_answer = False
                for qa in paragraph["qas"]:
                    if len(qa["answers"]) > 0:
                        atleast_one_answer = True
                if atleast_one_answer:
                    para_lengths.append(len(paragraph["context"]))
        return qa_pairs, par_lengths, para_lengths


    def show_heading_histogram(topics):
        labels, values = zip(*topics.most_common())
        indexes = np.arange(len(labels))
        width = 1
        plt.bar(indexes, values, width)
        #plt.tick_params(
        #axis='x',
        #which='both',
        #bottom=False,
        #top=False,
        #labelbottom=False)
        plt.title('{} - Histogram of Heading Occurrence Frequency (Training Data)'.format(dataset_title))
        plt.ylabel('Heading Occurrence Frequency')
        plt.xlabel('List of Headings')
        plt.show()


    def show_qapairs(x, qa_pairs, dataset_type):
        plt.figure(figsize=(10,2))
        plt.plot(x, qa_pairs, 'mediumpurple')
        plt.xlabel('Heading Occurrence Frequency')
        plt.ylabel('# P-Q-A triples')
        plt.title('{} - Number of Paragraph-Question-Answer Triples - {}'.format(dataset_title, dataset_type))
        plt.show()

    def show_par_avg_lens(x, par_lengths_train, par_lengths_dev, par_lengths_test, para_lengths_train, para_lengths_dev, para_lengths_test):
        plt.figure(figsize=(10, 6))
        plt.plot(x, [np.mean(lens) for lens in par_lengths_train], 'darkkhaki', label='Train')
        plt.plot(x, [np.min(lens) for lens in par_lengths_train], '--', color='darkkhaki')
        plt.plot(x, [np.max(lens) for lens in par_lengths_train], '--', color='darkkhaki')
        plt.plot(x, [np.mean(lens) for lens in par_lengths_dev], 'steelblue', label='Dev')
        plt.plot(x, [np.min(lens) for lens in par_lengths_dev], '--', color='steelblue')
        plt.plot(x, [np.max(lens) for lens in par_lengths_dev], '--', color='steelblue')
        plt.plot(x, [np.mean(lens) for lens in par_lengths_test], 'palegreen', label='Test')
        plt.plot(x, [np.min(lens) for lens in par_lengths_test], '--', color='palegreen')
        plt.plot(x, [np.max(lens) for lens in par_lengths_test], '--', color='palegreen')
        plt.xlabel('Heading Occurrence Frequency')
        plt.ylabel('Average Paragraph Length (characters)')
        plt.title('{} - Paragraph Lengths (Average, Min, Max)'.format(dataset_title))
        plt.legend()
        plt.show()

    for ft in fts:
        print(ft, end='\r')
        train_pars, train_topics = Paragraphizer.paragraphize(data = train, title=title, frequency_threshold = ft)
        dev_pars, _ = Paragraphizer.paragraphize(data = dev, title=title, frequency_threshold = ft, topics=train_topics)
        test_pars, _ = Paragraphizer.paragraphize(data = test, title=title, frequency_threshold = ft, topics=train_topics)

        qa_pairs, par_lengths, para_lengths = get_lens_qas(train_pars)
        qa_pairs_train.append(qa_pairs)
        par_lengths_train.append(par_lengths)
        para_lengths_train.append(para_lengths)
        
        qa_pairs, par_lengths, para_lengths = get_lens_qas(dev_pars)
        qa_pairs_dev.append(qa_pairs)
        par_lengths_dev.append(par_lengths)
        para_lengths_dev.append(para_lengths)
        
        qa_pairs, par_lengths, para_lengths = get_lens_qas(test_pars)
        qa_pairs_test.append(qa_pairs)
        par_lengths_test.append(par_lengths)
        para_lengths_test.append(para_lengths)


    # 1. graph
    #   - 1.1 graph: QA-pairs for training
    #   - 1.2 graph: QA-pairs for dev
    #   - 1.3 graph: QA-pairs for test
    #show_qapairs(x, qa_pairs_train, "Train")
    #show_qapairs(x, qa_pairs_dev, "Dev")
    #show_qapairs(x, qa_pairs_test, "Test")
    # 2. graph
    #   - heading frequency histogram for train
    show_heading_histogram(train_topics)
    # 3. graph
    #   - average + max + min of train/dev/test paragraphs given heading frequency
    #   - average + max + min of train/dev/test paragraphs with answer given heading frequency
    #show_par_avg_lens(x, par_lengths_train, par_lengths_dev, par_lengths_test, para_lengths_train, para_lengths_dev, para_lengths_test)
        