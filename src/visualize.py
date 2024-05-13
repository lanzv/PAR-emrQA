import matplotlib.pyplot as plt
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
        ft_map = relations_map
        x_min, x_max = 200, 5100
        qa_y_min, qa_y_max = 85, 100
        pr_y_min, pr_y_max = 94, 100.1
    elif experiment_title == "medication":
        ft_map = medication_map
        x_min, x_max = 300, 6500
        qa_y_min, qa_y_max = 20, 80
        pr_y_min, pr_y_max = 75, 100.1
    elif experiment_title == "relations_uniform":
        ft_map = relations_uniform_map
        x_min, x_max = 200, 4200
        qa_y_min, qa_y_max = 85, 100
        pr_y_min, pr_y_max = 94, 100.1
    elif experiment_title == "medication_uniform":
        ft_map = medication_uniform_map
        x_min, x_max = 300, 6000
        qa_y_min, qa_y_max = 20, 80
        pr_y_min, pr_y_max = 75, 100.1
    else:
        raise Exception()


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

    
    plt.plot(x, [np.mean(p1[ft]) for ft in fts], 'g', label='p@1')
    plt.plot(x, [np.min(p1[ft]) for ft in fts], 'g--')
    plt.plot(x, [np.max(p1[ft]) for ft in fts], 'g--')
    plt.plot(x, [np.mean(p2[ft]) for ft in fts], 'b', label='p@2')
    plt.plot(x, [np.min(p2[ft]) for ft in fts], 'b--')
    plt.plot(x, [np.max(p2[ft]) for ft in fts], 'b--')
    plt.plot(x, [np.mean(p3[ft]) for ft in fts], 'r', label='p@3')
    plt.plot(x, [np.min(p3[ft]) for ft in fts], 'r--')
    plt.plot(x, [np.max(p3[ft]) for ft in fts], 'r--')
    plt.xlabel('Avg Paragraph Length (chr)')
    plt.ylabel('Scores')
    plt.xlim(x_min, x_max)
    plt.ylim(pr_y_min, pr_y_max)
    plt.title('{} - {}: PR'.format(experiment_title, model))
    plt.legend()
    plt.show()

    
    plt.plot(x, [np.mean(qa_f1[ft]) for ft in fts], 'r', label='f1')
    plt.plot(x, [np.min(qa_f1[ft]) for ft in fts], 'r--')
    plt.plot(x, [np.max(qa_f1[ft]) for ft in fts], 'r--')
    plt.plot(x, [np.mean(qa_em[ft]) for ft in fts], 'm', label='exact match')
    plt.plot(x, [np.min(qa_em[ft]) for ft in fts], 'm--')
    plt.plot(x, [np.max(qa_em[ft]) for ft in fts], 'm--')
    
    plt.xlabel('Avg Paragraph Length (chr)')
    plt.ylabel('Scores')
    plt.xlim(x_min, x_max)
    plt.ylim(qa_y_min, qa_y_max)
    plt.title('{} - {}: QA'.format(experiment_title, model))
    plt.legend()
    plt.show()

    plt.plot(x, [np.mean(prqa_f1[ft]) for ft in fts], 'r', label='f1')
    plt.plot(x, [np.min(prqa_f1[ft]) for ft in fts], 'r--')
    plt.plot(x, [np.max(prqa_f1[ft]) for ft in fts], 'r--')
    plt.plot(x, [np.mean(prqa_em[ft]) for ft in fts], 'm', label='exact match')
    plt.plot(x, [np.min(prqa_em[ft]) for ft in fts], 'm--')
    plt.plot(x, [np.max(prqa_em[ft]) for ft in fts], 'm--')

    plt.xlabel('Avg Paragraph Length (chr)')
    plt.ylabel('Scores')
    plt.xlim(x_min, x_max)
    plt.ylim(qa_y_min, qa_y_max)
    plt.title('{} - {}: PRQA'.format(experiment_title, model))
    plt.legend()
    plt.show()