import numpy as np


"""method 0: Sample original affordance label"""
def rand_label_as_describe(affordance):
    res = []
    for i in range(len(affordance)):
        aspects = affordance[i].split('*')
        res.append(aspects[0])
    return np.random.choice(res, len(res), replace=True)


"""method 1: Sample random one phrase in single perspective"""
def fixed_single_aspect_rand_singe_phrase(affordance):
    res = []
    for i in range(len(affordance)):
        aspects = affordance[i].split('*')
        aspects = aspects[1:]  # reject original label
        aspect_phrases = aspects[0].split('#')  # Actions:0, Function:1, Appearance:2, Environment:3
        res.append(", ".join(np.random.choice(aspect_phrases, size=1, replace=False)))
    return res


"""method 2: Sample random number of samples in single perspective then concatenate them"""
def fixed_single_aspect_rand_phrases(affordance):
    res = []
    for i in range(len(affordance)):
        aspects = affordance[i].split('*')
        aspects = aspects[1:]  # reject original label
        aspect_phrases = aspects[0].split('#')  # Actions:0, Function:1, Appearance:2, Environment:3
        res.append(", ".join(np.random.choice(aspect_phrases, size=np.random.randint(1, len(aspect_phrases) + 1), replace=False)))
    return res

"""method 3: Sample random one phrase in random one perspective"""
def rand_single_aspect_rand_singe_phrase(affordance):
    res = []
    for i in range(len(affordance)):
        aspects = affordance[i].split('*')
        aspects = aspects[1:]  # reject original label
        phrases = np.random.choice(aspects, size=1, replace=False)[0].split('#')
        res.append(", ".join(np.random.choice(phrases, size=1, replace=False)))
    return res


"""method 4: Sample random number of phrases in one random perspective then concatenate them"""
def rand_single_aspect_rand_phrases(affordance):
    res = []
    for i in range(len(affordance)):
        aspects = affordance[i].split('*')
        aspects = aspects[1:]  # reject original label
        phrases = np.random.choice(aspects, size=1, replace=False)[0].split('#')
        res.append(", ".join(np.random.choice(phrases, size=np.random.randint(1, len(phrases) + 1), replace=False)))
    return res


"""method 5: Sample random number of phrases in random number of perspectives then concatenate them"""
def rand_aspects_rand_phrases(affordance):
    res = []
    for i in range(len(affordance)):
        aspects = affordance[i].split('*')
        chosen = np.random.choice(aspects, size=np.random.randint(1, len(aspects) + 1), replace=False)
        tmp_strs = []
        for _ in chosen:
            phrases = _.split('#')
            tmp_strs += [", ".join(np.random.choice(phrases, size=np.random.randint(1, len(phrases) + 1), replace=False))]
            if len(phrases) == 1:  # Adopting this condition will increase the frequency of original label appearances.
                break
        res.append(", ".join(tmp_strs))
    return res

"""method 6: Sample random one phrase regardless of its real affordance"""
def label_as_describe(affordance):
    res = []
    for i in range(len(affordance)):
        aspects = affordance[i].split('*')
        res.append(aspects[0])
    return res


"""method """
def compose_method(affordance, method=5):
    res = []
    if method == 0:
        res = label_as_describe(affordance)
    elif method == 1:
        res = fixed_single_aspect_rand_singe_phrase(affordance)
    elif method == 2:
        res = fixed_single_aspect_rand_phrases(affordance)
    elif method == 3:
        res = rand_single_aspect_rand_singe_phrase(affordance)
    elif method == 4:
        res = rand_single_aspect_rand_phrases(affordance)
    elif method == 5:
        res = rand_aspects_rand_phrases(affordance)
    elif method == 6:
        res = rand_label_as_describe(affordance)
    else:
        raise ValueError("method must be 0 - 6")
    return res