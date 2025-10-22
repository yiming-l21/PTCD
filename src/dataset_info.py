# labels_and_templates.py
"""
Central registry for label spaces and prompt templates.
"""

# === fine-grained (with ASPECT) ===
t1_fine = {
    # text-only: [MASK] [P] [A] [P] [S] [P]
    'content': ['[CLS] [MASK]', '[SEP]'],
    'map': [0, 'p', 'a', 'p', 's', 'p', 1],
    # default multimodal
    'default': {
        # [Img] [P] [MASK] [P] [A] [P] [S] [P]
        'content': ['[CLS]', '[MASK]', '[SEP]'],
        'map': [0, 'i', 'p', 1, 'p', 'a', 'p', 's', 'p', 2],
    },
}

t2_fine = {
    'content': ['[CLS] Text : " ', ' " . Aspect: " ', ' " . Sentiment of aspect : [MASK] . [SEP]'],
    'map': [0, 's', 1, 'a', 2],
    'default': {
        'content': ['[CLS]', '[SEP] Text : " ', ' " . Aspect: " ', ' " . Sentiment of aspect : [MASK] . [SEP]'],
        'map': [0, 'i', 1, 's', 2, 'a', 3],
    },
}

template_fine = {1: t1_fine, 2: t2_fine}

# === coarse-grained (no ASPECT) ===
t1_coarse = {
    'content': ['[CLS] [MASK]', '[SEP]'],
    'map': [0, 'p', 's', 'p', 1],
    'default': {
        'content': ['[CLS]', '[MASK]', '[SEP]'],
        'map': [0, 'i', 'p', 1, 'p', 's', 'p', 2],
    },
}

t2_coarse = {
    'content': ['[CLS] Text : " ', ' " . Sentiment of text : [MASK] . [SEP]'],
    'map': [0, 's', 1],
    'default': {
        'content': ['[CLS]', '[SEP] Text : " ', ' " . Sentiment of text : [MASK] . [SEP]'],
        'map': [0, 'i', 1, 's', 2],
    },
}

template_coarse = {1: t1_coarse, 2: t2_coarse}

# === processors (dataset -> (labels, label_map, template-chooser)) ===
def twitter(template: int):
    label_list = ['negative', 'neutral', 'positive']
    label_map = {'0': 'negative', '1': 'neutral', '2': 'positive'}
    return label_list, label_map, template_fine[template]

def masad(template: int):
    label_list = ['negative', 'positive']
    label_map = {'negative': 'negative', 'positive': 'positive'}
    return label_list, label_map, template_fine[template]

def mvsa(template: int):
    label_list = ['negative', 'neutral', 'positive']
    label_map = {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'}
    return label_list, label_map, template_coarse[template]

def tumemo(template: int):
    label_list = ['angry', 'bored', 'calm', 'fear', 'happy', 'love', 'sad']
    label_map = {'Angry': 'angry', 'Bored': 'bored', 'Calm': 'calm', 'Fear': 'fear',
                 'Happy': 'happy', 'Love': 'love', 'Sad': 'sad'}
    return label_list, label_map, template_coarse[template]

processors = {
    't2015': twitter,
    't2017': twitter,
    'masad': masad,
    'mvsa-s': mvsa,
    'mvsa-m': mvsa,
    'tumemo': tumemo,
}

def get_labels_and_template(dataset_name: str, template_id: int):
    """Return (label_list, label_map, template_dict) for a dataset name."""
    name = dataset_name.lower()
    if name not in processors:
        label_list = ['negative', 'neutral', 'positive']
        label_map = {l: l for l in label_list}
        return label_list, label_map, template_coarse.get(template_id, t2_coarse)
    return processors[name](template_id)
