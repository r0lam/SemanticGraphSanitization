import transformers
from typing import List, Tuple, Union, Optional
import json
import os
from dataclasses import dataclass, field, asdict
import flair
import torch
from flair.data import Sentence
from flair.models import SequenceTagger

ner = 'flair'
ner_model = "flair/ner-english-ontonotes-large"
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.




@dataclass
class PII:
    text: str
    entity_class: str
    start: Optional[int] = None
    end: Optional[int] = None
    score: Optional[float] = None

    def lower(self):
        return self.text.lower()

    def match(self, other):
        return self.text.lower() == other.lower()


class PIIEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, PII):
            return asdict(o)
        elif isinstance(o, ListPII):
            return {'data': [asdict(pii) if isinstance(pii, PII) else pii for pii in o.data]}
        elif isinstance(o, DatasetPII):
            try:
                return {
                    'data': {k: [asdict(pii) if isinstance(pii, PII) else pii for pii in v] for k, v in o.data.items()}}
            except TypeError as e:
                for k, v in o.data.items():
                    for item in v:
                        if not isinstance(item, PII) and not isinstance(item, dict):
                            print(f"Unexpected type in DatasetPII data: {type(item)}")
                raise e
        return super().default(o)


class PIIDecoder(json.JSONDecoder):
    def decode(self, s):
        decoded = super().decode(s)

        # If the decoded data is a list, it might correspond to a ListPII object
        if isinstance(decoded, list):
            return ListPII([PII(**item) if isinstance(item, dict) else item for item in decoded])

        # If the decoded data is a dictionary, it might correspond to a PII or DatasetPII object
        elif isinstance(decoded, dict):
            if 'data' in decoded:
                if isinstance(decoded['data'], list):
                    return ListPII([PII(**item) if isinstance(item, dict) else item for item in decoded['data']])
                elif isinstance(decoded['data'], dict):
                    return DatasetPII(
                        {k: [PII(**item) if isinstance(item, dict) else item for item in v] for k, v in
                         decoded['data'].items()}
                    )
            else:
                return PII(**decoded)

        # In case we can't match the type, just return the decoded data
        return decoded


@dataclass
class ListPII:
    data: List[PII] = field(default_factory=lambda: [], metadata={"help": "list of PII"})

    def get_entity_classes(self) -> List[str]:
        return list(set([pii.entity_class for pii in self.data]))

    def unique(self):
        mentions = []
        result = []
        for d_i in self.data:
            if d_i.text not in mentions:
                mentions.append(d_i.text)
                result.append(d_i)
        return ListPII(data=result)

    def mentions(self) -> List[str]:
        return [pii.text for pii in self.data]

    def get_by_entity_class(self, entity_class: str) -> 'ListPII':
        return ListPII([pii for pii in self.data if pii.entity_class == entity_class])

    def group_by_class(self) -> dict[str, 'ListPII']:
        return {
            entity_class: ListPII([pii for pii in self.data if pii.entity_class == entity_class])
            for entity_class in self.get_entity_classes()
        }

    def dumps(self) -> str:
        return json.dumps(self, cls=PIIEncoder)

    def sort(self, reverse=False):
        self.data.sort(key=lambda x: x.start, reverse=reverse)
        return self

    def __iter__(self):
        return self.data.__iter__()

    def __len__(self):
        return len(self.data)


@dataclass
class DatasetPII:
    data: dict[int, List[PII]] = field(default_factory=lambda: {}, metadata={"help": "batch_idx->PII"})

    @staticmethod
    def load(path: str):
        if os.path.exists(path):
            with open(path, 'r') as f:
                print(f"> Loading PII from {path} ...")
                d = json.load(f, cls=PIIDecoder)
                d.data = {int(k): v for k, v in d.data.items()}
                return d
        return DatasetPII()

    def save(self, path: str) -> str:
        data = json.dumps(self, cls=PIIEncoder)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(data)
        return data

    def limit(self, n: int):
        if self.last_batch_idx() > n:
            self.data = {k: v for k, v in self.data.items() if k <= n}
        return self

    def flatten(self, entity_classes: List[str] = None) -> ListPII:
        if entity_classes is not None:
            return ListPII(data=[item for sublist in self.data.values() for item in sublist if (
                item['entity_class'] in entity_classes if isinstance(item,
                                                                     dict) else item.entity_class in entity_classes)])
        return ListPII(data=[item for sublist in self.data.values() for item in sublist])

    def get_unique_pii(self, entity_classes: List[str] = None):
        """ gets all unique PII mentions of the entity classes (all if none is specified) """
        if entity_classes is not None:
            return [x for x in list(set(list(self.flatten()))) if x.entity_class in entity_classes]
        return list(set(list(self.flatten())))

    def get_pii_count(self, pii: PII):
        """ counts the number of times a PII occurs """
        return len([x for x in self.flatten() if pii.match(x)])

    def last_batch_idx(self) -> int:
        """ Gets the highest batch idx. """
        if len(self.data) == 0:
            return 0
        return max([int(x) for x in list(self.data.keys())])

    def add_pii(self, idx: int, piis: List[PII]):
        """ Adds a list of PII to the idx. """
        self.data[idx] = self.data.setdefault(idx, []) + [x for x in piis]

    def __len__(self):
        return len(self.data)


class FlairTagger():
    ENTITY_CLASSES = {
        "CARDINAL": "Refers to a numerical quantity or value, such as 'one', 'two', or 'three'.",
        "DATE": "Refers to a date, typically in the format of year-month-day or month-day-year.",
        "FAC": "Refers to a specific building or facility, such as a school or hospital.",
        "GPE": "Refers to a geopolitical entity, such as a city, state, or country.",
        "LANGUAGE": "Refers to a natural language, such as English or Spanish.",
        "LAW": "Refers to a legal document, such as a law or treaty.",
        "LOC": "Refers to a general location, such as a mountain range or body of water.",
        "MONEY": "Refers to a monetary value, such as a dollar amount or currency symbol.",
        "NORP": "Refers to a national or religious group, such as 'the French' or 'the Muslim community'.",
        "ORDINAL": "Refers to a numerical ranking or position, such as 'first', 'second', or 'third'.",
        "ORG": "Refers to an organization, such as a company or institution.",
        "PERCENT": "Refers to a percentage value, such as '50%' or '75%'.",
        "PERSON": "Refers to a specific individual or group of people, such as a celebrity or family.",
        "PRODUCT": "Refers to a specific product or brand, such as a car or electronics.",
        "QUANTITY": "Refers to a quantity, such as '12 ounces' or '3 meters'.",
        "TIME": "Refers to a specific time of day or duration, such as '3:00 PM' or 'three hours'.",
        "WORK_OF_ART": "Refers to a creative work, such as a book, painting, or movie.",
        "EVENT": "Refers to a specific event or occurrence, such as a concert or sports game."
    }

    def _load(self):
        """ Loads the flair tagger. """
        flair.device = torch.device('cuda')
        return SequenceTagger.load(ner_model).to('cuda')

    def get_entity_classes(self) -> List[str]:
        """ get taggable entities """
        return list(self.ENTITY_CLASSES.keys())

    def analyze(self, text: Union[List[str], str]) -> ListPII:
        """ Analyze a string or list of strings for PII. """

        if isinstance(text, list):
            sentences = [Sentence(x) for x in text]
            verbose = True
        else:
            sentences = [Sentence(text)]
            verbose = False

        self._load().predict(sentences,
                                 verbose=verbose,
                                 mini_batch_size=32)

        result_list: List[PII] = []

        for sentence in sentences:
            for entity in sentence.get_spans('ner'):
                for entity_class in self.get_entity_classes():
                    if any([x.to_dict()['value'] == entity_class for x in entity.get_labels()]):
                        result_list += [PII(entity_class=entity_class, start=entity.start_position,
                                            text=entity.text, end=entity.end_position,
                                            score=entity.score)]

        return ListPII(data=result_list)

    def pseudonymize(self, text: str) -> Tuple[str, ListPII]:
        """ Pseudonymizes a string if the ner_args.anonymize flag is True. """
        piis: ListPII = self.analyze(text)  # these PII contain a start and an end.

        # # Do we need to anonymize?
        # if not self.ner_args.anonymize:
        #     return text, piis

        # # 1. sort pii by start token starting with the last pii
        # piis.sort(reverse=True)
        #
        # # 2. remove all pii
        # for pii in piis:
        #     text = text[:pii.start] + self.ner_args.anon_token + text[pii.end:]

        return text, piis

@dataclass
class GeneratedText:
    text: str  # the generated text
    #score: torch.Tensor  # the score for the text

    def __str__(self):
        return self.text

@dataclass
class GeneratedTextList:
    data: List[GeneratedText]

    def __getitem__(self, item):
        return self.data[item]

    def __str__(self):
        return "\n".join([str(x) for x in self.data])

import pickle
from tqdm import tqdm
def generate(dataset_name,dataset_path):
    corpus_parts: List[GeneratedText] = []
    with open(f'{dataset_path}\\{dataset_name}.json', 'r') as f:
        data = json.load(f)
    if dataset_name == 'QNLI':
        corpus_parts.extend(
            GeneratedText(f"{data_i['text1']} {data_i['text2']}") for data_i in tqdm(data)
        )
    if dataset_name == 'SAMsum':
        corpus_parts.extend(
            GeneratedText(f"{data_i['dialogue']}") for data_i in tqdm(data)
        )
    if dataset_name == 'reClor':
        corpus_parts.extend(
            GeneratedText(f"{data_i['context']} {data_i['question']} {data_i['answers'][0]} {data_i['answers'][1]} {data_i['answers'][2]} {data_i['answers'][3]}") for data_i in tqdm(data)
        )

    return GeneratedTextList(data=corpus_parts)


pii_type = ["CARDINAL", "DATE", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", "MONEY", "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON", "PRODUCT", "QUANTITY", "TIME", "WORK_OF_ART", "EVENT"]

class NaiveExtractionAttack:

    def __init__(self):
        pass

    def _get_tagger(self):
        self._tagger = FlairTagger()
        return self._tagger

    def attack(self, generated_text):
        pii_entities = {}
        # Generating text using the language model.

        # Analyzing the generated text with the tagger to extract entities.
        tagger = self._get_tagger()
        entities = tagger.analyze([str(x) for x in generated_text])


        def get_pii_set(pii_name):
            pii = entities.get_by_entity_class(pii_name)
            pii_mentions = [p.text for p in pii]
            return list(set(pii_mentions))

        for i in pii_type:
            if get_pii_set(i) == list():
                continue
            else:
                pii_entities[i] = get_pii_set(i)

        return pii_entities
        # # Filter out the entities that are classified as the target entity class.
        # pii = entities.get_by_entity_class('PERSON')
        #
        # # Extracting the text of the entities.
        # pii_mentions = [p.text for p in pii]
        #
        # # Counting the occurrence of each entity mention.
        # result = {p: pii_mentions.count(p) for p in set(pii_mentions)}
        #
        # # Sorting the result dictionary based on the count of each entity mentions in descending order and returning it.
        # return {k: v for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True)}
attack: NaiveExtractionAttack = NaiveExtractionAttack()

dataset_name = 'reClor'
dataset_path = f""
data = generate(dataset_name,dataset_path)
results = attack.attack(data)
# if not os.path.exists(f'./extract_out/{dataset_name}/'):
#     os.mkdir(f'./extract_out/{dataset_name}/')
with open(f'./atk_{dataset_name}.json', 'w') as f:
    json.dump(results, f)
