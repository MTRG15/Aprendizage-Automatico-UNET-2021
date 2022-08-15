# -*- coding: utf-8 -*-

'''
Created on 17 mar. 2022

@author: jose-lopez
@modified-by: marco-ramirez
'''
from pathlib import Path
import json
import math
import random
import sys

# from spacy.matcher import Matcher
from spacy.matcher import Matcher
from spacy.tokens import Span, DocBin
import spacy


def load_jsonl(path):
    data = []

    with open(path, 'r', encoding='utf-8') as reader:
        for line in reader:
            if not line == "\n":
                data.append(json.loads(line))

    return data


def setting_patterns(patterns, matcher):
    # Loading protein names
    reader = open("C:\\Users\\marco\\Bruno\\Dev\\Python3\\Spacy\\data\\bio_objects\\proteins", "r")
    for line in reader.readlines():
        # Truncating the '\n' at the end of each line
        line = line[0:len(line) - 1]
        # Splitting the line by names
        line_names = line.split(";")
        # Extracting the name of the pattern for each item name set
        pattern_name = line_names[0]
        # print("processing protein ", pattern_name)

        # iterate over each name on the line
        for name in line_names:
            if "-" in name:
                sub_names = name.strip("-")
                pattern = [{"LOWER": sub_names[0].lower()}, {"TEXT": "-"}, {"LOWER": sub_names.lower()}, {"OP": "?"},
                           {"OP": "?"}, {"OP": "?"}, {"POS": "VERB"}, {"OP": "?"}, {"OP": "?"}, {"OP": "?"},
                           {"POS": "PROPN"}]
            elif "/" in line_names:
                sub_names = name.strip("/")
                pattern = [{"LOWER": sub_names[0].lower()}, {"TEXT": "/"}, {"LOWER": sub_names.lower()}, {"OP": "?"},
                           {"OP": "?"}, {"OP": "?"}, {"POS": "VERB"}, {"OP": "?"}, {"OP": "?"}, {"OP": "?"},
                           {"POS": "PROPN"}]
            else:
                pattern = [{"LOWER": name.lower()}, {"OP": "?"}, {"OP": "?"}, {"OP": "?"}, {"POS": "VERB"}, {"OP": "?"},
                           {"OP": "?"}, {"OP": "?"}, {"POS": "PROPN"}]
            # print("Pattern: ", pattern)
            # When all names are processed, add the pattern to the matcher
            matcher.add(pattern_name, [pattern])
    # Close the file
    reader.close()

    # Extracting the list of valid verbs from relations-functions.txt
    labels = open("C:\\Users\\marco\Bruno\\Dev\\Python3\\Spacy\\data\\relations\\relations-functions.txt").readlines()
    verbs = list()

    for label in labels:
        # removing titles
        if "/" in label:
            # print("removing ", label)
            del labels[labels.index(label) - 1]
        else:
            verbs.append(label.strip())

    # Compare the verb in the sentence with the list of verbs on relations-functions.txt
    from spacy.symbols import nsubj, VERB, dobj
    doc = nlp(
        open("C:\\Users\\marco\\Bruno\\Dev\\Python3\\Spacy\\data\corpus_covid\\corpus_1.txt", encoding="UTF-8").read())
    matched_sentences = list()

    for match_id, start, end in matcher(doc):
        span = doc[start:end]
        matched_sentences.append(span)
        # print(span)

    # Creating knowledge base
    for span in matched_sentences:
        for np_nsubj in span.noun_chunks:
            # Check if the verb found is on the list
            if np_nsubj.root.head.lemma_ in verbs:
                if np_nsubj.root.dep == nsubj and np_nsubj.root.head.pos == VERB:
                    for np_dobj in span.noun_chunks:
                        if np_dobj.root.dep == dobj and np_nsubj.root.head.pos == np_dobj.root.head.pos:
                            print(
                                f'event("{np_nsubj.root.text}",{np_nsubj.root.head.lemma_},"{np_dobj.root.text}")')
        # Showing examples of the event (CAUTION: FINDS WAY TOO MANY, BLOATS UP MEMMORY)
        # counter = 0
        # for np in span.noun_chunks:
        # print(f'"{np.text}", {np.root.text}, {np.root.dep_}, "{spacy.explain(np.root.dep_)}", {np.root.head.text}, {np.root.head.pos_}')
        # counter += 1
        # if counter == 3:
        # break


def token_from_span_in(spans, current_span):
    already_present = False

    current_span_tokens = [t.i for t in current_span]

    for span in spans:

        span_tokens = [t.i for t in span]

        for token_index in span_tokens:
            if token_index in current_span_tokens:
                already_present = True

        if already_present is True:
            break

    return already_present


def tagging_file_sentences(sentences, matcher, nlp):
    no_entities_docs = []
    entities_docs = []

    for doc in nlp.pipe(sentences):

        matches = matcher(doc)
        doc.ents = []
        spans = []

        for match_id, start, end in matches:

            label_ = nlp.vocab.strings[match_id]

            current_span = Span(
                doc, start, end, label=label_)

            if not token_from_span_in(spans, current_span):
                spans.append(current_span)

        doc.ents = spans

        if doc.ents:
            entities_docs.append(doc)
        else:
            no_entities_docs.append(doc)

    return entities_docs, no_entities_docs


def from_corpus(CORPUS_PATH):
    corpus_length = 0

    files_ = [str(x) for x in Path(CORPUS_PATH).glob("**/*.txt")]

    if files_:

        for file_path_ in files_:
            with open(file_path_, 'r', encoding="utf8") as f:
                sentences = list(f.readlines())

            corpus_length += len(sentences)

    else:
        print(f'Not files at {CORPUS_PATH}')
        sys.exit()

    return corpus_length, files_


if __name__ == '__main__':

    if len(sys.argv) == 4:
        args = sys.argv[1:]
        MODEL = args[0].split("=")[1]
        PATTERNS_PATH = args[1].split("=")[1]
        CORPUS_PATH = args[2].split("=")[1]
    else:
        print("Please check the arguments at the command line")
        sys.exit()

    print("\n" + ">>>>>>> Starting the entities tagging..........." + "\n")

    print(f'Loading the model ({MODEL})....')
    nlp = spacy.load(MODEL)
    print(".. done" + "\n")

    print(f'Loading the patterns ({PATTERNS_PATH})....')
    matcher = Matcher(nlp.vocab)
    patterns = list()
    setting_patterns(patterns, matcher)
    print(".. done" + "\n")

    print(f'Processing the corpus ({CORPUS_PATH})....')

    # Total of sentences in the corpus and the its list of files
    corpus_size, files = from_corpus(CORPUS_PATH)

    with_entities = []
    with_out_entities = []

    FILE_ON_PROCESS = 1

    if not len(files) == 0:

        for file_path in files:
            file_name = file_path.split("\\")[2]

            with open(file_path, 'r', encoding="utf8") as fl:
                SENTENCES = [line.strip() for line in fl.readlines()]

            print(
                f'..tagging entities for -> {file_name}: {FILE_ON_PROCESS} | {len(files)}')

            entities_docs, no_entities_docs = tagging_file_sentences(SENTENCES,
                                                                     matcher, nlp)
            with_entities += entities_docs
            with_out_entities += no_entities_docs

            FILE_ON_PROCESS += 1

        print(".... done")

        print(
            f'Sentences: {corpus_size}; with entities|without_entities: {len(with_entities)}|{len(with_out_entities)}')

        docs = with_entities + with_out_entities

        random.shuffle(docs)

        training_samples = math.floor(len(docs) * 0.7)

        train_docs = docs[:training_samples]
        dev_docs = docs[training_samples:]

        train_docbin = DocBin(docs=train_docs)
        train_docbin.to_disk("./train.spacy")

        dev_docbin = DocBin(docs=dev_docs)
        dev_docbin.to_disk("./dev.spacy")

    else:

        print("No files to tag. Please check the contents in the data/corpus folder" + "\n")
        sys.exit()

    print("\n" + ">>>>>>> Entities tagging finished...........")
