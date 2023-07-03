import re
import spacy
import errant
import en_core_web_sm
from happytransformer import TTSettings

# Define asr function
def speech_to_text(model_base, audio_path):
    result = model_base.transcribe(audio_path)
    return result["text"]

# Define grammar function
def grammar_test(model_base, text):
    beam_settings =  TTSettings(num_beams=8, min_length=1, max_length=1024)
    text_generate = "grammar: " + text
    result = model_base.generate_text(text_generate, args=beam_settings)
    return result.text

# Define error calculation function
def error_calculation(annotator, original_text = "", correction_text = ""):
    orig = annotator.parse(original_text)
    cor = annotator.parse(correction_text)
    edits = annotator.annotate(orig, cor)

    correct_text_len = len(re.findall(r"\w+", correction_text))
    error_len = len(edits)

    percent = (1 - (error_len/correct_text_len)) * 100

    return percent

# Define level function
from collections import Counter

def to_level(band_eval):
    band = 0
    list_isSimple = []
    list_error = []

    if band_eval.asr_text == "":
        band = 0
    else:
        if len(band_eval.sentences) == 1:
            if band_eval.sentences[0].isSimple == True:
                band = 2
            else:
                if band_eval.sentences[0].error < 40:
                    band = 2
                elif band_eval.sentences[0].error < 80 and band_eval.sentences[0].error >= 40:
                    band = 3
                else:
                    band = 4
        else:
            for sentence in band_eval.sentences:
                list_isSimple.append(sentence.isSimple)
                list_error.append(sentence.error)
              
            result_isSimple = all(element == True for element in list_isSimple)

            if (result_isSimple):
                average_error = sum(list_error) / len(list_error)

                if average_error < 50:
                    band = 3
                else:
                    band = 4
            else:
                counter = Counter(list_isSimple)

                max_count = max(counter.values())
                mode = [k for k,v in counter.items() if v == max_count]

                if len(mode) == 1:
                    if mode[0] == True:
                        band = 5
                    else:
                        average_error = sum(list_error) / len(list_error)
                        if average_error < 50:
                            band = 7
                        else:
                            band = 8
                else:
                    average_error = sum(list_error) / len(list_error)
                    if average_error < 50:
                        band = 6
                    else:
                        band = 7

                result_isSimple = all(element == False for element in list_isSimple)

                if (result_isSimple):
                    average_error = sum(list_error) / len(list_error)

                    if average_error < 50:
                        band = 8
                    else:
                        band = 9

    return band
    
# Define function to find root
def find_root_of_sentence(doc):
    root_token = []

    for token in doc:
        if (token.dep_ == "ROOT"):
            root_token.append(token)

    return root_token

# Define function to find verb
def find_other_verbs(doc, root_token):
    other_verbs = []
    for token in doc:
        ancestors = list(token.ancestors)
        for root in root_token:
            if (token.pos_ == "VERB" and len(ancestors) == 1\
                and ancestors[0] == root):
                other_verbs.append(token)
    return other_verbs

# Define function to get clause token
def get_clause_token_span_for_verb(verb, doc, all_verbs):
    this_verb_children = list(verb.children)
    list_of_token = []
    list_of_token.append(verb.i)
    for child in this_verb_children:
        depend_child = list(child.children)
        if (child not in all_verbs):
            list_of_token.append(child.i)
            if len(depend_child) != 0:
                for d_child in depend_child:
                    depend_child_2 = list(d_child.children)
                    if (d_child not in all_verbs):
                        list_of_token.append(d_child.i)
                        if len(depend_child_2) != 0:
                            for d_child_2 in depend_child_2:
                                if (d_child_2 not in all_verbs):
                                    list_of_token.append(d_child_2.i)

    return list_of_token

# Define evaluation object
class Sentence():
  def __init__(self, text="", correct="", isSimple=False, error=0, clauses=[]):
    self.text = text
    self.correct = correct
    self.isSimple = isSimple
    self.error = error
    self.clauses = clauses

class GrammarEval():
  def __init__(self, asr_text="", sentences=[]):
    self.asr_text = asr_text
    self.sentences = sentences

# Define IELTS Grammar function
def grammar(model_gec, model_whisper, wav, evaluation):
    nlp_annotate = en_core_web_sm.load()
    annotator = errant.load('en', nlp_annotate)    

    # annotator = errant.load("en")
    
    nlp = spacy.load("en_core_web_sm")
    asr = speech_to_text(model_whisper, wav)
    evaluation.asr_text = asr
    evaluation.sentences = []
    doc = nlp(evaluation.asr_text)

    for sent in doc.sents:
        isSimple = False
        correct = grammar_test(model_gec, sent.text)
        error_percent = error_calculation(annotator, sent.text, correct)
        doc_sentence = nlp(sent.text)
        root_token = find_root_of_sentence(doc_sentence)
        other_verbs = find_other_verbs(doc_sentence, root_token)
        token_spans = []
        all_verbs = root_token + other_verbs

        if len(all_verbs) == 1:
            isSimple = True

        for other_verb in all_verbs:
            list_of_token = get_clause_token_span_for_verb(other_verb, 
                                            doc, all_verbs)
            token_spans.append(sorted(list_of_token))
            
        sentence_clauses = []
        for token_span in token_spans:
            clause = []
            for token in token_span:
                text = doc[token]
                clause.append(text)
            clause_text = " ".join(map(str, clause))
            sentence_clauses.append(clause_text)
        
        remove_sentence_clauses = []
        for sentence in sentence_clauses:
            words = len(re.findall(r'\w+', sentence))
            if words == 1:
                remove_sentence_clauses.append(sentence)
        
        for remove in remove_sentence_clauses:
            if remove in sentence_clauses:
                sentence_clauses.remove(remove)
        
        if len(sentence_clauses) == 1:
            isSimple = True

        eval_sentence = Sentence(text=sent.text, 
                                 correct=correct, 
                                 isSimple=isSimple, 
                                 error=error_percent, 
                                 clauses=sentence_clauses
                                 )
        
        evaluation.sentences.append(eval_sentence)

    return evaluation