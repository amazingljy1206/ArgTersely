import nltk
import numpy as np
from rouge import Rouge
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

model = 'gpt3' # bart, gpt2, dialogpt, llama, alpaca, argllama, gpt3
rouge_scores = []
bleu_scores = []
meteor_scores = []
word_nums = []
rouge = Rouge()
smooth = SmoothingFunction()

def tokenizer(text):
    stop_words = nltk.corpus.stopwords.words('english')
    return [token for token in word_tokenize(text) if token not in stop_words]

with open('argtersely/test/counter-argument.txt', 'r', encoding='UTF-8') as f:
    refs = [line.strip() for line in f.readlines()]
with open('experiments/{}_res.txt'.format(model), 'r', encoding='UTF-8') as f:
    hyps = [line.strip() for line in f.readlines()]

for ref, hyp in zip(refs, hyps):
    if hyp == '':
        hyp = 'UNK'
    rouge_scores.append(rouge.get_scores(hyp, ref, avg=True)["rouge-l"]['r'])
    bleu_scores.append(sentence_bleu([tokenizer(ref)], tokenizer(hyp), weights = (1,0,0,0), smoothing_function=smooth.method1))
    meteor_scores.append(meteor_score([tokenizer(ref)], tokenizer(hyp)))
    word_nums.append(len(tokenizer(hyp)))
print('BLEU-1 score:{:.4f}'.format(np.mean(bleu_scores)))
print('ROUGE-L score:{:.4f}'.format(np.mean(rouge_scores)))
print('METEOR score:{:.4f}'.format(np.mean(meteor_scores)))
print('# Word:{:.0f}'.format(np.mean(word_nums)))