"""
Script to tokenize the dataset into a list of pairs where in the first position of
the pair there is the tokenized sentence in the source language (as a list) and in the second one 
the tokenized sentence in the target language.

Example:
-Parallele sentences in the dataset: 
('In the U.S., it is illegal to torture people in order to get information from them.',
 'Negli Stati Uniti è illegale torturare le persone per poter ottenere informazioni da loro.') 
 
 -After tokenization: 
 ([818,262,471,13,50,1539,340,318,5293,284,11543,661,287,1502,284,651,1321,422,606,13,50258],
 [32863,4528,5133,72,791,8846,6184,101,4416,1000,7619,333,533,443,2774,505,583,1787,263,267,32407,567,4175,1031,295,72,12379,300,16522,13,50258])
 
In general the tokenized italian sentece is longer than the english one beacuse i 
used gpt2 tokenizer which works really well on english texts and eas primarly designed for them.
"""

import pickle 
from tokenizer import enc

# Function to read parallel sentences from two separate files
parallel_sentences = []
with open('../data/europarl_it.txt', 'r', encoding='utf-8') as it_file:
    italian_sentences = it_file.readlines()

with open('../data/europarl_en.txt', 'r', encoding='utf-8') as en_file:
    english_sentences = en_file.readlines()

# Ensure the same number of sentences in both files
assert len(italian_sentences) == len(english_sentences), "Files do not have the same number of sentences!"

parallel_sentences = (zip(english_sentences, italian_sentences))

# tokenize the sentences adding the end of sequence token
tokenized_parallel_sentences = [(enc.encode(eng + "</s>", allowed_special={'</s>'}), enc.encode(it + "</s>", allowed_special={'</s>'}))
                                for eng, it in parallel_sentences]    

# Save the tokenized sentences into a file
with open('../data/tokenized_parallel_sentences.pkl', 'wb') as f:
    pickle.dump(tokenized_parallel_sentences, f)    
