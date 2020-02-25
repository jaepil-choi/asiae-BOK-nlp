# -*- coding: utf-8 -*-
from ekonlpy.sentiment import MPCK

import pandas as pd
import multiprocessing as mp
import pickle

import os, sys
import re

def tidy_sentences(section):
    sentence_enders = re.compile(r'((?<=[함음됨임봄짐움])(\s*\n|\.|;)|(?<=다)\.)\s*')
    splits = list((m.start(), m.end()) for m in re.finditer(sentence_enders, section))
    starts = [0] + [i[1] for i in splits]
    ends = [i[0] for i in splits]
    sentences = [section[start:end] for start, end in zip(starts[:-1], ends)]
    for i, s in enumerate(sentences):
        sentences[i] = (s.replace('\n', ' ').replace(' ', ' ')) + '.'

    text = '\n'.join(sentences) if len(sentences) > 0 else ''

    return sentences, text

def preprocess_minutes(minutes):
    pos = re.search('(.?국내외\s?경제\s?동향.?과 관련하여,?|\(가\).+경제전망.*|\(가\) 국내외 경제동향 및 평가)\n?\s*일부 위원은', minutes, re.MULTILINE)
    s1 = pos.start() if pos else -1
    pos = re.search('(.?외환.?국제금융\s?동향.?과 관련하여.*|\(나\) 외환.국제금융\s?(및 금융시장)?\s?동향)\n?\s*(일부 위원은|대부분의 위원들은)', minutes,re.MULTILINE)
    s2 = pos.start() if pos else -1
    pos = re.search('(.?금융시장\s?동향.?과 관련하여,?|\(다\) 금융시장\s?동향)\n?\s*일부 위원은', minutes, re.MULTILINE)
    s3 = pos.start() if pos else -1
    pos = re.search('((\((다|라)\) )?.?통화정책\s?방향.?에 관한 토론,?|이상과 같은 의견\s?교환을 바탕으로.*통화정책\s?방향.*에.*토론.*)\n?', minutes,re.MULTILINE)
    s4 = pos.start() if pos else -1
    pos = re.search('(\(4\) 정부측 열석자 발언.*)\n?', minutes, re.MULTILINE)
    s5 = pos.start() if pos else -1
    pos = re.search('(\(.*\) 한국은행 기준금리 결정에 관한 위원별 의견\s?개진|이상과 같은 토론에 이어 .* 관한 위원별 의견개진이 있었음.*)\n?', minutes,re.MULTILINE)
    s6 = pos.start() if pos else -1
    positer = re.finditer('(\(\s?.*\s?\) ()(심의결과|토의결론))\n?', minutes, re.MULTILINE)
    s7 = [pos.start() for pos in positer if pos.start() > s6]
    s7 = s7[0] if s7 else -1

    # 국내외 경제동향
    bos = s1
    eos = s2
    section = minutes[bos:eos] if bos >= 0 or eos >= 0 else ''
    pos = re.search('(일부|대부분의) 위원들?은', section, re.MULTILINE)
    bos = pos.start() if pos else -1
    section = section[bos:] if bos >= 0 else section
    section1, section1_txt = tidy_sentences(section)

    # 외환․국제금융 동향
    bos = s2
    eos = s3 if s3 >= 0 else s4
    section = minutes[bos:eos] if bos >= 0 or eos >= 0 else ''
    pos = re.search('(일부|대부분의) 위원들?은', section, re.MULTILINE)
    bos = pos.start() if pos else -1
    section = section[bos:] if bos >= 0 else section
    section2, section2_txt = tidy_sentences(section)
    #print(section)

    # 금융시장 동향
    bos = s3
    eos = s4
    section = minutes[bos:eos] if bos >= 0 or eos >= 0 else ''
    pos = re.search('(일부|대부분의) 위원들?은', section, re.MULTILINE)
    bos = pos.start() if pos else -1
    section = section[bos:] if bos >= 0 else section
    section3, section3_txt = tidy_sentences(section)

    # 통화정책방향
    bos = s4
    eos = s5 if s5 >= 0 else s6 if s6 >= 0 else s7
    section = minutes[bos:eos] if bos >= 0 or eos >= 0 else ''
    pos = re.search('(일부|대부분의) 위원들?은', section, re.MULTILINE)
    bos = pos.start() if pos else -1
    section = section[bos:] if bos >= 0 else section
    section4, section4_txt = tidy_sentences(section)

    # 위원별 의견 개진
    bos = s6
    eos = s7
    section = minutes[bos:eos] if bos >= 0 or eos >= 0 else ''
    pos = re.search('(일부|대부분의) 위원들?은', section, re.MULTILINE)
    bos = pos.start() if pos else -1
    section = section[bos:] if bos >= 0 else section
    section5, section5_txt = tidy_sentences(section)

    # 정부측 열석자 발언
    bos = s5
    eos = s6
    section = minutes[bos:eos] if bos >= 0 or eos >= 0 else ''
    pos = re.search('정부측 열석자 발언', section, re.MULTILINE)
    bos = pos.end() + 1 if pos else -1
    section = section[bos:] if bos >= 0 else section
    section6, section6_txt = tidy_sentences(section)

    sections = ['Economic Situation', 'Foreign Currency', 'Financial Markets',
                'Monetary Policy', 'Participants’ Views', 'Government’s View']
    section_texts = (section1, section2, section3, section4, section5, section6)

    return sections, section_texts

def getStopwords(data_path) :
    f = open(data_path, 'r',-1,'utf-8')
    stop_word = []

    while True:
        line = f.readline()
        stop_word.append(line.strip())
        if not line: break

    return stop_word

def text2ngram(text) :
    mpck = MPCK()
    
    tokens = mpck.tokenize(text)
    tokens = [token for token in tokens if len(token.split('/')[0])>1]
    tokens = [token for token in tokens if token.split('/')[0] not in stop_word]
    
    ngrams = mpck.ngramize(tokens)

    return tokens+ngrams

data_path = './data_files/stop_wd.txt'
stop_word = getStopwords(data_path)