#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 13:19:15 2017

@author: thiagodepaulo
"""
import os
import sys

def read_text(arq):
    f = open(arq,'r')
    content = f.read()
    return content.strip()

def load_directory(root_path):
    corpus = {}
    for root, dirs, files in os.walk(root_path):        
        for arq in files:
            if arq.endswith('.txt'):                           
                key = os.path.basename(root)
                if key not in corpus:
                    corpus[key] = []
                corpus[key].append(read_text(os.path.join(root,arq)))
    return corpus                

def save_corpus(arq, corpus, rem_virgula=False):
    def remove_virgula(txt):
        if rem_virgula:
            return txt.replace(',',' ')
        return txt
    with open(arq,'w') as fout:
        for k in corpus.keys():            
            fout.write('\n'.join([k+', '+remove_virgula(content) for content in corpus[k]]))
    
    

                
