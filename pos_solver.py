#!/usr/bin/env python3
###################################
# (Based on skeleton code by D. Crandall)
#
#
####
# Training: 
# The following probabilities are calculated while training the dataset,
# 1. Initial state probabilities: Out of all the sentences, how much times a particular Part
#                                 of Speech starts the sentence.
# 2. Transition probabilities: Calculate the transition from one POS to another POS by taking subsequent
#                              pairs of words.
# 3. Emission probabilities: Calculate the probability of a any word occuring as different Parts of Speech
#
# Simplified model abstraction:
#  We used Naive Bayes method for our simplified model. Naive Bayes equation for calculating posterior probability
#  is as follows,
#                    P(A|B) = p(B|A) * P(A) / P(B)
#  Here, P(B)(Probability of a given word) is common among all words so we eliminate it in our calculations.
#  The calculated P(B) will be proportional to the actual probabilities. For calculating the probability of a
#  POS for a word, the emission probability of the POS given word is multiplied by the probability of the
#  POS occuring in the training data. Natural logarithms are taken for the calculated probabilities which converts
#  the probabilities to cost. For predicting the POS of a word we take the POS which has minimum cost.  
#
# HMM model abstraction:
#  As the HMM model has a chain structure, Viterbi decoding is used for predicting the parts of speech of
#  words. For calculating the POS probabilities of the first word the initial probability of each pos is
#  multiplied with their corresponding emission probabilities given the word and the calculated probability
#  is stored in a dictionary with the POS as the key. And to calculate the POS probabilities of the remaining
#  words the probability of transition from a particular POS of previous state to a particular POS of current
#  state is multiplied with the probability of the that POS in previous state. This is done for all POS in 
#  previous state and the maximum is taken and multiplied with the emission probability of that POS given word.
#  The same is repeated for all POS for the current state and the probabilities are stored in a dictionary.
#  As the probabilities are too low natural logarithm is taken to convert them to cost, higher the probability
#  lower the cost. The correct sequence of words is predicted by backtracking from the last word to the first 
#  word using the  path with minimum cost.
#
# Complex model abstraction:
#  As the complex model doesnot have a chain structure Markov Chain Monte Carlo is used instead of Viterbi 
#  algorithm. Initially all the words are assumed to be noun and gibbs sampling is done for 5000 iterations 
#  with warmup iterations as 4000. For calculating the probabilities of each pos for gibbs sampling the 
#  transition probabilities between the current state and previous state is multiplied with the transition 
#  probability between the current state and the state before the previous state and the emission probability
#  of a POS. The same is calculated for all POS for the word and given as the probability for sampling.
#  The POS with the max probability at the end of iteration is predicted as the POS.
#  
# Description of program working:
#  This program takes the bc.train as input to calculate the initial state probabilities, transition probabilities,
#  and emission probabilities using the tuple combination of word and POS provided by the skeleton code. The 
#  probabilities are stored in seperate dictionaries. Seperate functions are written for Simple, HMM, and complex 
#  model. So the program label.py takes two arguments (training dataset and test dataset). The model gets trained 
#  using the first argument (bc.train) and tests the trained model on the second argument(bc.test).
#
# Assumptions made:
#  1) It is obvious that emission probabilities of a POS will not have all the words in English language, in 
#     instances where a word is not in the dictionary of the POS for emission probability the probability is
#     assumed to be float(1/10**8).
#
# Word and sentency accuracy achieved with the bc.test dataset:
#                      Words correct:     Sentences correct: 
#      Ground truth:      100.00%              100.00%
#            Simple:       93.95%               47.60%
#               HMM:       95.07%               54.50%
#           Complex:       92.78%               42.05%
#                            
#                               
####

import random
import math
import numpy as np
from collections import Counter
import sys

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    
    POSLIST = []
    EMIPROB = {}
    TRANSPROB = {}
    INIPROB = {} 
    POSPROB = {}
    SECPROB = {}
    OTHERPROB = {}
    
    SENTENCE_COUNT = 0
    
    POSTSENTENCE_COUNT = 1
    
    POSTPROB = {"Simple" : {}, "Complex" : {}, "HMM" : {}}
    
    def posterior(self, model, sentence, label):
        '''
        This function takes in the model, the sentence and part of speech and returns 
        logrithm value of posterior probabilities calculated by each model. 
        
        Args:
        -----
        model : string
        sentence : list
        label : list
        
         Returns:
        --------
        cost : float
    
        '''
        
        if model == "Simple":
            
            cost = sum([((math.log(self.EMIPROB[label[i]][sentence[i]])) + (math.log(self.POSPROB[label[i]])) if sentence[i] in self.EMIPROB[label[i]] else (math.log(1/float(10**8))) + (math.log(self.POSPROB[label[i]]))) for i in range(len(sentence))])

            return cost
        
        elif model == "Complex":
            post_array = []
            for i in range(len(sentence)):
                if i == 0 :
                    post_array.append(self.EMIPROB[label[i]][sentence[i]] * self.INIPROB[label[i]] if sentence[i] in self.EMIPROB[label[i]] else (1/float(10**8)) * self.INIPROB[label[i]])
                elif i == 1:
                    post_array.append(self.EMIPROB[label[i]][sentence[i]] * (self.TRANSPROB[label[i-1]][label[i]] * self.POSPROB[label[i-1]] / self.POSPROB[label[i]])* self.POSPROB[label[i]] if sentence[i] in self.EMIPROB[label[i]] else (1/float(10**8)) * (self.TRANSPROB[label[i-1]][label[i]] * self.POSPROB[label[i-1]] / self.POSPROB[label[i]])* self.POSPROB[label[i]])
                else:
                    post_array.append(self.EMIPROB[label[i]][sentence[i]] * (self.TRANSPROB[label[i-1]][label[i]] * self.POSPROB[label[i-1]] / self.POSPROB[label[i]]) * (self.TRANSPROB[label[i-1]][label[i]] * self.POSPROB[label[i-2]] / self.POSPROB[label[i]] )* self.POSPROB[label[i]] if sentence[i] in self.EMIPROB[label[i]] else (1/float(10**8)) * (self.TRANSPROB[label[i-1]][label[i]] * self.POSPROB[label[i-1]] / self.POSPROB[label[i]]) * (self.TRANSPROB[label[i-2]][label[i]] * self.POSPROB[label[i-2]] / self.POSPROB[label[i]]) * self.POSPROB[label[i]])
            
            post_array = [math.log(p) for p in post_array]
            
            cost = sum(post_array)
            
            return cost
        
        elif model == "HMM":
            post_array = []
            for i in range(len(sentence)):
                if i  == 0:

                    post_array.append((((self.INIPROB[label[i]])) * ((self.EMIPROB[label[i]][sentence[i]]))) if sentence[i] in self.EMIPROB[label[i]] else  (((self.INIPROB[label[i]])) * (((1/float(10**8))))))

                else:
                    emi = (self.EMIPROB[label[i]][sentence[i]]) if sentence[i] in self.EMIPROB[label[i]] else ((1/float(10**8)))
                    
                    min_val = (post_array[i-1] * ((self.TRANSPROB[label[i-1]][label[i]])))
                
                    post_array.append(emi * min_val)
            
            post_array = [math.log(p) for p in post_array]
            
            cost = sum(post_array)
            
            return cost
        else:
            print("Unknown algo!")
        

    # Do the training!
    #
    def train(self, data):
        '''
        This function takes in the training data and calculates 
        the transistion probability, emission probability and initial probaility.
        
        Args:
        -----
        data : list
    
        '''
        
        pos_list = ['ADJ', 'ADV', 'ADP', 'CONJ', 'DET', 'NOUN', 'NUM', 'PRON', 'PRT', 'VERB', 'X', '.']
        pos_list = [pos.lower() for pos in pos_list]
        print("Inside train")
        
        wordpos_list = [tuple([line[0][i], line[1][i]]) for line in data for i in range(len(line[0]))]
        
        pos_dict = {pos : {} for pos in pos_list}
        
        wordpos_count = Counter(wordpos_list)

        for w in wordpos_count:
            pos_dict[w[1]].update({w[0] : wordpos_count[w]})
        
        
        pos_prob = {pos: float(sum(pos_dict[pos].values()))/len(wordpos_list) for pos in pos_dict.keys()}
                
        emi_prob = {pos : {word : float(pos_dict[pos][word])/sum(pos_dict[pos].values()) for word in pos_dict[pos].keys()} for pos in pos_dict.keys()}

        pair_list = []
        unique_list = []
        
        trans_count = {pos : {} for pos in pos_list}
        trans_prob = {pos : {} for pos in pos_list}
        
        pair_list = [tuple([line[1][i],line[1][i+1]]) for line in data for i in range(len(line[1])-1)]

        unique_list = list(set(pair_list))

        for element in unique_list:
            trans_count[element[0]].update({element[1] : pair_list.count(element)})
        
        for pos in pos_list:
            trans_prob[pos] = {pos : (1/float(10**8)) for pos in pos_list}
            for key,value in trans_count[pos].items():
                trans_prob[pos].update({key: (value/float(sum(trans_count[pos].values())))})
        
#        print(trans_prob)
        
        initial_prob = {pos : {} for pos in pos_list}
        
        initial_list =  [line[1][0] for line in data]

        initial_count = Counter(initial_list)
        
        initial_prob = {pos : float(initial_count[pos])/sum(initial_count.values()) for pos in initial_count.keys()}
        
        #print(initial_prob)
        
        
        self.POSLIST = pos_list
        
        self.EMIPROB = emi_prob
        
        self.TRANSPROB =  trans_prob
        
        self.INIPROB = initial_prob
        
        self.POSPROB = pos_prob
        
        
        

    def post_first(self, word):
        '''
        This function takes in the first word in a sentence and calculates 
        the posterior of the same for all part of speech.
        
        Args:
        -----
        word : string
    
        Returns:
        --------
        prob[word] : dictionary
        '''

        prob = {}
        if word not in prob.keys():
            prob[word] = {pos : self.EMIPROB[pos][word] * self.INIPROB[pos] if word in self.EMIPROB[pos] else (1/float(10**8)) * self.INIPROB[pos] for pos in self.POSLIST}
        
        return prob[word]
    
    def post_second(self, word, prev_pos):
        '''
        This function takes in the second word in a sentence and calculates 
        the posterior of the same for all part of speech.
        
        Args:
        -----
        word : string
    
        Returns:
        --------
        prob[word] : dictionary
        '''

        prob = {}
        if word not in prob.keys():
      
            prob[word] = {pos : self.EMIPROB[pos][word] * (float(self.TRANSPROB[prev_pos][pos] * self.POSPROB[prev_pos]) / self.POSPROB[pos])* self.POSPROB[pos] if word in self.EMIPROB[pos] else (1/float(10**8)) * (float(self.TRANSPROB[prev_pos][pos] * self.POSPROB[prev_pos]) / self.POSPROB[pos])* self.POSPROB[pos] for pos in self.POSLIST}
            
            
        return prob[word]
    
    def post_other(self, word, prev_pos, prev_pos_sec):
        '''
        This function takes in the word in a sentence that is other than first and second word. 
        Calculates the posterior of the same for all part of speech.
        
        Args:
        -----
        word : string
    
        Returns:
        --------
        prob[word] : dictionary
        '''
        
        prob = {}
        if word not in prob.keys():
  
            prob[word] = {pos : self.EMIPROB[pos][word] * (float(self.TRANSPROB[prev_pos][pos] * self.POSPROB[prev_pos]) / self.POSPROB[pos]) * (float(self.TRANSPROB[prev_pos_sec][pos] * self.POSPROB[prev_pos_sec]) / self.POSPROB[pos] )* self.POSPROB[pos] if word in self.EMIPROB[pos] else (1/float(10**8)) * (float(self.TRANSPROB[prev_pos][pos] * self.POSPROB[prev_pos]) / self.POSPROB[pos]) * (float(self.TRANSPROB[prev_pos_sec][pos] * self.POSPROB[prev_pos_sec]) / self.POSPROB[pos]) * self.POSPROB[pos] for pos in self.POSLIST}
            
        return prob[word]
            
    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        '''
        This function takes in a sentence and returns a list of POS label 
        calculated using Naive Bayes algorithm.
        
        Args:
        -----
        sentence : list
    
        Returns:
        --------
        sequence : list
        '''
        
        #print("Inside Simplified")
        
        output_list = [ self.POSLIST[np.argmin([(-math.log(self.EMIPROB[pos][sentence[i]])) + (-math.log(self.POSPROB[pos])) if sentence[i] in self.EMIPROB[pos] else (-math.log(1/float(10**8))) + (-math.log(self.POSPROB[pos])) for pos in self.POSLIST ])] for i in range(len(sentence))]
        
        return output_list

    def complex_mcmc(self, sentence):
        '''
        This function takes in a sentence and returns a list of POS label 
        calculated using MCMC algorithm.
        
        Args:
        -----
        sentence : list
    
        Returns:
        --------
        sequence : list
        '''
        iteration = 5000
        warmup = 2000
        
#        print("Inside MCMC")
        
        pos_mcmc_dict = {"POS_" + str(i) : {} for i in range(len(sentence))}
        
        sequence = ['noun'] * len(sentence)
        for i in range(len(sentence)):
            if i == 0:
                prob_first = self.post_first(sentence[i])

                sample_first = list(np.random.choice([keys for keys in prob_first.keys()], iteration, p = [float(prob_first[keys])/sum(prob_first.values()) for keys in prob_first.keys()]))
                
                sample_first = sample_first[warmup:] 

                pos_mcmc_dict["POS_" + str(i)] = {pos :  (float(sample_first.count(pos))/len(sample_first)) for pos in self.POSLIST }

                sequence[i] = max(pos_mcmc_dict["POS_" + str(i)], key = pos_mcmc_dict["POS_" + str(i)].get)
            
            elif i == 1:
                prob_second = self.post_second(sentence[i], sequence[i-1]) 
                
                sample_second = list(np.random.choice([keys for keys in prob_second.keys()], iteration, p = [float(prob_second[keys])/sum(prob_second.values()) for keys in prob_second.keys()]))
                sample_second = sample_second[warmup:] 
                
                pos_mcmc_dict["POS_" + str(i)] = {pos :  (float(sample_second.count(pos))/len(sample_second)) for pos in self.POSLIST }

                sequence[i] = max(pos_mcmc_dict["POS_" + str(i)], key = pos_mcmc_dict["POS_" + str(i)].get)
                
            else:
                prob_other= self.post_other(sentence[i], sequence[i-1], sequence[i-2]) 
                
                sample_other = list(np.random.choice([keys for keys in prob_other.keys()], iteration, p = [float(prob_other[keys])/sum(prob_other.values()) for keys in prob_other.keys()]))
                sample_other = sample_other[warmup:] 
                
                pos_mcmc_dict["POS_" + str(i)] = {pos :  (float(sample_other.count(pos))/len(sample_other)) for pos in self.POSLIST }

                sequence[i] = max(pos_mcmc_dict["POS_" + str(i)], key = pos_mcmc_dict["POS_" + str(i)].get)

        return sequence

    def hmm_viterbi(self, sentence):
        
        '''
        This function takes in a sentence and returns a list of POS label 
        calculated using Viterbi algorithm.
        
        Args:
        -----
        sentence : list
    
        Returns:
        --------
        sequence : list
        '''
        
#        print("Inside Viterbi")
      
        viterbi_dict = {i : {} for i in range(len(sentence))}
        
        for i in range(len(sentence)):
            
            temp_dict = {}
            
            if i  == 0:
                for pos in self.POSLIST:
                    
                    temp_dict[pos] = tuple([(((- math.log(self.INIPROB[pos])) + (- math.log(self.EMIPROB[pos][sentence[i]]))) if sentence[i] in self.EMIPROB[pos] else  ((- math.log(self.INIPROB[pos])) + (- math.log((1/float(10**8)))))), "Start"])
                    
                viterbi_dict[i] = temp_dict

            else:
                for pos in self.POSLIST:
                    emi = - math.log(self.EMIPROB[pos][sentence[i]]) if sentence[i] in self.EMIPROB[pos] else (- math.log((1/float(10**8))))
                    
                    min_val = min([ ((viterbi_dict[i-1][pos_prev][0] + (- math.log(self.TRANSPROB[pos_prev][pos]))), pos_prev) for pos_prev in self.POSLIST])
                
                    temp_dict[pos] = tuple([emi + min_val[0], min_val[1]])
                    
                viterbi_dict[i] = temp_dict
            
            
            i = i + 1
        

        pos_list = []
        prev = ""
        for i in range(len(sentence) - 1, -1, -1):
            
            if i == len(sentence) - 1:
                minimum = min(list(viterbi_dict[i].values()))

                for each in viterbi_dict[i].keys():
                    if viterbi_dict[i][each][0] == minimum[0] and viterbi_dict[i][each][1] == minimum[1]:

                        pos_list.append(each)
                        prev = minimum[1]
            else:
                pos_list.append(prev)
                prev = viterbi_dict[i][prev][1]
                    
        pos_list.reverse()
        
        
        return pos_list
    
    

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        
        '''
        This function takes in a sentence and a model and applies the model to the sentence and returns the
        list of labels.
        
        Args:
        -----
        model : string
        sentence : list
    
        Returns:
        --------
        list
        '''
        
        #print("Inside Solve")
        
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")

