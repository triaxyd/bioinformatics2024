
import random
import numpy as np
from math import log, ceil
import os
import ete3

#----------------------------------------------
#DEFINE PATTERNS
def change_pattern(pattern, alphabet):
    #change up to 2 characters, either by a random character 
    # or by removing the character and replacing it with '_'
    r = np.random.randint(0, 3)    
    for _ in range(r):
        if np.random.choice([True, False]):
            #either change character of random index to other character from alphabet but not the same
            index = np.random.randint(0, len(pattern))
            new_char = np.random.choice([x for x in alphabet if x != pattern[index]])
            pattern[index] = new_char
        else:
            #either delete character of random index and replace it with empty space
            pattern[np.random.randint(0, len(pattern))] = ''
    return pattern


def synthesize(alphabet):
    
    #create patterns
    pattern1 = ['A', 'A', 'T', 'T', 'G', 'A']
    pattern2 = ['C', 'G', 'C', 'T', 'T', 'A', 'T']
    pattern3 = ['G', 'G', 'A', 'C', 'T', 'C', 'A', 'T']
    pattern4 = ['T', 'T', 'A', 'T', 'T', 'C', 'G', 'T', 'A']

    patterns = [pattern1, pattern2, pattern3, pattern4]

    sequence = ''

    #randomly add 1 to 3 characters from alphabet to the start of the sequence
    sequence = sequence.join(np.random.choice(alphabet, np.random.randint(1, 4)))

    #change patterns and add them to the sequence
    for pattern in patterns:
        pattern = change_pattern(pattern, alphabet)
        sequence = sequence + ''.join(pattern)

    #randomly add 1 or 2 characters from alphabet to the end of the sequence
    sequence = sequence + ''.join(np.random.choice(alphabet, np.random.randint(1, 3)))

    return sequence


#----------------------------------------------
#NEEDLEMAN-WUNSCH ALGORITHM FOR GLOBAL SEQUENCE ALIGNMENT

#needleman-wunsch algorithm for global sequence alignment (with gap penalty = -2 , mismatch penalty = -a/2, match reward = 1)
def needleman_wunsch(sequence1, sequence2, match = 1, mismatch = -1, gap = -2):#sequences and scoring system as inputs
    len1 = len(sequence1)#vertical sequence to the left
    len2 = len(sequence2)#horizontal sequence to the top
    #create matrix with dimensions len(sequence1)+1 x len(sequence2)+1
    matrix = np.zeros((len1 + 1, len2 + 1), dtype=int)#+1 as extra column/row for initialized values

    #initialize the first row and column with gap penalties
    for i in range(1, len1 + 1):
        matrix[i][0] = matrix[i-1][0] + gap #as we move through the columns, we are adding more gaps to sequences
    for j in range(1, len2 + 1):
        matrix[0][j] = matrix[0][j-1] + gap #same for the rows

    #fill matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if sequence1[i-1] == sequence2[j-1]: #if the characters match, we calculate using the match score
                matrix[i][j]= max(matrix[i][j-1]+gap, matrix[i-1][j]+gap, matrix[i-1][j-1]+match)#pick the best score out of the three moves
            else:#if not we use the mismatch score
                matrix[i][j]= max(matrix[i][j-1]+gap, matrix[i-1][j]+gap, matrix[i-1][j-1]+mismatch)

    #now we backtrack and compare
    alignedseq1=""
    alignedseq2=""

    i=len1
    j=len2

    while(i>0 or j>0):
        if sequence1[i-1] == sequence2[j-1]:#if the characters match append and go to the next diagonal char
            alignedseq1+= sequence1[i-1]
            alignedseq2+= sequence2[j-1]
            i-=1
            j-=1
        elif sequence1[i-1]!=sequence2[j-1]:#if they dont match, we create a list to find the max value in order to backtrack
            mismatch_list= [matrix[i-1][j-1], matrix[i-1][j], matrix[i][j-1]]

            if max(mismatch_list) ==mismatch_list[0] :#if the diagonal is max, backtrack there
                alignedseq1+= sequence1[i-1]
                alignedseq2+= sequence2[j-1]
                i-=1
                j-=1
            if max(mismatch_list)==mismatch_list[1]:#if the top value is max
                alignedseq1+=sequence1[i-1]# keep the char from the top
                alignedseq2+="-"#put a gap for the other
                i-=1#move up
            if max(mismatch_list)==mismatch_list[2]:#if the left value is max
                alignedseq1+="-"#put a gap for the mismatched char
                alignedseq2+=sequence2[j-1]# keep the char from the left
                j-=1#move left

    alignedseq1=alignedseq1[::-1]
    alignedseq2=alignedseq2[::-1]

    return matrix[len1][len2], alignedseq1, alignedseq2
  

#----------------------------------------------
#PROGRESSIVE MULTIPLE SEQUENCE ALIGNMENT ALGORITHM
#CALCULATION OF SCORE FOR EACH PAIR
def calc_pair_scores(sequences, match=1, mismatch=-1, gap=-2):
    n = len(sequences)
    scores = [[0 for _ in range(n)] for _ in range(n)]  # Use list of lists instead of np.zeros
    for i in range(n):
        for j in range(i + 1, n):  # one sequence ahead
            score, _, _ = needleman_wunsch(sequences[i], sequences[j])
            scores[i][j] = score
            scores[j][i] = score
    return scores  # return the score matrix as a list of lists

def cal_pair_distances(sequences, score_matrix):
    n = len(sequences)
    distance_matrix = [[0 for _ in range(n)] for _ in range(n)]  # Use list of lists
    for i in range(n):
        for j in range(i + 1, n):
            score = score_matrix[i][j]
            distance = -score
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance
    return distance_matrix  # return the distance matrix as a list of lists

#building guide trees using UPGMA
def find_min_distance(distance_matrix):
    min_val= float('inf')#initial value is +inf
    x,y=-1,-1#variables to store the indices of a pair
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[i])):
            if distance_matrix[i][j]<min_val:
                min_val= distance_matrix[i][j]
                x,y=i,j
    return x,y
#####################################################################
#UPGMA- Unweighted Pair Group Method with Arithmetic Mean- Making guide trees in order to implement progressive alignment
'''
Upgma steps
1. start with each sequence as its own cluster
2. find the pair of clusters with the smallest distance and merge them
3. update the distance matrix
4. repeat until all sequencies are clustered into one tree
'''

def create_labels(sequences):#creating labels in order to cluster sequences
    return [chr(ord('A') + i) for i in range(len(sequences))]


def join_labels(labels,clust1,clust2):#combines the labels of two clusters(lists of sequences)
    if clust2 < clust1:#swap indices if they are not in order, to avoid potential errors
        clust1, clust2 = clust2, clust1
    labels[clust1]="("+labels[clust1]+","+labels[clust2]+")"#join cluster on smaller index
    del labels[clust2]#delete the other cluster

def update_distance_matrix(distance_matrix, clust1, clust2):
    if clust2 < clust1:
        clust1, clust2 = clust2, clust1
    
    for i in range(0, clust1):# reconstruct the row for the merged cluster
        distance_matrix[clust1][i] = (distance_matrix[clust1][i] + distance_matrix[clust2][i]) / 2
    
    for i in range(clust1 + 1, clust2):
        distance_matrix[i][clust1] = (distance_matrix[i][clust1] + distance_matrix[clust2][i]) / 2

    for i in range(clust2 + 1, len(distance_matrix)):
        distance_matrix[i][clust1] = (distance_matrix[i][clust1] + distance_matrix[i][clust2]) / 2

    for row in distance_matrix:# delete the clust2 row and column
        del row[clust2]
    
    del distance_matrix[clust2]

def UPGMA(distance_matrix, labels):
    while len(labels)>1:
        x,y=find_min_distance(distance_matrix)
        update_distance_matrix(distance_matrix,x,y)
        join_labels(labels,x,y)
    return labels[0]

#TREE PARSING
def parse_guide_tree(newick_str):
    t = ete3.Tree(newick_str + ';')
    return t
###############################
#implement progressive alignment

def progressive_alignment(sequences,guide_tree):
    label_to_sequence = {label: sequence for label, sequence in zip(create_labels(sequences), sequences)}
    alignments={}#dictionary to store alignments
    for node in guide_tree.traverse("postorder"):#traverse the tree in post-order
        if node.is_leaf():#if the node is a leaf
            alignments[node.name]=label_to_sequence[node.name] #retrieve sequence
        else:#else align sequences from children
            children=node.get_children()
            if len(children)==2:#if there are 2 children, retrieve them
                align1=alignments[children[0].name]
                align2=alignments[children[1].name]
                _,aligned_seq1,aligned_seq2=needleman_wunsch(align1,align2)
                alignments[node.name]=aligned_seq1

    final_alignment = alignments[guide_tree.name]
    return final_alignment

def align_to_final(final, sequences):#align all sequences to final sequence
    aligned_sequences = {}
    for seq in sequences:
        _, aligned_ref, aligned_seq = needleman_wunsch(final, seq)
        aligned_sequences[seq] = aligned_seq
    max_aligned_length = max(len(aligned_seq) for aligned_seq in aligned_sequences.values())#find max len of aligned and pad accordingly
    # Pad sequences to ensure they all have the same length
    for seq in aligned_sequences:
        aligned_seq = aligned_sequences[seq]
        if len(aligned_seq) < max_aligned_length:
            aligned_sequences[seq] = aligned_seq + '-' * (max_aligned_length - len(aligned_seq))
    
    return aligned_sequences





def _get_states(i):
    return 'M{}'.format(i), 'I{}'.format(i), 'D{}'.format(i)



#----------------------------------------------
#HMM FOR MULTIPLE SEQUENCE ALIGNMENT
class profile_HMM:

    def __init__(self, sequences):
        self.sequences = sequences
        self.t_prob = {}  
        self.e_prob = {}  
        self.char_list = set()

        #remove gaps from sequences
        for sequence in sequences:
            self.char_list = self.char_list.union(set(sequence))
        self.char_list.discard('-') 

        self.num_of_strings = len(sequences)
        self.num_of_chars = len(sequences[0])

        self.frequency_list = [{} for _ in range(self.num_of_chars + 1)]
        for sequence in sequences:
            for index, char in enumerate(sequence):
                if char in self.frequency_list[index]:
                    self.frequency_list[index][char] += 1
                else:
                    self.frequency_list[index][char] = 1

        #determine match states 
        self.match_states = [
            k for n, k in zip(self.frequency_list, range(self.num_of_chars + 1))
            if int(n.get('-', 0)) < ceil(self.num_of_strings / 2)
        ]

        #initialize states
        match_state = ['M{}'.format(k) for k in range(0, len(self.match_states) + 1)]
        insert_state = ['I{}'.format(k) for k in range(0, len(self.match_states))]
        delete_state = ['D{}'.format(k) for k in range(1, len(self.match_states))]


        #initialize transition probabilities
        self.t_prob.update({key: {'strs': []} for key in match_state})
        self.t_prob.update({key: {'strs': []} for key in insert_state})
        self.t_prob.update({key: {'strs': []} for key in delete_state})
        self.t_prob['M0']['strs'] = [n for n in range(self.num_of_strings)]


    def build_model(self):
        i = 0  #counter for positions in sequences
        j = 0  #current state index

        while i < self.num_of_chars + 1:
            M, I, D = _get_states(j)
            nextM, nextD = _get_states(j + 1)[::2]


            #if the current state is match state
            if i in self.match_states:
                deltodel, deltomatch = [], []
                instomatch, instodel = [], []
                matchtodel, matchtomatch = [], []

                # D --> D and D --> M
                if self.t_prob.get(D, {}).get('strs', []) and i != 0:
                    try:
                        deltodel = [n for n in self.t_prob[D]['strs'] if self.sequences[n][i] == '-']
                    except IndexError:
                        pass
                    deltomatch = [n for n in self.t_prob[D]['strs'] if n not in deltodel]

                    # D --> D
                    if deltodel:
                        self.t_prob[D][nextD] = {
                            'prob': float(len(deltodel) / len(self.t_prob[D]['strs'])),
                            'strs': deltodel
                        }
                        self.t_prob[nextD]['strs'].extend(deltodel)

                    # D --> M
                    if deltomatch:
                        self.t_prob[D][nextM] = {
                            'prob': float(len(deltomatch) / len(self.t_prob[D]['strs'])),
                            'strs': deltomatch
                        }
                        self.t_prob[nextM]['strs'].extend(deltomatch)

                # I --> M and I --> D
                if self.t_prob[I]['strs'] and i != 0:
                    try:
                        instodel = list(set([n for n in self.t_prob[I]['strs'] if self.sequences[n][i] == '-']))
                    except IndexError:
                        pass
                    instomatch = list(set([n for n in self.t_prob[I]['strs'] if n not in instodel]))

                    # I --> D
                    if instodel:
                        self.t_prob[I][nextD] = {
                            'prob': float(len(instodel) / len(self.t_prob[I]['strs'])),
                            'strs': instodel
                        }
                        self.t_prob[nextD]['strs'].extend(instodel)

                    # I --> M
                    if instomatch:
                        self.t_prob[I][nextM] = {
                            'prob': float(len(instomatch) / len(self.t_prob[I]['strs'])),
                            'strs': instomatch
                        }
                        self.t_prob[nextM]['strs'].extend(instomatch)

                # M --> D and M --> M
                if self.t_prob[M]['strs']:
                    try:
                        matchtodel = [n for n in self.t_prob[M]['strs'] if self.sequences[n][i] == '-' and n not in self.t_prob[I]['strs']]
                    except IndexError:
                        pass

                    matchtomatch = [n for n in self.t_prob[M]['strs'] if n not in matchtodel + self.t_prob[I]['strs']]

                    # M --> D
                    if matchtodel:
                        self.t_prob[M][nextD] = {
                            'prob': float(len(matchtodel) / len(self.t_prob[M]['strs'])),
                            'strs': matchtodel
                        }
                        self.t_prob[nextD]['strs'].extend(matchtodel)

                    # M --> M
                    if matchtomatch:
                        self.t_prob[M][nextM] = {
                            'prob': float(len(matchtomatch) / len(self.t_prob[M]['strs'])),
                            'strs': matchtomatch
                        }
                        self.t_prob[nextM]['strs'].extend(matchtomatch)
                j += 1
            else:
                insert_states = []

                #handling insert states
                while True:
                    insert_states.extend([n for n in range(self.num_of_strings) if self.sequences[n][i] != '-'])
                    if i + 1 in self.match_states or i + 1 == self.num_of_chars:
                        break
                    i += 1  #next insert state

                if insert_states:
                    come_from_match = [n for n in self.t_prob[M]['strs'] if n in insert_states]
                    come_from_del = [n for n in self.t_prob.get(D, {}).get('strs', []) if n in insert_states]
                    come_from_ins = [n for n in set(insert_states) for _ in range(insert_states.count(n) - 1)]

                    # M(j) --> I(j)
                    if come_from_match:
                        self.t_prob[M][I] = {
                            'prob': float(len(come_from_match) / len(self.t_prob[M]['strs'])),
                            'strs': come_from_match
                        }

                    # D(j) --> I(j)
                    if come_from_del:
                        self.t_prob[D][I] = {
                            'prob': float(len(come_from_del) / len(self.t_prob[D]['strs'])),
                            'strs': come_from_del
                        }

                    # I(j) --> I(j)
                    if come_from_ins:
                        self.t_prob[I][I] = {
                            'prob': float(len(come_from_ins) / len(insert_states)),
                            'strs': list(set(come_from_ins))
                        }
                    self.t_prob[I]['strs'].extend(insert_states)

            #get emission probabilities without '-'
            num_of_dot = self.frequency_list[i].get('-', 0)
            self.e_prob[nextM] = {
                n: self.frequency_list[i][n] / (self.num_of_strings - num_of_dot)
                for n in self.frequency_list[i] if n != '-'
            }
            i += 1

    def calculate_viterbi(self, sequence):
        states = sorted(self.t_prob.keys())
        V = [{}]  #viterbi matrix
        path = {}

        #initialize base cases (t == 0)
        for state in states:
            #access the transition probability correctly from the nested dictionary
            transition_prob = self.t_prob['M0'].get(state, {}).get('prob', 1e-6)
            emission_prob = self.e_prob.get(state, {}).get(sequence[0], 1e-6)
            
            #probability of the first state
            V[0][state] = log(transition_prob) + log(emission_prob)
            path[state] = [state]

        #run Viterbi for t > 0
        for t in range(1, len(sequence)):
            V.append({})
            new_path = {}

            for y in states:
                #choose the best state
                (prob, state) = max(
                    (
                        V[t - 1][y0] + log(self.t_prob[y0].get(y, {}).get('prob', 1e-6)) + log(self.e_prob.get(y, {}).get(sequence[t], 1e-6)), 
                        y0
                    ) 
                    for y0 in states
                )
                V[t][y] = prob
                new_path[y] = path[state] + [y]

            path = new_path

        #find the most probable end state
        n = 0 if len(sequence) == 1 else t
        (prob, state) = max((V[n][y], y) for y in states)
        return prob, path[state]




    
    

            
    


#----------------------------------------------
#MAIN
if __name__ == '__main__':

    print('\n'+'*'*20+'\n')

    #define alphabet
    alphabet = ['A', 'T', 'C', 'G']

    synthesized_sequences = [synthesize(alphabet) for _ in range(50)]
    
    #get 15 random indexes from the synthesized sequences
    indexes = random.sample(range(50), 15)

    #datasetA will be the 15 random synthesized sequences
    datasetA = [synthesized_sequences[i] for i in indexes]

    #datasetB will be the rest of the synthesized sequences
    datasetB = [seq for i, seq in enumerate(synthesized_sequences) if i not in indexes]

    print('\n1. Created datasets A and B')

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    #save datasets to files
    try:
        with open('datasetA.txt', 'w') as f:
            for seq in datasetA:
                f.write(seq + '\n')
        with open('datasetB.txt', 'w') as f:
            for seq in datasetB:
                f.write(seq + '\n')
    except IOError:
        print("\nError writing to file")
    else:
        print('\nDatasets saved to files in directory')

    

    #Global and multiple alignment
    score_matrix=calc_pair_scores(datasetA)
    distance_matrix=cal_pair_distances(datasetA,score_matrix)
    labels= create_labels(datasetA)
    result= UPGMA(distance_matrix,labels)
    guide_tree = parse_guide_tree(result)

    final_alignment=progressive_alignment(datasetA , guide_tree)
    print('\n'*2 + '\n2.    Final Multiple Sequence Alignment: ',final_alignment)


    aligned_sequences_datasetA = align_to_final(final_alignment, datasetA)
    try:
        with open('msa_sequences.txt', 'w') as f:#put final sequences in file
            for seq,aligned_seq in aligned_sequences_datasetA.items():
                f.write(aligned_seq + '\n')
    except IOError:
        print("\nError writing MSA sequences to file")
    else:
        print('\nFinal MSA sequences saved to file')
        


    #get the msa_sequences from the file in a list
    with open('msa_sequences.txt', 'r') as f:
        msa_sequences_datasetA = [line.strip() for line in f]

    #build the hmm profile model
    print('\n'*2 + '\n3.    Building HMM Profile')
    model = profile_HMM(msa_sequences_datasetA)
    model.build_model()

    
    #calculate viterbi path and score for each sequence in datasetB
    print('\nCalculating Viterbi Path and Score for each sequence in datasetB\n')
    for sequence in datasetB:
        prob, alignment_path = model.calculate_viterbi(sequence)
        print(f"Sequence in datasetB: {sequence}")
        print(f"Alignment Viterbi Path: {alignment_path}")
        print(f"Alignment Viterbi Score: {prob}\n")

    



