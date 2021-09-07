import exceptions as pcexc
import functions as pcfun
import copy
import numpy as np
import csv
import math

class Tpath:
    """A single branch entailed to a path for list decoder"""
    #These branches represent the paths as well
    def __init__(self, N=128, m=6):
        self.N = N  # codeword length
        self.n = int(pcfun.np.log2(N))  # number of levels
        # self.llrs = [0.0 for i in range(2 * self.N - 1)]  # LLRs at butterfly
        self.llrs = pcfun.np.zeros(2 * self.N - 1)
        # self.bits = [[0 for i in range(self.N - 1)] for j in range(2)]  # bits at butterfly
        self.bits = pcfun.np.zeros((2, self.N-1), dtype=int)
        # self.decoded = [0 for i in range(self.N)]  # results of decoding
        self.decoded = pcfun.np.zeros(self.N, dtype=int)
        self.polar_decoded = pcfun.np.zeros(self.N, dtype=int)
        #self.pathmetric = 1  # probability of correct decoding
        #self.edgeOrder = 0  # Path Metric
        self.pathmetric = 0  # Path Metric
        self.forkmetric = 0  # probability for forking
        self.forkval = 0  # value for forking

        self.cur_state = [0 for i in range(m)]  # Current state
        #self.cur_state = pcfun.np.zeros((self.N,m), dtype=pcfun.np.int8) #[0 for i in range(m)]  # Current state

    def __repr__(self):
        return repr((self.llrs, self.bits, self.decoded, self.pathmetric, self.forkval, self.forkmetric))

    """def update_state(self, in_bit: int):
        if in_bit == 0:
            self.cur_state = [0] + self.cur_state[0:self.m-1] # extend (the elements), not append
        else:
            self.cur_state = [1] + self.cur_state[0:self.m-1]"""     


    def update_llrs(self, position: int):
        if position == 0:
            nextlevel = self.n
        else:
            lastlevel = (bin(position)[2:].zfill(self.n)).find('1') + 1
            start = int(pcfun.np.power(2, lastlevel - 1)) - 1
            end = int(pcfun.np.power(2, lastlevel) - 1) - 1
            for i in range(start, end + 1):
                self.llrs[i] = pcfun.lowerconv(self.bits[0][i],
                                               self.llrs[end + 2 * (i - start) + 1],
                                               self.llrs[end + 2 * (i - start) + 2])
            nextlevel = lastlevel - 1
        for lev in range(nextlevel, 0, -1):
            start = int(pcfun.np.power(2, lev - 1)) - 1
            end = int(pcfun.np.power(2, lev) - 1) - 1
            for indx in range(start, end + 1):
                exp1 = end + 2 * (indx - start)
                llr1 = self.llrs[exp1 + 1]
                llr2 = self.llrs[exp1 + 2]
                #self.llrs[indx] = pcfun.upperconv(self.llrs[exp1 + 1], self.llrs[exp1 + 2])
                #SPCparams[irs].LLR[indx] = SIGN(llr1)*SIGN(llr2)*(float)min(fabs(llr1), fabs(llr2));
                self.llrs[indx] = np.sign(llr1)*np.sign(llr2)*min(abs(llr1),abs(llr2))
                #intLLR = self.llrs[indx]
    def update_bits(self, position: int):
        N = self.N
        latestbit = self.polar_decoded[position]
        #print("d{0}".format(self.decoded[position]))
        n = self.n
        if position == N - 1:
            return
        elif position < N // 2:
            self.bits[0][0] = latestbit
        else:
            lastlevel = (bin(position)[2:].zfill(n)).find('0') + 1
            self.bits[1][0] = latestbit
            for lev in range(1, lastlevel - 1):
                st = int(pcfun.np.power(2, lev - 1)) - 1
                ed = int(pcfun.np.power(2, lev) - 1) - 1
                for i in range(st, ed + 1):
                    self.bits[1][ed + 2 * (i - st) + 1] = (self.bits[0][i] + self.bits[1][i]) % 2
                    self.bits[1][ed + 2 * (i - st) + 2] = self.bits[1][i]

            lev = lastlevel - 1
            st = int(pcfun.np.power(2, lev - 1)) - 1
            ed = int(pcfun.np.power(2, lev) - 1) - 1
            for i in range(st, ed + 1):
                self.bits[0][ed + 2 * (i - st) + 1] = (self.bits[0][i] + self.bits[1][i]) % 2
                self.bits[0][ed + 2 * (i - st) + 2] = self.bits[1][i]
        #print("s{0}".format(self.bits[0][0]))
            
    def update_pathmetric(self):
        self.pathmetric += self.forkmetric
        #self.pathmetric *= self.forkmetric




class PolarCode:
    """Represent constructing polar codes,
    encoding and decoding messages with polar codes"""

    def __init__(self, N=128, K=64, construct="dega", dSNR=0.0, L=1):
        if K >= N:
            raise pcexc.PCLengthError
        elif pcfun.np.log2(N) != int(pcfun.np.log2(N)):
            raise pcexc.PCLengthDivTwoError
        else:
            self.codeword_length = N
            self.log2_N = int(math.log2(N))
            self.information_size = K
            self.designSNR = dSNR
            self.n = int(pcfun.np.log2(self.codeword_length))
            self.bitrev_indices = [pcfun.bitreversed(j, self.n) for j in range(self.codeword_length)]
            self.polarcode_mask = pcfun.rm_build_mask(N, K, dSNR) if construct=="rm" else pcfun.pw_build_mask(N, K) if  construct=="pw" else pcfun.RAN87_build_mask(N, K, dSNR) if  construct=="ran87" else pcfun.build_mask(N, K, dSNR)
            self.rate_profile = self.polarcode_mask[self.bitrev_indices]
            self.LLRs = np.zeros(2 * self.codeword_length - 1, dtype=float)
            self.BITS = np.zeros((2, self.codeword_length - 1), dtype=int)
            self.stem_LLRs = np.zeros(2 * self.codeword_length - 1, dtype=float)
            self.stem_BITS = np.zeros((2, self.codeword_length - 1), dtype=int)
            self.list_size = L
            self.curr_list_size = 1
            self.exp_step = 0
            self.corr_path_exist = 1
            self.sc_list = list()
            self.viterbi = list()
            self.edgeOrder = [0 for k in range(L)] #np.zeros(L, dtype=int)
            self.dLLRs = [0 for k in range(L)]
            self.PMs = [0 for k in range(L)]
            self.PMR = 0
            self.trdata = np.zeros(N, dtype=int)
            self.corr_pos = np.zeros((N,L), dtype=int)
            
            self.m = 0
            self.gen = []
            self.cur_state = [] #np.zeros(self.m, dtype=int)#
            self.num_paths = L
            self.curr_num_paths = 1
            self.path_select = 1
            #list([iterbale]) is the list constructor
            self.modu = 'BPSK'
            

            self.A = pcfun.A(self.polarcode_mask, N, K)
            self.sigma = 0
            self.snrb_snr = 'SNRb'
            self.iter = 0
            self.iterations = 0
            

    def __repr__(self):
        return repr((self.codeword_length, self.information_size, self.designSNR))
#__str__ (read as "dunder (double-underscore) string") and __repr__ (read as "dunder-repper" (for "representation")) are both special methods that return strings based on the state of the object.
    def mul_matrix(self, precoded):
        """multiplies message of length N with generator matrix G"""
        """Multiplication is based on factor graph"""
        N = self.codeword_length
        polarcoded = precoded
        for i in range(self.n):
            if i == 0:
                polarcoded[0:N:2] = (polarcoded[0:N:2] + polarcoded[1:N:2]) % 2
            elif i == (self.n - 1):
                polarcoded[0:int(N/2)] = (polarcoded[0:int(N/2)] + polarcoded[int(N/2):N]) % 2
            else:
                enc_step = int(pcfun.np.power(2, i))
                for j in range(enc_step):
                    polarcoded[j:N:(2 * enc_step)] = (polarcoded[j:N:(2 * enc_step)]
                                                    + polarcoded[j + pcfun.np.power(2, i):N:(2 * enc_step)]) % 2
        return polarcoded
    # --------------- ENCODING -----------------------

    def precode(self, info):
        """Apply polar code mask to information message and return precoded message"""
        precoded = pcfun.np.zeros(self.codeword_length, dtype=int) #array
        precoded[self.polarcode_mask == 1] = info
        self.trdata = copy.deepcopy(precoded)
        return precoded
    
    def pac_encode(self, info, conv_gen, mem):
        """Encoding function"""
        # Non-systematic encoding
        V = self.precode(info)
        U = pcfun.conv_encode(V, conv_gen, mem)
        X = self.mul_matrix(U)
        return X

    # -------------------------- DECODING -----------------------------------

    def extract(self, decoded_message):
        """Extracts bits from information positions due to polar code mask"""
        decoded_info = pcfun.np.array(list(), dtype=int)
        mask = self.polarcode_mask
        for i in range(len(self.polarcode_mask)):
            if mask[i] == 1:
                # decoded_info.append(decoded_message[i])
                decoded_info = pcfun.np.append(decoded_info, decoded_message[i])
        return decoded_info




    # --- Viterbi Decoding ---------------------------------------------------------------------------------
    # Number of states = number of current paths
    def trellis_fork(self, trellisPath, pos):
        # Take note of difference b/w self.trellisPath and trellisPath 
        """forks current stage of Trellis
        and makes decisions on decoded values based on llr values"""
        pos_rev = pcfun.bitreversed(pos,self.log2_N)
        edgeValue = [0 for i in range(2*self.curr_num_paths)]   #encoded by CE
        msgValue = [0 for i in range(2*self.curr_num_paths)]    #Msg bit
        pathMetric = [0.0 for i in range(2*self.curr_num_paths)]
        pathState = [[] for i in range(2*self.curr_num_paths)]
        pathStateMap =  [[] for i in range(int(self.num_paths/self.path_select))]
        #activeStates = [0 for i in range(self.curr_num_paths)]
        
        for i in range(self.curr_num_paths): #Every path is split into two by considering two options for the concatenated bit; 0, 1
            i2 = i+self.curr_num_paths
            #curr_state = trellisPath[i].cur_state
            if trellisPath[i].llrs[0] > 0:
                edgeValue[i] = pcfun.conv_1bit(0, trellisPath[i].cur_state, self.gen)
                edgeValue[i2] = 1 - edgeValue[i]
                pathMetric[i] = trellisPath[i].pathmetric + (0 if edgeValue[i]==0 else 1) * np.abs(trellisPath[i].llrs[0])
                pathMetric[i2] = trellisPath[i].pathmetric + (0 if edgeValue[i2]==0 else 1) * np.abs(trellisPath[i].llrs[0])
                if pathMetric[i2] > pathMetric[i]:
                    msgValue[i] = 0
                    msgValue[i2] = 1
                    pathState[i] = pcfun.getNextState(0, trellisPath[i].cur_state, self.m)
                    pathState[i2] = pcfun.getNextState(1, trellisPath[i].cur_state, self.m)
                else:
                    edgeValue[i] = 1 - edgeValue[i]
                    edgeValue[i2] = 1 - edgeValue[i2]
                    tempPM = pathMetric[i]
                    pathMetric[i] = pathMetric[i2]
                    pathMetric[i2] = tempPM
                    msgValue[i] = 1
                    msgValue[i2] = 0
                    pathState[i] = pcfun.getNextState(1, trellisPath[i].cur_state, self.m)
                    pathState[i2] = pcfun.getNextState(0, trellisPath[i].cur_state, self.m)
                    
            else:
            #elif trellisPath[i].llrs[0] < 0:
                edgeValue[i] = pcfun.conv_1bit(1, trellisPath[i].cur_state, self.gen)
                edgeValue[i2] = 1 - edgeValue[i]
                pathMetric[i] = trellisPath[i].pathmetric + (0 if edgeValue[i]==1 else 1) * np.abs(trellisPath[i].llrs[0])
                pathMetric[i2] = trellisPath[i].pathmetric + (0 if edgeValue[i2]==1 else 1) * np.abs(trellisPath[i].llrs[0])
                if pathMetric[i2] > pathMetric[i]:  #to avoid deepcopy in SC state (not helpful in deletion or duplicate states)
                    msgValue[i] = 1
                    msgValue[i2] = 0
                    pathState[i] = pcfun.getNextState(1, trellisPath[i].cur_state, self.m)
                    pathState[i2] = pcfun.getNextState(0, trellisPath[i].cur_state, self.m)
                else:
                    edgeValue[i] = 1 - edgeValue[i]
                    edgeValue[i2] = 1 - edgeValue[i2]
                    tempPM = pathMetric[i]
                    pathMetric[i] = pathMetric[i2]
                    pathMetric[i2] = tempPM
                    msgValue[i] = 0
                    msgValue[i2] = 1
                    pathState[i] = pcfun.getNextState(0, trellisPath[i].cur_state, self.m)
                    pathState[i2] = pcfun.getNextState(1, trellisPath[i].cur_state, self.m)
            # The paths connected to each state :
            pathStateMap[pcfun.bin2dec(pathState[i])].append(i)
            pathStateMap[pcfun.bin2dec(pathState[i2])].append(i2)
            #activeStates[pcfun.bin2dec(pathState[i])] = 1
            #activeStates[pcfun.bin2dec(pathState[i2])] = 1
        ##PM_sorted_idx = np.argsort(pathMetric)
        if 2*self.curr_num_paths <= self.num_paths:    #Has the paths been expanded to the width of the trellis?
            #self.edgeOrder = PM_sorted_idx[:2*self.curr_num_paths]
            for i in range(self.curr_num_paths):
                i2 = i+self.curr_num_paths
                copy_path = Tpath(self.codeword_length)
                #If we don't use deepcopy, the new object will refer to the original one and works as a pointer
                copy_path = copy.deepcopy(trellisPath[i])
                self.trellisPath[i].pathmetric = pathMetric[i]
                self.trellisPath[i].decoded[pos] = msgValue[i]
                self.trellisPath[i].polar_decoded[pos] = edgeValue[i]
                self.trellisPath[i].cur_state = pathState[i]
                copy_path.pathmetric = pathMetric[i2]
                copy_path.decoded[pos] = msgValue[i2]
                copy_path.polar_decoded[pos] = edgeValue[i2]
                copy_path.cur_state = pathState[i2]
                self.trellisPath.append(copy_path)
        else:
            ##self.edgeOrder = PM_sorted_idx[:self.curr_num_paths]
            #Recognizing inactive paths:
            discarded_paths = np.zeros(2*self.curr_num_paths, dtype=np.int8)
            survived_paths = np.zeros(2*self.curr_num_paths, dtype=np.int8)
            duplicated_paths = np.zeros(self.curr_num_paths, dtype=np.int8)
            swapping_paths = np.zeros(self.curr_num_paths, dtype=np.int8)
            retaining_paths = np.zeros(self.curr_num_paths, dtype=np.int8)
            survivors = []
            deleted_paths = []
            num_states = int(self.curr_num_paths/self.path_select)
            """#Begin: For v2 of LVA:
            bM_L = [] #[0.0 for i in range(num_states)]
            stateMap_prune_L = [0 for i in range(num_states)]
            for i in range(num_states): # In order of curr_states, based on radix-2
                num_branches = len(pathStateMap[i])
                if num_branches > 0:
                    branchMetrics =  np.zeros(num_branches, dtype=float)
                    for k in range(num_branches):
                        branchMetrics[k] = pathMetric[pathStateMap[i][k]]
                    bM_sorted_idx = np.argsort(branchMetrics)
                    #adding code for selecting Lmax and then metric of Lmax for all the states and sorting them. Then tagging for discarding or survuving.
                    #bM_L[i] = branchMetrics[bM_sorted_idx[self.path_select-1]]  #Selecting  the L-th metric
                    bM_L.append(branchMetrics[bM_sorted_idx[self.path_select-1]])  #Selecting  the L-th metric
            if len(bM_L) == num_states:
                State_sorted_idx = np.argsort(bM_L) ##Error: Sorting when len(bM_L) < num_states, then index i in line 754
                for i in range(int(num_states/2),num_states):
                    stateMap_prune_L[State_sorted_idx[i]] = 1
            #End: For v2 of LVA"""
            
            # Tagging paths to discard :
            for i in range(num_states): # In order of curr_states, based on radix-2
                num_branches = len(pathStateMap[i])
                prune_start_idx = int(num_branches/2)
                if num_branches > 0: #After frozen bits, we have less states, i.e. num_branches =0 at some states
                    branchMetrics =  np.zeros(num_branches, dtype=float)
                    for k in range(num_branches):
                        branchMetrics[k] = pathMetric[pathStateMap[i][k]]
                    bM_sorted_idx = np.argsort(branchMetrics)
                    """#Begin: For v2 of LVA:
                    if num_branches == self.path_select*2:
                        if stateMap_prune_L[i] == 1:
                            prune_start_idx = int(num_branches/2) - 1
                        else:
                            prune_start_idx = int(num_branches/2) + 1
                    #End: For v2 of LVA"""
                    for k in range(prune_start_idx,num_branches):
                        if pathStateMap[i][bM_sorted_idx[k]] < self.curr_num_paths: #if ? > self.curr_num_paths-1, it will be discarded anyway and we don't need to copy another path to its place.
                            discarded_paths[pathStateMap[i][bM_sorted_idx[k]]] = 1
                    for k in range(prune_start_idx):
                        if pathStateMap[i][bM_sorted_idx[k]] > self.curr_num_paths-1:
                            survived_paths[pathStateMap[i][bM_sorted_idx[k]]] = 1
                            survivors.append(pathStateMap[i][bM_sorted_idx[k]])
                """   
                if pathMetric[pathStateMap[i][0]] < pathMetric[pathStateMap[i][1]]:
                    if pathStateMap[i][1] < self.curr_num_paths:
                        discarded_paths[pathStateMap[i][1]] = 1
                    if pathStateMap[i][0] > self.curr_num_paths:
                        survived_paths[pathStateMap[i][0]] = 1
                        survivors.append(pathStateMap[i][0])
                else:
                    if pathStateMap[i][0] < self.curr_num_paths:
                        discarded_paths[pathStateMap[i][0]] = 1
                    if pathStateMap[i][1] > self.curr_num_paths:
                        survived_paths[pathStateMap[i][1]] = 1
                        survivors.append(pathStateMap[i][1])
                """
            for i in range(self.curr_num_paths): # In order of the paths stored in the memory
                if  discarded_paths[i] == 1 and survived_paths[i+self.curr_num_paths] == 1:
                    swapping_paths[i] = 1
                    discarded_paths[i] = 0
                elif discarded_paths[i] == 1 and survived_paths[i+self.curr_num_paths] == 0:
                    deleted_paths.append(i) 
                elif discarded_paths[i] == 0 and survived_paths[i+self.curr_num_paths] == 1:
                    duplicated_paths[i] = 1 #1; duplicated, 0: deleted 
                elif  discarded_paths[i] == 0 and survived_paths[i+self.curr_num_paths] == 0:
                    retaining_paths[i] = 1
                    
            #k = 0
            for i in range(self.curr_num_paths): # In order of the paths stored in the memory
                if  swapping_paths[i] == 1: # Swapping the i-th path with i2-th path
                    self.trellisPath[i].decoded[pos] = msgValue[i+self.curr_num_paths]
                    self.trellisPath[i].polar_decoded[pos] = edgeValue[i+self.curr_num_paths]
                    self.trellisPath[i].cur_state = pathState[i+self.curr_num_paths]
                    self.trellisPath[i].pathmetric = pathMetric[i+self.curr_num_paths]
                    survivors.remove(i+self.curr_num_paths)
                elif  retaining_paths[i] == 1: #the i-th path retained, not the i2-th path
                    self.trellisPath[i].decoded[pos] = msgValue[i]
                    self.trellisPath[i].polar_decoded[pos] = edgeValue[i]
                    self.trellisPath[i].cur_state = pathState[i]
                    self.trellisPath[i].pathmetric = pathMetric[i]
                elif  duplicated_paths[i] == 1: #Issue: when duplicating, if there is no deleted path? Can't be
                    self.trellisPath[i].decoded[pos] = msgValue[i]
                    self.trellisPath[i].polar_decoded[pos] = edgeValue[i]
                    self.trellisPath[i].cur_state = pathState[i]
                    self.trellisPath[i].pathmetric = pathMetric[i]
                    self.trellisPath[deleted_paths[0]] = copy.deepcopy(self.trellisPath[i]) #in v2, the index of deleted_paths[0] might be > self.curr_num_paths
                    self.trellisPath[deleted_paths[0]].decoded[pos] = msgValue[i+self.curr_num_paths]
                    self.trellisPath[deleted_paths[0]].polar_decoded[pos] = edgeValue[i+self.curr_num_paths]
                    self.trellisPath[deleted_paths[0]].cur_state = pathState[i+self.curr_num_paths]
                    self.trellisPath[deleted_paths[0]].pathmetric = pathMetric[i+self.curr_num_paths]
                    deleted_paths.remove(deleted_paths[0])
                    survivors.remove(i+self.curr_num_paths)
                    #k += 1
            #self.trdata[pcfun.bitreversed(ij, log_N)]
            #l.decoded[pcfun.bitreversed(ij, log_N)]
            stop_point = 1
        self.edgeOrder = np.argsort(pathMetric)#[:self.curr_num_paths]



    def pac_viterbi_decoder(self, soft_mess, issystematic=False):
        """Successive cancellation list decoder"""
        # init list of decoding branches
        codeword_length = len(self.polarcode_mask)
        log_N = int(math.log2(codeword_length))
        #self.num_paths = 2**self.m
        #print("N={0}".format(codeword_length))
        self.trellisPath = [Tpath(codeword_length,self.m)]    #Branch is equivalent to one edge of the paths at each step on the binary tree, whihch carries intermediate LLRs, Partial sums, prob
        #print("L={0}".format(len(self.viterbi)))
        # initial/channel LLRs
        self.trellisPath[0].llrs[codeword_length - 1:] = soft_mess
        #N-1 out of @N-1 are reserved for intermediate LLRs
        #print(self.trellisPath[0].llrs)
        #("Mask:")
        #print(self.polarcode_mask)
        
        elim_recorded = 0
        decoding_failed = False
        #elim_not_indicated = True
        #curr_i_err = self.MHWlastBit
        # MHW:
        """MHWcricBitCntr = 0
        cricBits = [0 for i in range(35)]
        for j in range(codeword_length):
            MHWOnesCnt = pcfun.countOnes(j)
            if MHWOnesCnt == self.MHWexpWt:
                
                cricBits[MHWcricBitCntr] = j
                MHWcricBitCntr += 1
        self.MHWlastBit = cricBits[self.MHWbitCntr]"""
        
        for j in range(codeword_length):
                
            corr_path_not_found = 0
            i = pcfun.bitreversed(j, self.n)
            self.curr_num_paths = len(self.trellisPath)
            #print("i={0}".format(i))
            for l in self.trellisPath:
                l.update_llrs(i)    #Update intermediate LLRs

            #if self.polarcode_mask[i] == 1 and j > self.MHWlastBit:
            if self.polarcode_mask[i] == 1:
                self.trellis_fork(self.trellisPath, i)
                """
                self.fork(self.trellisPath)
                self.branch_and_reduce(i)
                """
                #print("{0} {1:.2f} {2:.3f} {3:.3f} {4:.3f} {5:.3f}".format(j,self.trellisPath[len(self.trellisPath)-1].pathmetric-self.trellisPath[0].pathmetric, self.trellisPath[0].pathmetric, self.trellisPath[1].pathmetric, self.trellisPath[2].pathmetric, self.trellisPath[3].pathmetric))
                #print("{0} {1:.2f}".format(j,self.trellisPath[len(self.trellisPath)-1].pathmetric-self.trellisPath[0].pathmetric))
                #if (j>106 and j<110):
                    #print("{0:.3f} {1:.3f} {2:.3f} {3:.3f}".format(self.trellisPath[0].pathmetric, self.trellisPath[1].pathmetric, self.trellisPath[2].pathmetric, self.trellisPath[3].pathmetric))
            #elif self.polarcode_mask[i] == 1 and j == self.MHWlastBit:
                # MHW:
                """
                print(j)
                self.MHWbitCntr += 1
                for l in self.trellisPath:
                    edgeValue0 = pcfun.conv_1bit(1, l.cur_state, self.gen)
                    l.cur_state = pcfun.getNextState(1, l.cur_state, self.m)
                    cur_state0 = l.cur_state
                    l.decoded[i] = self.polarcode_mask[i]
                    l.polar_decoded[i] = edgeValue0
                    dLLR = l.llrs[0]
                    penalty = np.abs(l.llrs[0])
                    if l.llrs[0] < 0:
                        pathMetric0 = l.pathmetric + (0 if edgeValue0==1 else 1) * penalty
                    else:
                        pathMetric0 = l.pathmetric + (0 if edgeValue0==0 else 1) * penalty
                    l.pathmetric = pathMetric0
                """
            else:
                for l in self.trellisPath:
                    edgeValue0 = pcfun.conv_1bit(0, l.cur_state, self.gen)
                    l.cur_state = pcfun.getNextState(0, l.cur_state, self.m)
                    cur_state0 = l.cur_state
                    l.decoded[i] = self.polarcode_mask[i]
                    l.polar_decoded[i] = edgeValue0
                    #dLLR = l.llrs[0]
                    penalty = np.abs(l.llrs[0])
                    if l.llrs[0] < 0:
                        pathMetric0 = l.pathmetric + (0 if edgeValue0==1 else 1) * penalty
                    else:
                        pathMetric0 = l.pathmetric + (0 if edgeValue0==0 else 1) * penalty
                    l.pathmetric = pathMetric0
            ii=0    #counter for list elements
            #For some reason, len(self.trellisPath) != self.curr_num_paths
            #pm = [0 for i in range(len(self.trellisPath))]
            for l in self.trellisPath:
                l.update_bits(i)
                
                #Tracking the correct path
                """
                #pm[ii] = l.pathmetric
                if self.curr_num_paths > 1:
                    jj=0 #counter for decoded bits
                    for ij in range(j+1):
                        if self.trdata[pcfun.bitreversed(ij, log_N)]==l.decoded[pcfun.bitreversed(ij, log_N)]:
                            jj+=1
                        else:
                            break
                            #print("path{0} at b{1} not corr".format(ii,pcfun.bitreversed(pos, 9)))
                    if jj-1==j:# and ii>1:
                        #corrPath_rows[j, np.where(self.edgeOrder == ii)] = 1   #self.edgeOrder.index(ii-1)
                        for v in range(self.curr_num_paths):
                            if self.edgeOrder[v] == ii:
                                idx = v
                                break
                        self.corr_pos[j][idx] += 1
                        #print("Path {0} @ bit {1} is correct. PMc={2:.2f}".format(idx,j, l.pathmetric))

                    else:
                        corr_path_not_found += 1
                #if corr_path_not_found==self.curr_num_paths:
                    #decoding_failed = True
                    #print("corr_path eliminated at bit {0}".format(j))
                    #print("corr_path does not exist")
                ii+=1  #Loop counter
            #self.pmr_accum[j] += max(pm)-min(pm) #if min(pm)!=0 else min(pm)
            #print(max(pm),min(pm),self.pmr_accum[j],self.curr_num_paths)
            if (corr_path_not_found==self.curr_num_paths and decoding_failed == False):
                if elim_recorded == 0: #and self.repeat==False:
                    elim_recorded = 1
                    self.elim_freq[j] += 1
                    decoding_failed = True
                    print("corr_path eliminated at bit {0}".format(j))
            """
        #if decoding_failed == False:
            #print("corr_path exists")

        self.trellisPath.sort(key=lambda trellis_path: trellis_path.pathmetric, reverse=False)
        """
        #MHW: Calc Hamming weight
        weight0 = np.zeros(self.num_paths, dtype=int)
        weight = np.zeros(self.num_paths, dtype=int)
        cntr = 0
        for l in self.trellisPath:
            weight0[cntr] = np.sum(l.decoded)
            U = pcfun.conv_encode(l.decoded, self.gen, self.m)
            X = self.mul_matrix(U)
            weight[cntr] = np.sum(X)
            #print(weight[cntr],end=" ")
            cntr += 1
        #unique_elements, counts_elements = np.unique(weight0, return_counts=True)
        #print("Frequency of unique weights:")
        #print(np.asarray((unique_elements, counts_elements)))
        unique_elements, counts_elements = np.unique(weight, return_counts=True)
        print("Frequency of unique weights:")
        print(np.asarray((unique_elements, counts_elements)))
        """
        best = self.trellisPath[0].decoded
        #print(" ")
        #if issystematic:
            #self.mul_matrix(best)
        return self.extract(best)




    def pac_viterbi_crc_decoder(self, soft_mess, issystematic=False, iscrc=False, crc8_table=None):
        """Successive cancellation list decoder"""
        # init list of decoding branches
        codeword_length = len(self.polarcode_mask)
        log_N = int(math.log2(codeword_length))
        #self.num_paths = 2**self.m
        #print("N={0}".format(codeword_length))
        self.trellisPath = [Tpath(codeword_length,self.m)]    #Branch is equivalent to one edge of the paths at each step on the binary tree, whihch carries intermediate LLRs, Partial sums, prob
        #print("L={0}".format(len(self.viterbi)))
        # initial/channel LLRs
        self.trellisPath[0].llrs[codeword_length - 1:] = soft_mess
        #N-1 out of @N-1 are reserved for intermediate LLRs
        #print(self.trellisPath[0].llrs)
        #("Mask:")
        #print(self.polarcode_mask)
        
        elim_recorded = 0
        decoding_failed = False
        #elim_not_indicated = True
        #curr_i_err = self.MHWlastBit
        # MHW:
        """MHWcricBitCntr = 0
        cricBits = [0 for i in range(35)]
        for j in range(codeword_length):
            MHWOnesCnt = pcfun.countOnes(j)
            if MHWOnesCnt == self.MHWexpWt:
                
                cricBits[MHWcricBitCntr] = j
                MHWcricBitCntr += 1
        self.MHWlastBit = cricBits[self.MHWbitCntr]"""
        
        for j in range(codeword_length):
                
            corr_path_not_found = 0
            i = pcfun.bitreversed(j, self.n)
            self.curr_num_paths = len(self.trellisPath)
            #print("i={0}".format(i))
            for l in self.trellisPath:
                l.update_llrs(i)    #Update intermediate LLRs

            #if self.polarcode_mask[i] == 1 and j > self.MHWlastBit:
            if self.polarcode_mask[i] == 1:
                self.trellis_fork(self.trellisPath, i)
                """
                self.fork(self.trellisPath)
                self.branch_and_reduce(i)
                """
                #print("{0} {1:.2f} {2:.3f} {3:.3f} {4:.3f} {5:.3f}".format(j,self.trellisPath[len(self.trellisPath)-1].pathmetric-self.trellisPath[0].pathmetric, self.trellisPath[0].pathmetric, self.trellisPath[1].pathmetric, self.trellisPath[2].pathmetric, self.trellisPath[3].pathmetric))
                #print("{0} {1:.2f}".format(j,self.trellisPath[len(self.trellisPath)-1].pathmetric-self.trellisPath[0].pathmetric))
                #if (j>106 and j<110):
                    #print("{0:.3f} {1:.3f} {2:.3f} {3:.3f}".format(self.trellisPath[0].pathmetric, self.trellisPath[1].pathmetric, self.trellisPath[2].pathmetric, self.trellisPath[3].pathmetric))
            #elif self.polarcode_mask[i] == 1 and j == self.MHWlastBit:
                # MHW:
                """
                print(j)
                self.MHWbitCntr += 1
                for l in self.trellisPath:
                    edgeValue0 = pcfun.conv_1bit(1, l.cur_state, self.gen)
                    l.cur_state = pcfun.getNextState(1, l.cur_state, self.m)
                    cur_state0 = l.cur_state
                    l.decoded[i] = self.polarcode_mask[i]
                    l.polar_decoded[i] = edgeValue0
                    dLLR = l.llrs[0]
                    penalty = np.abs(l.llrs[0])
                    if l.llrs[0] < 0:
                        pathMetric0 = l.pathmetric + (0 if edgeValue0==1 else 1) * penalty
                    else:
                        pathMetric0 = l.pathmetric + (0 if edgeValue0==0 else 1) * penalty
                    l.pathmetric = pathMetric0
                """
            else:
                for l in self.trellisPath:
                    edgeValue0 = pcfun.conv_1bit(0, l.cur_state, self.gen)
                    l.cur_state = pcfun.getNextState(0, l.cur_state, self.m)
                    cur_state0 = l.cur_state
                    l.decoded[i] = self.polarcode_mask[i]
                    l.polar_decoded[i] = edgeValue0
                    #dLLR = l.llrs[0]
                    penalty = np.abs(l.llrs[0])
                    if l.llrs[0] < 0:
                        pathMetric0 = l.pathmetric + (0 if edgeValue0==1 else 1) * penalty
                    else:
                        pathMetric0 = l.pathmetric + (0 if edgeValue0==0 else 1) * penalty
                    l.pathmetric = pathMetric0
            ii=0    #counter for list elements
            #For some reason, len(self.trellisPath) != self.curr_num_paths
            #pm = [0 for i in range(len(self.trellisPath))]
            for l in self.trellisPath:
                l.update_bits(i)
                
                #Tracking the correct path
                """
                #pm[ii] = l.pathmetric
                if self.curr_num_paths > 1:
                    jj=0 #counter for decoded bits
                    for ij in range(j+1):
                        if self.trdata[pcfun.bitreversed(ij, log_N)]==l.decoded[pcfun.bitreversed(ij, log_N)]:
                            jj+=1
                        else:
                            break
                            #print("path{0} at b{1} not corr".format(ii,pcfun.bitreversed(pos, 9)))
                    if jj-1==j:# and ii>1:
                        #corrPath_rows[j, np.where(self.edgeOrder == ii)] = 1   #self.edgeOrder.index(ii-1)
                        for v in range(self.curr_num_paths):
                            if self.edgeOrder[v] == ii:
                                idx = v
                                break
                        self.corr_pos[j][idx] += 1
                        #print("Path {0} @ bit {1} is correct. PMc={2:.2f}".format(idx,j, l.pathmetric))

                    else:
                        corr_path_not_found += 1
                #if corr_path_not_found==self.curr_num_paths:
                    #decoding_failed = True
                    #print("corr_path eliminated at bit {0}".format(j))
                    #print("corr_path does not exist")
                ii+=1  #Loop counter
            #self.pmr_accum[j] += max(pm)-min(pm) #if min(pm)!=0 else min(pm)
            #print(max(pm),min(pm),self.pmr_accum[j],self.curr_num_paths)
            if (corr_path_not_found==self.curr_num_paths and decoding_failed == False):
                if elim_recorded == 0: #and self.repeat==False:
                    elim_recorded = 1
                    self.elim_freq[j] += 1
                    decoding_failed = True
                    print("corr_path eliminated at bit {0}".format(j))
            """
        #if decoding_failed == False:
            #print("corr_path exists")

        #self.trellisPath.sort(key=lambda trellis_path: trellis_path.pathmetric, reverse=False)


        if iscrc:
            self.trellisPath.sort(key=lambda branch: branch.pathmetric, reverse=False) #for prob-based: reverse=True #key: a function to specify the sorting criteria(s), reverse=True : in descending order
            if issystematic:
                self.mul_matrix(self.trellisPath[0].decoded)
            best = self.extract(self.trellisPath[0].decoded)
            self.iter += 1
            if pcfun.np.sum(pcfun.crc8_table_method(best, crc8_table)) == 0:
                #print("crcPath=1")
                self.repeat_no = -1
                self.shft_idx = 0
                return best[0:len(best)-8]
            else:
                idx=2
                for br in self.trellisPath[1:]:
                    if issystematic:
                        self.mul_matrix(br.decoded)
                    rx = self.extract(br.decoded)
                    if pcfun.np.sum(pcfun.crc8_table_method(rx, crc8_table)) == 0:
                        print("*****************************crcPath={0}".format(idx))
                        self.repeat_no = -1
                        self.shft_idx = 0
                        return rx[0:len(rx)-8]
                    idx+=1
            return best[0:len(best) - 8]


        else:
            self.trellisPath.sort(key=lambda branch: branch.pathmetric, reverse=False)
            best = self.trellisPath[0].decoded
            if issystematic:
                self.mul_matrix(best)
            return self.extract(best)








