from operator import itemgetter
#itemgetter(item) return a callable object that fetches item from its operand using the operandâ€™s __getitem__() method. If multiple items are specified, returns a tuple of lookup values
import numpy as np
import math
from scipy.stats import norm



def fails(list1, list2):
    """returns number of bit errors"""
    return np.sum(np.absolute(list1 - list2))


def bitreversed(num: int, n) -> int:
    """"""
    return int(''.join(reversed(bin(num)[2:].zfill(n))), 2)
#numpy.core.defchararray.zfill(a, width) [source]
#Return the numeric string left-filled with zeros
#int(num,base) method returns an integer object from any number or string.
#The join() method takes all items in an iterable and joins them into one string.



# ------------ building polar code mask -----------------


def bhattacharyya_count(N: int, design_snr: float):
    # bhattacharya_param = [0.0 for i in range(N)]
    bhattacharya_param = np.zeros(N, dtype=float)
    # snr = pow(10, design_snr / 10)
    snr = np.power(10, design_snr / 10)
    bhattacharya_param[0] = np.exp(-snr)
    for level in range(1, int(np.log2(N)) + 1):
        B = np.power(2, level)
        for j in range(int(B / 2)):
            T = bhattacharya_param[j]
            bhattacharya_param[j] = 2 * T - np.power(T, 2)
            bhattacharya_param[int(B / 2 + j)] = np.power(T, 2)
    return bhattacharya_param


def phi_inv(x: float):
    if (x>12):
        return 0.9861 * x - 2.3152
    elif (x<=12 and x>3.5):
        return x*(0.009005 * x + 0.7694) - 0.9507
    elif (x<=3.5 and x>1):
        return x*(0.062883*x + 0.3678)- 0.1627
    else:
        return x*(0.2202*x + 0.06448)


def dega_construct(N: int, K: int, dsnr_db: float):
    # bhattacharya_param = [0.0 for i in range(N)]
    mllr = np.zeros(N, dtype=float)
    # snr = pow(10, design_snr / 10)
    #dsnr = np.power(10, dsnr_db / 10)
    sigma_sq = 1/(2*K/N*np.power(10,dsnr_db/10))
    mllr[0] = 2/sigma_sq
    #mllr[0] = 4 * K/N * dsnr
    for level in range(1, int(np.log2(N)) + 1):
        B = np.power(2, level)
        for j in range(int(B / 2)):
            T = mllr[j]
            mllr[j] = phi_inv(T)
            mllr[int(B / 2 + j)] = 2 * T
    return mllr

def pe_dega(N: int, K: int, dsnr_db: float):
    # bhattacharya_param = [0.0 for i in range(N)]
    mllr = np.zeros(N, dtype=float)
    pe = np.zeros(N, dtype=float)
    # snr = pow(10, design_snr / 10)
    #dsnr = np.power(10, dsnr_db / 10)
    sigma = np.sqrt(1/(2*K/N*np.power(10,dsnr_db/10)))
    mllr[0] = 2/np.square(sigma)
    #mllr[0] = 4 * K/N * dsnr
    for level in range(1, int(np.log2(N)) + 1):
        B = np.power(2, level)
        for j in range(int(B / 2)):
            T = mllr[j]
            mllr[j] = phi_inv(T)
            mllr[int(B / 2 + j)] = 2 * T
    #mean = 2/np.square(sigma)
    #var = 4/np.square(sigma)
    for ii in range(N):
        #z = (mllr - mean)/np.sqrt(var)
        #pe[ii] = 1/(np.exp(mllr[ii])+1)
        #pe[ii] = 1 - norm.cdf( np.sqrt(mllr[ii]/2) )
        pe[ii] = 0.5 - 0.5 * math.erf( np.sqrt(mllr[ii])/2 )
    return pe

def A(mask, N, K):
    j = 0
    A_set = np.zeros(K, dtype=int)
    for ii in range(N):
        if mask[ii] == 1:
            A_set[j] = bitreversed(ii, int(math.log2(N)))
            j += 1
    A_set = np.sort(A_set)
    return A_set

def countOnes(num:int):
    ones = 0
    binary = bin(num)[2:]
    len_bin = len(binary)
    for i in range(len_bin):
        if binary[i]=='1':
            ones += 1
    return(ones)

def pw_construct(N: int, K: int, dsnr_db: float):
    w = np.zeros(N, dtype=float)
    n = int(np.log2(N))
    for i in range(N):
        wi = 0
        binary = bin(i)[2:].zfill(n)
        for j in range(n):
            wi += int(binary[j])*pow(2,(j*0.25))
        w[i] = wi
    return w


def G_rows_wt(N: int, K: int):
    w = np.zeros(N, dtype=int)
    for i in range(N):
        w[i] = countOnes(i)
    return w
    

def build_mask(N: int, K: int, design_snr=0):
    """Generates mask of polar code
    in mask 0 means frozen bit, 1 means information bit"""
    # each bit has 3 attributes
    # [order, bhattacharyya value, frozen / imformation position]
    # 0 - frozen, 1 - information
    mask = [[i, 0.0, 1] for i in range(N)]
    # Build mask using Bhattacharya values
    #values = G_rows_wt(N, K)
    values = dega_construct(N, K, design_snr)
    #values = bhattacharyya_count(N, design_snr)
    # set bhattacharyya values
    for i in range(N):
        mask[i][1] = values[i]
    # sort channels due to bhattacharyya values
    mask = sorted(mask, key=itemgetter(1), reverse=False)   #DEGA, RM
    #mask = sorted(mask, key=itemgetter(1), reverse=True)    #bhattacharyya
    # set mask[i][2] in 1 for channels with K lowest bhattacharyya values
    for i in range(N-K):
        mask[i][2] = 0
    # sort channels due to order
    mask = sorted(mask, key=itemgetter(0))
    # return positions bits
    return np.array([i[2] for i in mask])

def rm_build_mask(N: int, K: int, design_snr=0):
    """Generates mask of polar code
    in mask 0 means frozen bit, 1 means information bit"""
    # each bit has 3 attributes
    # [order, bhattacharyya value, frozen / imformation position]
    # 0 - frozen, 1 - information
    mask = [[i, 0, 0.0, 1] for i in range(N)]
    # Build mask using Bhattacharya values
    values = G_rows_wt(N, K) # row_wt(i)=2**(wt(bin(i)), value=wt(bin(i))
    values2 = dega_construct(N, K, design_snr)
    #values = bhattacharyya_count(N, design_snr)
    #Bit Error Prob.
    # set bhattacharyya values
    for i in range(N):
        mask[i][1] = values[i]
        mask[i][2] = values2[i]
    # Sort the channels by Bhattacharyya values
    weightCount = np.zeros(int(math.log2(N))+1, dtype=int)
    for i in range(N):
        weightCount[values[i]] += 1
    bitCnt = 0
    k = 0
    while bitCnt + weightCount[k] <= N-K:
        for i in range(N):
            if values[i]==k:
                mask[i][3] = 0
                bitCnt += 1
        k += 1
    mask2 = []
    for i in range(N):
        if mask[i][1] == k:
            mask2.append(mask[i])
    mask2 = sorted(mask2, key=itemgetter(2), reverse=False)   #DEGA
    remainder = (N-K)-bitCnt
    available = weightCount[k]
    for i in range(remainder):
        mask[mask2[i][0]][3] = 0

    rate_profile = np.array([i[3] for i in mask])
    #mask = sorted(mask, key=itemgetter(0))  #sort based on bit-index
    # return positions bits
    #Modify the profile:
    """
    toFreeze = [21]
    toUnfreeze = [18]
    n = int(math.log2(N))
    for i in range(len(toFreeze)):
        #rate_profile[bitreversed(toFreeze[i], n)] = 0
        #rate_profile[bitreversed(toUnfreeze[i], n)] = 1
        rate_profile[toFreeze[i]] = 0
        rate_profile[toUnfreeze[i]] = 1
    """    
    
    return rate_profile

         
    
# ------------ SC decoding functions -----------------


    
def lowerconv(upperdecision: int, upperllr: float, lowerllr: float) -> float:
    """PERFORMS IN LOG DOMAIN
    llr = lowerllr * upperllr - - if uppperdecision == 0
    llr = lowerllr / upperllr - - if uppperdecision == 1
    """
    if upperdecision == 0:
        return lowerllr + upperllr
    else:
        return lowerllr - upperllr


def logdomain_sum(x: float, y: float) -> float:
    if x < y:
        return y + np.log(1 + np.exp(x - y))
    else:
        return x + np.log(1 + np.exp(y - x))


def upperconv(llr1: float, llr2: float) -> float:
    """PERFORMS IN LOG DOMAIN
    llr = (llr1 * llr2 + 1) / (llr1 + llr2)"""
    #return logdomain_sum(llr1 + llr2, 0) - logdomain_sum(llr1, llr2)
    return np.sign(llr1)*np.sign(llr2)*min(abs(llr1),abs(llr2))


def logdomain_sum2(x, y):
    return np.array([x[i] + np.log(1 + np.exp(y[i] - x[i])) if x[i] >= y[i]
                     else y[i] + np.log(1 + np.exp(x[i] - y[i]))
                     for i in range(len(x))])

    
def upperconv2(llr1, llr2):
    """PERFORMS IN LOG DOMAIN
    llr = (llr1 * llr2 + 1) / (llr1 + llr2)"""
    return logdomain_sum2(llr1 + llr2, np.zeros(len(llr1))) - logdomain_sum2(llr1, llr2)



####PAC########################################

def conv_1bit(in_bit, cur_state, gen): 
    #This function calculates the 1 bit convolutional output during state transition
    g_len = len(gen)    #length of generator 
    g_bit = in_bit * gen[0]        

    for i in range(1,g_len):       
        if gen[i] == 1:
            #print(i-1,len(cur_state))
            #if i-1 > len(cur_state)-1 or i-1 < 0:
                #print("*****cur_state idex is {0} > {1}, g_len={2}".format(i-1,len(cur_state),g_len))
            g_bit = g_bit ^ cur_state[i-1]
    return g_bit



def getNextState(in_bit, cur_state, m):
#This function finds the next state during state transition
    #next_state = []
    if in_bit == 0:
        next_state = [0] + cur_state[0:m-1] # extend (the elements), not append
    else:
        next_state = [1] + cur_state[0:m-1]  #np.append([0], cur_state[0:m-1])     
    return next_state


def conv1bit_getNextStates(in_bit, cur_state1, cur_state2, gen1, gen2, bit_flag):
    m1 = len(gen1)-1
    m2 = len(gen2)-1

    g_bit = in_bit       

    if bit_flag == 1:
        for i in range(2,m1+1):       
            if gen1[i] == 1:
                g_bit = g_bit ^ cur_state1[i-1]
        for i in range(1,m2+1):       
            if gen2[i] == 1:
                g_bit = g_bit ^ cur_state2[i-1]
        if in_bit == 0:
            next_state2 = [0] + cur_state2[0:m2-1] # extend (the elements), not append
        else:
            next_state2 = [1] + cur_state2[0:m2-1]  #np.append([0], cur_state[0:m-1])
        if in_bit == 0:
            next_state1 = [0] + cur_state1[0:m1-1] # extend (the elements), not append
        else:
            next_state1 = [1] + cur_state1[0:m1-1]  #np.append([0], cur_state[0:m-1])
        #next_state1 = cur_state1
    else:
        for i in range(1,m1+1):       
            if gen1[i] == 1:
                g_bit = g_bit ^ cur_state1[i-1]
        for i in range(2,m2+1):       
            if gen2[i] == 1:
                g_bit = g_bit ^ cur_state2[i-1]
        if in_bit == 0:
            next_state1 = [0] + cur_state1[0:m1-1] # extend (the elements), not append
        else:
            next_state1 = [1] + cur_state1[0:m1-1]  #np.append([0], cur_state[0:m-1])     
        next_state2 = cur_state2
    
    return g_bit, next_state1, next_state2





def conv_encode(in_code, gen, m):
    # function to find the convolutional code for given input code (input code must be padded with zeros)
    #cur_state = np.zeros(m, dtype=np.int)         # intial state is [0 0 0 ...]
    cur_state = [0 for i in range(m)]#np.zeros(m, dtype=int)
    len_in_code = len(in_code)           # length of input code padded with zeros
    conv_code = np.zeros(len_in_code, dtype=int)     
    log_N = int(math.log2(len_in_code))
    for j in range(0,len_in_code):
        i = bitreversed(j, log_N)
        in_bit = in_code[i]              # 1 bit input 
        #if cur_state.size==0:
            #print("*****cur_state len is {0}, m={1}".format(cur_state.size,m))
        output = conv_1bit(in_bit, cur_state, gen);    # transition to next state and corresponding 2 bit convolution output
        cur_state = getNextState(in_bit, cur_state, m)    # transition to next state and corresponding 2 bit convolution output
        #conv_code = conv_code + [output]  #list   # append the 1 bit output to convolutional code
        conv_code[i] = output
    return conv_code



def bin2dec(binary): 
    decimal = 0
    for i in range(len(binary)): 
        decimal = decimal + binary[i] * pow(2, i) 
    return decimal







