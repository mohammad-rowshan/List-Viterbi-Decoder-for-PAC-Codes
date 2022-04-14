# Simulator ###########################################################################
#
# Copyright (c) 2021, Mohammad Rowshan
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, 
# are permitted provided that:
# the source code retains the above copyright notice, and te redistribtuion condition.
# 
# Freely distributed for educational and research purposes
#######################################################################################

from time import time
from polar_code import PolarCode
from channel import channel
import functions as pcf
import numpy as np

N = 2**7  # Block length
R = 0.5  # Code rate
K = int(N*R)  # info bits
# Rate profile:
construct = "rm" #"rm": Reed-Muller Polar, #"dega": Density Evolution with Gaussian Approximation
dsnr = 3.5  # Design SNR for DEGA
# Coefficients of the convoutional generator polynomial
conv_gen = [1,1,1] #[1,1,1,0,1,1,0,1,1] #[1,1,1,0,1,1] #,1,0,1] #[1,0,1,1,0,1,1] #[1,1,1,1,1,1,1]#0o177 [1,1,1,1,0]#:0o36 [1,1,0,1,1,1,0,0,1,1]#:0o1563 #[1,0,1,1,0,1,1]:0o133 ##[1,0]# by convention: c_0=1, c_m=1
m = len(conv_gen)-1 # Memory size = constraint_length-1 # Do not change it

S = 2**3     # local list size #2**0=1 is equivalent to Viterbi Algortithm
list_size = 2**m * S  # Total # paths # Do not change it

pcode = PolarCode(N, K, construct, dsnr, L=list_size)

pcode.path_select = S #For list Viterbi Alg.
pcode.num_paths = list_size

pcode.iterations = 10**7
err_cnt = 9  # Total number of error to count at each SNR # Default: 49
# Alternatively, consider a fixed number of iterations, e.g., at each SNR. You need to change the code, specifically line 85
pcode.snrb_snr = 'SNRb' # 'SNRb':Eb/N0, 'SNR':Es/N0
snr_range = np.arange(1,1.5,0.5)#arange(start,endpoint+step,step )

pcode.m = m
pcode.gen = conv_gen
pcode.modu = 'BPSK'

print("PAC({0},{1}) constructed by {2}({3}dB), conv_gen={4}".format(N, K,construct,dsnr,conv_gen))
print("LVA, Local list Size(S)={}, Total # paths={}".format(S,list_size))
print("BER & FER evaluation is started")

st = time()
systematic = False

class BERFER():
    """structure that keeps results of BER and FER tests"""
    def __init__(self):
        self.fname = str()
        self.label = str()
        self.snr_range = list()
        self.ber = list()
        self.fer = list()
        self.mod_type = str()
        self.xrange = list()

result = BERFER()

for snr in snr_range:
    pcode.cur_state = [0 for i in range(m)]
    print("\nSNR={0} dB".format(snr))

    ber = 0
    fer0 = 0
    fer = 0
    ch = channel(pcode.modu, snr, pcode.snrb_snr, (K / N)) 
    for t in range(pcode.iterations):

        pcode.cur_state = [0 for i in range(m)] #This line wasn't here initially and that was the cause of degradation
        message = np.random.randint(0, 2, size=K, dtype=int)
        x = pcode.pac_encode(message, conv_gen, m)
        modulated_x = ch.modulate(x)
        y = ch.add_noise(modulated_x)
        llr_ch = ch.calc_llr(y)
        decoded = pcode.pac_viterbi_decoder(llr_ch, issystematic=systematic)
        
        
        err_bits = pcf.fails(message, decoded)
        ber += err_bits
        if err_bits > 0:
            fer += 1
            print("Error # {0} t={1}, FER={2:0.2e}".format(fer,t, fer/(t+1)))

        #fer += not np.array_equal(message, decoded)
        if fer > err_cnt:
            print("@ {0} dB FER is {1:0.2e}".format(snr, fer/(t+1)))
            break
        if t%2000==0:
            print("\nt={0} FER={1:0.2e}".format(t, fer/(t+1)))
            
        if t==pcode.iterations: 
            break

    result.snr_range.append(snr)
    result.ber.append(ber / ((t + 1) * K))
    result.fer.append(fer / (t + 1))


    print("\n\n")
    print(result.label)
    print("SNR\t{0}".format(result.snr_range))
    print("FER\t{0}".format(result.fer))
    print("BER\t{0}".format(result.ber))



#Filename for saving the results
result.fname += "PAC({0},{1}),L{2},m{3}".format(N, K,list_size,m)
    
#Writing the resuls in file
with open(result.fname + ".csv", 'w') as f:
    result.label = "PAC({0}, {1})\nL={2}\nRate-profile={3}\ndesign SNR={4}\n" \
                "Conv Poly={5}\n".format(N, K,
                pcode.list_size, construct, dsnr, conv_gen)
    f.write(result.label)

    f.write("\nSNR: ")
    for snr in result.snr_range:
        f.write("{0}; ".format(snr))
    f.write("\nBER: ")
    for ber in result.ber:
        f.write("{0}; ".format(ber))
    f.write("\nFER: ")
    for fer in result.fer:
        f.write("{0}; ".format(fer))

print("\n\n")
print(result.label)
print("SNR\t{0}".format(result.snr_range))
print("BER\t{0}".format(result.ber))
#print("FER\t{0:1.2e}".format(result.fer))
print("FER\t{0}".format(result.fer))

print("time on test = ", str(time() - st), ' s\n------------\n')



