# List Viterbi Decoder for PAC Codes
If you find this algorithm useful, please cite the following paper. Thanks.

M. Rowshan and E. Viterbo, "List Viterbi Decoding of PAC Codes," in IEEE Transactions on Vehicular Technology, vol. 70, no. 3, pp. 2428-2435, March 2021, doi: 10.1109/TVT.2021.3059370.

https://ieeexplore.ieee.org/document/9354542

Abstract: Polarization-adjusted convolutional (PAC) codes are special concatenated codes in which we employ a one-to-one convolutional transform as a pre-coding step before the polar transform. In this scheme, the polar transform (as a mapper) and the successive cancellation process (as a demapper) present a synthetic vector channel to the convolutional transformation. The numerical results show that this concatenation improves the Hamming distance properties of polar codes. Motivated by the fact that the parallel list Viterbi algorithm (LVA) sorts the candidate paths locally at each trellis node, in this work, we adapt the trellis, path metric, and the local sorter of LVA to PAC codes and show how the error correction performance moves from the poor performance of the Viterbi algorithm (VA) to the superior performance of list decoding by changing the constraint length, list size, and the sorting strategy (local sorting and global sorting) in the LVA. Also, we analyze the complexity of the local sorting of the paths in LVA relative to the global sorting in the list decoding and we show that LVA has a significantly lower sorting complexity compared with list decoding.

The main file is simulator.py in which you can set the parameters of the code and the channel.

Please report any bugs to mrowshan at ieee dot org
