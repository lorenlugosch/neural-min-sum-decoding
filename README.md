# Neural min-sum decoding in TensorFlow

This repo contains the code for the neural min-sum decoding algorithms used to obtain the results shown in the following three papers:

* **L. Lugosch** and W. J. Gross, "Neural offset min-sum decoding," IEEE International Symposium on Information Theory (ISIT), Aachen, 2017, pp. 1361-1365. https://arxiv.org/abs/1701.05931

* E. Nachmani, E. Marciano, **L. Lugosch**, W. J. Gross, D. Burshtein and Y. Be’ery, "Deep learning methods for improved decoding of linear codes," in IEEE Journal of Selected Topics in Signal Processing, special issue on "Machine Learning for Cognition in Radio Communications and Radar", vol. 12, no. 1, pp. 119-131, Feb. 2018. https://arxiv.org/abs/1706.07043

* **L. Lugosch** and W. J. Gross, "Learning from the syndrome," in Asilomar Conference on Signals, Systems, and Computers, special session on "Machine Learning for Wireless Systems", Oct. 2018. https://arxiv.org/abs/1810.10902

as well as my Masters thesis:

* **L. Lugosch**, “Learning algorithms for error correction”, Masters thesis, McGill University, 2018. http://lorenlugosch.github.io/Masters_Thesis.pdf

Please cite my work if this code or my papers are useful for you.

*Note:* I made a mistake implementing Nachmani *et al.*'s neural belief propagation decoder in the first version of the NOMS paper: I used just one weight per edge, whereas they use a separate weight for each input message while calculating each output message (so the message for a given edge will be multiplied by a different weight, depending on which output message you are computing). The final version of the NOMS paper that appeared at *ISIT 2017* (the final version on arXiv) has the correct equations and updated BER results. Nachmani *et al.* kindly gave me the BER results for the BCH(63,36) and BCH(63,45) decoders, but not for the BCH(127,106) decoder because of computational constraints, so the results in the NOMS paper for the last code use just one weight per edge. The "neural belief propagation decoder" in this repo is the version with one weight per edge.

I *might* release a cleaner version of this some time in the future, but probably not, because I'm working on other stuff now.

Good luck!
