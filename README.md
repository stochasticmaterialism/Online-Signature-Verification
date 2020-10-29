# Online_Signature_Verification
A novel method for online signature verification using 2-tire ensemble

In this method, we propose a 2-tire ensemble model for online signature verification. 3 types of features are initially extracted, namely physical,frequeny based and statistical. 

Ensemble-1 is applied to each of these 3 features individually. It consists of 7 classifiers. Normalised Distribution Summation framework has been used for this ensemble. So, basically we have 3 models one for each type of features.

Ensemble-2 is combines the 3 models of Ensemble-1. The ensemble strategy used here is Majority Voting.

The SVC-2004 dataset is available in text file format. It can be accessed using the link https://www.cse.ust.hk/svc2004/.

The MCYT-100 dataset is avalable in FPG format which can be converted to text format using the MATLAB code provided along with the dataset. The dataset is not open access and needs to be applied for. The link to access the dataset's application form is http://atvs.ii.uam.es/atvs/mcyt100s.html.
