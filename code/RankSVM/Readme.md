The code is taken from: https://filebox.ece.vt.edu/~parikh/relative_attributes/ranksvm_with_sim.m

The code is in Matlab

You have to pass X which is the feature vector of the images.

O_ and S_ can be computed using the ranksvm_rank.py file (this has to changed for every attribute)

C_O=0.1*ones(7,1)
C_S=0.1*ones(672,1) (for what we used )


To run the code we have to provide all these as inputs
 
Now with these we can obtain a weight vector.

Since the conversion to Rankspace was just a vector multiplication that code isn't added as it can be easily written down.

Use those weight vectors to get a Rank space for zero shot learning

