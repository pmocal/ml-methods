Parameters like MAXDEPTH, NUMTREES, and ITERS (number of iterations for cross validation) are all at the top of `decision-trees.py`.

Currently the code is calling validation() at the bottom which sets aside 1/5 of the training set as a validation set and prints the error on the validation set. To run the code on the test set call test() at the bottom instead.

To run a random forest run the code as is. To run just one decision tree with every feature considered a couple of modifications must be made. First of all, assign NUMTREES = 1 at the top of the code. Second, inside the buildDecisionTree function, comment out lines 65-67 and comment in line 69.
