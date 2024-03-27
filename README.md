# Transformer Hangman

Inspired by attention is all you need, write a transformer encoder from
scratch and use it to encoding english words with missing character, and 
then use the higher dimension to predict the missing words. The purpose of
this project is to show how useful and comprehensive transformer encoder can
be. The trained hangman player can beat the most sohpisticated frequency based 
alogrithm in out of sample. 
## Last Layer of model
After encoder, multiple method is used to generate the
prediction character. It turn out the pool max method is
the best among all of those(including pool average)
## Loss function
since the missing words can be multiple, this is a 
multi-label prediction question. It turn out to be
KL divergence is a better loss function than Binary
Cross Entrophy.
