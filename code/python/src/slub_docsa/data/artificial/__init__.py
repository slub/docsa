"""Various artificially generated datasets for testing purposes.

Artificially generated data allows to control data properties and highlight advantages and disadvantages of certain
models.

For example, random data, where there is no correlation between documents and their annotated subjects, can not
be meaningfully trained. Therefore, this data can be used to validate that a model implementation does not cheat in
some way when predicting.

More importantly, random data can also be generated in a way that documents correlate with artificially defined target
subjects based on certain patterns (e.g. word occurance statistics, n-gram statistics, etc.). Then, models can be
trained on this data and should report better performance than random guessing.

When generating hierarchically correlated artificial documents, models can be compared in order to understand whether
they actually are capable of integrating hierarchical relations between subjects, and to what degree this information
is improving their prediction performance.
"""
