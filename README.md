# Compass
AI system that determines the political leaning of a news article. A project by Spencer Evans, Daniel Alpert, and Dylan Farrell


# Running the RNN
To run the RNN you'll need to have downloaded tensorflow, keras, pandas, numpy, and sklearn. 

Once you have that, run either the cross_val.py or the transferKeras2.py file with the appropriate arguments. Note that the RNN will need to train itself first which takes a while. 

For all of the RNN's below, use the all_sentences.xlsx database as input.

For generating basic RNN + LSTM results:
run the ./transferKeras2.py file with epochs=30, cnn_flag=0 and comment out the GloVe embedding and comment in the embedding labeled "alternative to GloVe"

For generating RNN + LSTM + CNN results:
run the ./transferKeras2.py file with epochs=30, cnn_flag=1 and comment out the GloVe embedding and comment in the embedding labeled "alternative to GloVe"

For generating RNN + LSTM + CNN + GloVe results:
run the ./transferKeras2.py file with the below parameters and make sure GloVe embedding part is commented in. 

Parameters:
('epochs', 30), ('batch_size', 512), ('validation_split', 0.3), ('loss', 'binary_crossentropy'), ('dropout', 0.1), ('opt_flag', 0), ('opt_lr', 0.01), ('cnn_flag', 1), ('kernel_size', 64)


For generating RNN + GRU + CNN + GloVe results:
run the ./transferKeras2.py file with the below parameters and make sure GloVe embedding part is commented in. 

Parameters:
[('epochs', 20), ('batch_size', 128), ('validation_split', 0.3), ('loss', 'binary_crossentropy'), ('dropout', 0.0), ('opt_flag', 1), ('opt_lr', 0.01), ('cnn_flag', 1), ('kernel_size', 128), ('lstm_flag', 0)]
