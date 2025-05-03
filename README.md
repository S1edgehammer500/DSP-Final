# DSP-Final
The requirements for this code (run pip install for each) are:
for getting the dataset:
requests
bs4

To train the model:
datasets
transformers
torch
evaluate
numpy
pandas
collections
sklearn
spacy

To test the model:
transformers
evaluate
numpy
pandas
torch
sklearn

For running the UI with the model:
streamlit
transformers
torch
spacy
heapq
pandas



In order to train and run the model:
The dataset has already been gathered in movie_scripts but for interest gettingData.py creates the dataset with movie scripts, addingRatings.py adds ratings to the scripts, cleaningData.py removes empty files and those with invalid age ratings, and balancingDataset.py balances the dataset.
In order to train the model, fine_tuning_Longformer.py must be run. This will create an age_rating_classifier folder with the model in it and will print out where the best checkpoint is located as well as testing the data
Note: I had to run the training code on Google Colab using the A100 GPU and high-RAM and it took around 1 hour 10 mins
To run the following, you will need to change the path at the top of the files to match the best checkpoint from training. My finished model was too big to upload to GitHub or Blackboard so training may need to be preformed again if you wish to run this code outside of the demo.
If you want to just test data without training, you can run Longformer_Test_Data.py, though you can only do this after training as test data will have been created and preprocessed there
The file with the code for running the UI is with the model is predictingRating.py
You can run it with this in the Terminal:
python -m streamlit run predictingRating.py


