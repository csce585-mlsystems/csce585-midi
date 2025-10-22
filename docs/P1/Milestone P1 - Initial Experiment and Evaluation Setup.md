# csce585-midi: P1

## Group Info   
- Cade Stocker 
	- Email: cstocker@email.sc.edu  

## System / Dataset
This project is using the Nottingham dataset, a collection of 1200 British and American folk tunes. Using this small dataset allows quick training for multiple models, making it easy to compare training and sampling strategies. Eventuall, I would like to use the Lakh Piano Roll Dataset, which contains 21,425 multitrack piano rolls.

I am using two forms of tokenization for the experiments I will run. The first is naive, which represents each note by an integer (associated with each letter of the musical alphabet). Secondly is Miditok, which represents each note with several tokens, representing pitch, duration, and several other factors meant to give the model more context.

 So far with the small models I have trained,the naive tokens have generated higher quality music. I predict that as I train larger models (especially transformers), the output of models trained on Miditok tokens will become higher quality than naive tokens.

## Baseline Implementation
Please refer to slide 7 on the slideshow included in **/docs/P1/** for a visualization of the pipeline. I am implementing the architecture described in the Musenet paper. This includes a discriminator, which predicts the next measures chord (or more accurately, a list of several likely upcoming notes). Next, the generator selects the notes for the next measure based on the discriminator's output.

Currently, I have factory design patterns for both the discriminator and generator, allowing different types of models to be trained (transformer, lstm, mlp, etc.). Musenet tried both LSTMs and Transformers for the generator, and a MLP for the discriminator.

### Current Options for Discriminator and Generator:

#### Discriminator
- LSTM
- MLP
- Transformer

#### Generator
- GRU
- LSTM
- Transformer


I haven't began running experiments on these yet, but I will be able to configure different combinations of these models and empirically measure their generated music via the **pretty_midi** package.

## Preliminary Experiment
While running these experients, the pipeline was laid out as...

**MIDI Files -> Tokenization -> Single LSTM -> Output Tokens -> MIDI File**

Each time a model is trained, data about its training is sorted into the correct log file (one file for naive generators, one for miditok generators, etc.). Additionally, when a MIDI file is generated, it is automatically evaluated by several quantitative measurements (polyphony, pitch range, etc.) and is logged in the csv corresponding to the model type that did the generation.

**Refer to the graphics in the slideshow mentioned previously.**

Miditok models predictably had a much larger vocabulary, and took far longer to train. Models trained on naive tokens consistently created longer songs, since each token output is represented as a note or chord, whereas miditok models require many tokens to represent one note.

Naive songs were generally more pleasing to listen to, but often output chords in strange places where notes should have gone. Naive was able to take a seed that included a melody, and then reproduced the melody with some variation.

Miditok models have been producing all chords the majority of the time. I don't understand yet what causes this.