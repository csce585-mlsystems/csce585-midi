
## Problem
Our group is interested in AI music generation via MIDI files. Both of us are interested in music as well as machine learning, making this topic the perfect combination of subjects for us to work on. MIDI files contain instructions, rather than audio data, which tell software how to play a song. These instructions are contained in chunks, containing event data such as notes and control changes. Despite not being human readable, MIDI data is easily translatable into a variety of formats, and is used as the core for Digital Audio Workstation editors. Although AI models such as MusicLM exist to generate music, these create raw audio in the form of waveforms. As such, it is very hard for a user to iterate upon its creations, as changes would require the entire waveform to be regenerated. The use of MIDI allows for small, incremental tweaks, while still keeping the end user as part of the process through their DAW.

## Solution
We plan on using prompt chaining to create an automated workflow that allows a user to create MIDIs through describing a song in plaintext. First, an LLM - such as Llama 3.2 11b - will parse the user’s text input to determine the genre, style, and needed instruments for the track. Models like Llama support structured output, where they generate a JSON file following a pre-written structure. This data can then be fed into our model to generate a MIDI output, which either gets rendered or sent back to the user, depending on needs.

For our model, we plan on using the XMIDI data set, an open-source collection of 100,000 MIDIs which are pre-labeled with emotion and genres. However, this dataset is limited to only a dozen emotions, and even less genres. To supplement this data, we can also use the The Lakh MIDI Dataset. Although not labeled with genres or emotions, MIDI track data contains labels for the instruments used. So, we can train a model on both genres, and instrument-specific data.
![Project Tool Chain](https://drive.google.com/uc?export=view&id=1ZWAUYTGqqgNbOcxX6M16KOMATjNMAJo7)
https://drive.google.com/uc?export=view&id=1ZWAUYTGqqgNbOcxX6M16KOMATjNMAJo7
## Evaluation

The generated MIDI files can be statistically analyzed by properties like note density, pitch distribution, pitch range, and polyphony. There are several established Python libraries designed for symbolic music processing:
**pretty_midi:** easy way to parse, analyze, and manipulate MIDI files. This library is good for extracting features from MIDI files like polyphony, note density, rhythm histograms, etc.
**music21:** a framework for symbolic music analysis. This framework provides tools for tonal analysis (keys, chords, harmonic progressions, etc.)
**miditoolkit:** an alternative to pretty_midi for handling symbolic music. Used to manipulate symbolic sequences.

## References

Zhu, Y., Baca, J., Rekabdar, B., Rawassizadeh, R. (2023). A Survey of AI Music Generation Tools and Models. [arXiv:2308.12982](https://arxiv.org/abs/2308.12982)

Briot, J., Hadjeres, G., Pachet, F. (2017). Deep Learning Techniques for Music Generation -- A Survey.[arXiv:1709.01620](https://arxiv.org/abs/1709.01620)

Bhandari, K., Roy, A., Wang, K., Puri, G., Colton, S., Herremans, D. (2024). Text2midi: Generating Symbolic Music from Captions. [arXiv:2412.16526](https://arxiv.org/abs/2412.16526)

Yang, L., Chou, S., Yang, Y. (2017). MidiNet: A Convolutional Generative Adversarial Network for Symbolic-domain Music Generation. [arXiv:1703.10847](https://arxiv.org/abs/1703.10847)

Tian, S., Zhang, C., Yuan, W., Tan, W., Zhu, W. (2025). XMusic: Towards a Generalized and Controllable Symbolic Music Generation Framework. [arXiv:2501.08809](https://arxiv.org/abs/2501.08809)

Colin Raffel. **"Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching"**. _PhD Thesis_, 2016. https://colinraffel.com/publications/thesis.pdf
