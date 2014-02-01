import matplotlib.pyplot as plt
import numpy as np
import cPickle

"""
The sole purpose of this script is to show that data is plausible and unprocessed.
Reading it may also see how to actually use the data before someone 
makes a wrapper for it (it might even help you build that wrapper to share). 
"""


print "Loading sentences...", 
train_sentences = np.load("/data/lisa/data/timit/readable/train_x_raw.npy")
print "Done !"

# phonemes
print "Loading phonemes...", 
train_phonemes = np.load("/data/lisa/data/timit/readable/train_phn.npy")
train_sentence_to_phonemes = np.load("/data/lisa/data/timit/readable/train_seq_to_phn.npy")

f = open("/data/lisa/data/timit/readable/phonemes.pkl", "r")
phonemes_list = cPickle.load(f)
f.close()
print "Done !"

# words
print "Loading words...", 
train_words = np.load("/data/lisa/data/timit/readable/train_wrd.npy")
train_sentence_to_words = np.load("/data/lisa/data/timit/readable/train_seq_to_wrd.npy")

f = open("/data/lisa/data/timit/readable/words.pkl", "r")
words_list = cPickle.load(f)
f.close()
print "Done !"

# speakers
print "Loading speakers...", 
speakers = np.load("/data/lisa/data/timit/readable/train_spkr.npy")
speakers_info = np.load("/data/lisa/data/timit/readable/spkrinfo.npy").tolist().toarray()

f = open("/data/lisa/data/timit/readable/spkr_feature_names.pkl","r")
speakers_features = cPickle.load(f)
f.close()
print "Done !"

# picking a random sentence
k = np.random.randint(len(train_sentences))
sentence = train_sentences[k]

# plotting the vector
try:
    print "Look at that beautiful waveform..."
    plt.plot(range(len(sentence)), sentence)
    plt.show()
    print "Yes. This is what raw sound looks like. No preprocessing.\n"
except:
    print "I can't show you what it looks like. but.."

print "It's so beautiful I'll save a copy right here...", 

# saving the plot
is_female = (speakers_info[speakers[k],-1] == 0)
try:
    plt.plot(range(len(sentence)), sentence)
    if is_female:
        plt.savefig("that_s_what_she_said.png")
    else:
        plt.savefig("that_s_what_he_said.png")
        
    print "Done !"
except:
    print "Except I can't..."

# obtain the speaker informations
speaker_info = speakers_info[speakers[k]]
speakers_features_arr = np.array(speakers_features)
print "Speaker features are : "
print "Age : "+str(speaker_info[0])
print "Height : "+str(speaker_info[15]/100)+"m"
print speakers_features_arr[1:9][speaker_info[1:9] == 1]
print speakers_features_arr[9:15][speaker_info[9:15] == 1]
print speakers_features_arr[16:24][speaker_info[16:24] == 1]
if is_female:
    print "The speaker is female."
else:
    print "The speaker is male."

# obtain the list of phonemes
phonemes_used = []
for i in range(train_sentence_to_phonemes[k,0], train_sentence_to_phonemes[k,1]):
    phonemes_used.append(phonemes_list[train_phonemes[i,2]])
print "The speaker says : \""+" ".join(phonemes_used)+"\""

# obtain the list of words
words_used = []
for i in range(train_sentence_to_words[k,0], train_sentence_to_words[k,1]):
    words_used.append(words_list[train_words[i,2]])
print "That reads : \""+" ".join(words_used)+"\""

print "Did I forgot the punctuation ? Very well then I forgot the punctuation.\n"

print "You can look at the speaker features and words to trace back the wav file."
print "Also, try : less /data/lisa/data/timit/raw/TIMIT/README.DOC"
print "And : less /data/lisa/data/timit/raw/TIMIT/DOC/SPKRINFO.TXT"
