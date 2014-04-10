import mumbler.dataset.timit
import numpy as np
import scipy.io.wavfile

timit = mumbler.dataset.timit.TIMIT()
timit.load("train")

# create an array with the number of phonemes in each sentence
sizes = np.zeros((len(timit.train["seq_to_phones"]) + 1), dtype="int") 
sizes[:-1] = timit.train["seq_to_phones"][:,0]
sizes[-1] = timit.train["seq_to_phones"][-1,-1]
phones_intervals_in_seq = timit.train["phones_intervals"]

for k in xrange(len(timit.phones)):
    corresponding_intervals = phones_intervals_in_seq[(phones_intervals_in_seq[:,2] == k)]
    lengths = corresponding_intervals[:,1] - corresponding_intervals[:,0]
    mean_length = np.mean(lengths)
    n_seq = lengths.shape[0]
    print k, timit.phones[k], ":", mean_length, n_seq

k = 28
# get the corresponding intervals in sequence
phone_intervals_in_seq = timit.train["phones_intervals"][timit.train["phones_intervals"][:,2] == k,:-1]
# get the corresponding sentences
seq_ids = np.digitize(np.where(timit.train["phones_intervals"][:,2] == k)[0], sizes) -1

# get the position in the acoustic samples vector
starting_pos = timit.train["intervals"][seq_ids]
phone_intervals = starting_pos.reshape((len(phone_intervals_in_seq),1)) + phone_intervals_in_seq

phone_seqs = []
for i in xrange(len(phone_intervals_in_seq)):
    a, b = phone_intervals[i]
    phone_seqs.append(np.copy(timit.train["wav"][a:b]))

# save the data
np.save("wav_"+timit.phones[k]+".npy", phone_seqs)
