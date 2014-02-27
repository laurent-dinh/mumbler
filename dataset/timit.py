import numpy as np
import os
import os.path
import cPickle
from exceptions import *
from segmentaxis import segment_axis
import scipy.stats
import pdb
import collections


class TIMIT(object):
    """
    This class will encapsulate the interactions that we will have with TIMIT.
    You should have the environment variable MUMBLER_DATA_PATH set. One way to 
    do this is to put 'export MUMBLER_DATA_PATH=/path/to/your/datasets/folder/' 
    in your .bashrc file so that $MUMBLER_DATA_PATH/timit_concat link to 
    /data/lisatmp/dinhlaur/timit_concat
    
    """
    def __init__(self, mmap_mode = None):
        """
        Initialize the TIMIT class. 
        """
        timit_path = os.path.join(os.environ["MUMBLER_DATA_PATH"], \
                                  "timit_concat")
        
        if os.path.isdir(timit_path):
            self.timit_path = timit_path
        else:
            raise IOError(timit_path + " is not a valid path !")
        
        self.has_train = False
        self.has_valid = False
        self.has_test = False
        
        spkrinfo_path = os.path.join(self.timit_path, "spkrinfo.npy")
        phonemes_path = os.path.join(self.timit_path, "phonemes.pkl")
        phones_path = os.path.join(self.timit_path, "phones.pkl")
        wrds_path = os.path.join(self.timit_path, "words.pkl")
        spkrfeat_path = os.path.join(self.timit_path, "spkr_feature_names.pkl")
        spkrid_path = os.path.join(self.timit_path, "speakers_ids.pkl")
        shuffling_path = os.path.join(self.timit_path, "shuffling.npy")
        
        for p in [spkrinfo_path, wrds_path, phones_path, phonemes_path, \
                  spkrfeat_path, spkrid_path, shuffling_path]:
            if not os.path.isfile(p):
                raise IOError(p + " is not a valid path !")
        
        ## Speaker information
        print "Loading speaker information...", 
        self.spkrinfo = np.load(spkrinfo_path).tolist().toarray()
        print "Done !"
        
        print "Loading speakers list...", 
        self.spkrid = cPickle.load(open(spkrid_path, "r"))
        print "Done !"
        print str(len(self.spkrid)) + " different speakers."
        
        print "Loading speakers freatures...", 
        self.spkrfeat = cPickle.load(open(spkrfeat_path, "r"))
        print "Done !"
        print str(len(self.spkrfeat)) + " different features per speaker."
        
        # Words
        print "Loading words list...", 
        self.words = cPickle.load(open(wrds_path, "r"))
        print "Done !"
        print str(len(self.words)) + " different word."
        
        # Phonemes
        print "Loading phonemes list...", 
        self.phonemes = np.load(open(phonemes_path, "r"))
        print "Done !"
        print str(len(self.phonemes)) + " different phonemes."
        
        # Phones
        print "Loading phones list...", 
        self.phones = np.load(open(phones_path, "r"))
        print "Done !"
        print str(len(self.phones)) + " different phones."
        
        # Shuffling
        print "Loading shuffling...", 
        self.shuffling = np.load(open(shuffling_path, "r"))
        self.invert_shuffling = np.argsort(self.shuffling)
        print "Done !"
        
        self.shuffle_seq = True
         
        self.mode = None
        
        
    def load(self, subset):
        """
        Extract the data from the files given the path of the preprocessed 
        TIMIT. It also prints some information on the dataset. 
        timit_path: path to the preprocessed TIMIT. 
        subset: either "train", "valid" or "test".
        """
        self.check_subset_value(subset)
        
        print "Loading dataset subset."
        # Build paths
        print "Building paths...", 
        wav_path = os.path.join(self.timit_path, subset+"_wav.npy")
        intervals_path = os.path.join(self.timit_path, subset+"_intervals.npy")
        phones_path = os.path.join(self.timit_path, subset+"_phones.npy")
        phonemes_path = os.path.join(self.timit_path, subset+"_phonemes.npy")
        words_path = os.path.join(self.timit_path, subset+"_words.npy")
        words_intervals_path = os.path.join(self.timit_path, \
                                subset+"_words_intervals_by_seq.npy")
        seq_to_words_path = os.path.join(self.timit_path, \
                                       subset+"_seq_to_words.npy")
        speakers_path = os.path.join(self.timit_path, subset+"_speakers.npy")
        print "Done !"
        
        # Checking the validity of the paths
        print "Checking path validity...", 
        for p in [wav_path, phones_path, phonemes_path, intervals_path, \
                  words_intervals_path, seq_to_words_path, speakers_path]:
            if not os.path.isfile(p):
                raise IOError(p + " is not a valid path !")
        
        print "Done !"
        
        # Acoustic samples
        print "Loading accoustic samples...", 
        wav = np.load(wav_path, "r")
        intervals = np.load(intervals_path)
        print "Done !"
        print str(intervals.shape[0] -1) + " sentences."
        
        print "Compute normalizers...", 
        max_abs = max(np.max(wav), -np.min(wav))
        std = np.std(wav)
        print "Done !"
        print "Maximum amplitude:", max_abs
        print "Standard deviation: ", std
        
        # Side information
        ## Phonemes
        print "Loading phonemes...", 
        phones = np.load(phones_path, "r")
        phonemes = np.load(phonemes_path, "r") 
        print "Done !"
        
        ## Words
        print "Loading words...", 
        words = np.load(words_path, "r") 
        words_intervals = np.load(words_intervals_path, "r") 
        seq_to_words = np.load(seq_to_words_path, "r") 
        print "Done !"
        
        ## Speaker information
        print "Loading speaker information...", 
        speaker_id = np.asarray(np.load(speakers_path), 'int')
        print "Done !"
        
        
        data = {}
        data["wav"] = wav
        data["intervals"] = intervals
        data["n_seq"] = intervals.shape[0] - 1
        data["phones"] = phones
        data["phonemes"] = phonemes
        data["words"] = words
        data["words_intervals"] = words_intervals
        data["seq_to_words"] = seq_to_words
        data["speaker_id"] = speaker_id
        data["std"] = std
        data["max_abs"] = max_abs
        
        # Raise the flag advertising the presence of data
        self.__dict__["has_"+subset] = True
        
        self.__dict__[subset] = {}
        self.__dict__[subset].update(data)
        
        self.sanity_check(subset)
    
    def clear(self, subset):
        """
        Given the subset id, this method will unload the subset from the class. 
        
        """
        self.check_subset_value(subset)
        self.check_subset_presence(subset)
        
        
        del self.__dict__[subset]["wav"]
        del self.__dict__[subset]["intervals"]
        del self.__dict__[subset]["n_seq"]
        del self.__dict__[subset]["phones"]
        del self.__dict__[subset]["phonemes"]
        del self.__dict__[subset]["words"]
        del self.__dict__[subset]["words_intervals"]
        del self.__dict__[subset]["seq_to_words"]
        del self.__dict__[subset]["speaker_id"]
        del self.__dict__[subset]
        
        # Lower the flag advertising the presence of data
        data["has_"+subset] = False
    
    def check_subset_value(self, subset):
        if subset not in {"train", "test"}:
            raise ValueError("Invalid subset !")
    
    def check_subset_presence(self, subset):
        if not self.__dict__["has_"+subset]:
            raise AssertionError("The data was not loaded yet !")
    
    def sanity_check(self, subset):
        """
        Test of a given set for the consistency of our hypotheses. 
        
        """
        self.check_subset_value(subset)
        self.check_subset_presence(subset)
        
        print "Check the number of speakers..."
        if self.spkrinfo.shape[0] == len(self.spkrid):
            print "OK."
        else:
            print "KO."
        
        print "Check lengths..."
        short = ["phn", "wrd"]
        long = ["phonemes", "words"]
        if self.__dict__[subset]["seq_to_words"][-1,-1] == \
                    self.__dict__[subset]["words_intervals"].shape[0]:
            print "OK for words."
        else:
            print "KO for words."
        
        print "Check multinomial constraints..."
        feature_name = ["dialect", "education", "race", "sex"]
        feature_interval = [(1,9), (9,15), (16,24), (24,26)]
        for i in range(len(feature_name)):
            start = feature_interval[i][0]
            end = feature_interval[i][1]
            if self.spkrinfo[:,start:end].sum() == self.spkrinfo.shape[0]:
                print "OK for "+feature_name[i]+"."
            else:
                print "KO for "+feature_name[i]+"."
    
    """
    This section is about extracting sequences of varying size.
    
    """
    
    def get_raw_seq(self, subset, seq_id, frame_length, overlap, \
                    shuffling = True):
        """
        Given the id of the subset, the id of the sequence, the frame length and 
        the overlap between frames, this method will return a frames sequence 
        from a given set, the associated phonemes and words sequences (including 
        a binary variable indicating change) and the information vector on the 
        speaker.
        
        """
        self.check_subset_value(subset)
        self.check_subset_presence(subset)
        
        # Check if the id is valid
        n_seq = self.__dict__[subset]["n_seq"]
        if seq_id >= n_seq:
            raise ValueError("This sequence does not exist.")
        
        # Get the sequence
        if shuffling:
            seq_id = self.shuffling[seq_id]
        
        wav_start = self.__dict__[subset]["intervals"][seq_id]
        wav_end = self.__dict__[subset]["intervals"][seq_id+1]
        wav = self.__dict__[subset]["wav"][wav_start:wav_end]
        
        # Get the phones, phonemes and words
        phones = self.__dict__[subset]["phones"][wav_start:wav_end]
        phonemes = self.__dict__[subset]["phonemes"][wav_start:wav_end]
        words = self.__dict__[subset]["words"][wav_start:wav_end]
        
        # Find the speaker id
        spkr_id = self.__dict__[subset]["speaker_id"][seq_id]
        # Find the speaker info
        spkr_info = self.spkrinfo[spkr_id]
        
        # Segment into frames
        wav = segment_axis(wav, frame_length, overlap)
        
        # Take the most occurring phone in a sequence
        phones = segment_axis(phones, frame_length, overlap)
        phones = scipy.stats.mode(phones, axis=1)[0].flatten()
        phones = np.asarray(phones, dtype='int')
        
        # Take the most occurring phone in a sequence
        phonemes = segment_axis(phonemes, frame_length, overlap)
        phonemes = scipy.stats.mode(phonemes, axis=1)[0].flatten()
        phonemes = np.asarray(phonemes, dtype='int')
        
        # Take the most occurring word in a sequence
        words = segment_axis(words, frame_length, overlap)
        words = scipy.stats.mode(words, axis=1)[0].flatten()
        words = np.asarray(words, dtype='int')
        
        # Binary variable announcing the end of the word or phoneme
        end_phn = np.zeros_like(phones)
        end_wrd = np.zeros_like(words)
        
        for i in range(len(words) - 1):
            if phones[i] != phones[i+1]:
                end_phn[i] = 1
            if words[i] != words[i+1]:
                end_wrd[i] = 1
        
        end_phn[-1] = 1
        end_wrd[-1] = 1
        
        return [wav, phones, phonemes, end_phn, words, end_wrd, spkr_info]
    
    def get_n_seq(self, subset):
        """
        Given the subset id, return the number of sequence in it.
        
        """
        self.check_subset_value(subset)
        self.check_subset_presence(subset)
        
        return self.__dict__[subset]["n_seq"]
    
    """
    This section is about extracting sequences of fixed size. 
    
    """
    def init_frames_iter(self, subset, n_frames, frame_length, overlap, \
                        shuffling = True):
        """
        Given the subset id, the number of frames wanted, the frame length, 
        the overlap, initialize the associated iterator if need be. 
        
        """
        self.check_subset_value(subset)
        self.check_subset_presence(subset)
        
        # Check if the initialization is needed
        needed = not hasattr(self, "subset")
        needed |= not hasattr(self, "n_frames")
        needed |= not hasattr(self, "frame_length")
        needed |= not hasattr(self, "overlap")
        
        if not needed:
            needed = (self.subset != subset)
            needed |= (self.n_frames != n_frames)
            needed |= (self.frame_length != frame_length)
            needed |= (self.overlap != overlap)
            needed |= (self.mode != "fixed")
            needed |= (self.shuffle_seq != shuffling)
        
        if needed:
            self.subset = subset
            self.n_frames = n_frames
            self.frame_length = frame_length
            self.overlap = overlap
            self.mode = "fixed"
            self.shuffle_seq = shuffling and (subset == "train")
        
            # Compute the required length to build a frame sequence of
            # fixed size
            self.wav_length_required = (n_frames - 1)*(frame_length - overlap) \
                                + frame_length
        
            # Compute the number of unique frame sequence we can extract
            # from each acoustic samples sequence
            lengths = np.copy(self.__dict__[subset]["intervals"])
            lengths[1:] = lengths[1:] - lengths[:-1]
            lengths[1:] = lengths[1:] - self.wav_length_required + 1
            if self.shuffle_seq:
                lengths[1:] = lengths[1:][self.shuffling]
            
            if np.any(lengths[1:] <= 0):
                raise ValueError("A sequence is too short for this framelength value.")
            
            frame_seq_intervals = np.cumsum(lengths)
            self.frame_seq_intervals = np.asarray(frame_seq_intervals, dtype = "int")
    
    
    def get_fixed_size_seq(self, subset, n_frames, frame_length, overlap, ids, \
                            shuffling = True):
        """
        Given the subset id, the number of frames wanted, the frame length, 
        the overlap, and the ids, return multiple arrays corresponding to
        a minibatch of frame sequence of fixed size
        
        """
        
        self.init_frames_iter(subset, n_frames, frame_length, overlap, \
                                shuffling)
        if isinstance(ids, collections.Iterable):
            ids = np.asarray(ids)
        else:
            ids = np.array([ids])
        
        assert np.all(ids < self.frame_seq_intervals[-1])
        
        # Get the sequence
        seq_ids = np.digitize(ids, self.frame_seq_intervals) - 1
        if self.shuffle_seq:
            seq_ids = self.invert_shuffling[seq_ids]
        
        idx_in_seq = ids - self.__dict__[subset]["intervals"][seq_ids]
        wav_start = self.__dict__[subset]["intervals"][seq_ids] + idx_in_seq
        wav_end = wav_start + self.wav_length_required
        wav_intervals = zip(wav_start,wav_end)
        indices = map(lambda x:range(x[0],x[1]), wav_intervals)
        indices = reduce(lambda x,y: x+y, indices)
        indices = np.array(indices).reshape(ids.shape[0], \
                    self.wav_length_required)
        
        wav = self.__dict__[subset]["wav"][indices]
        
        # Get the phones, phonemes and words
        phones = self.__dict__[subset]["phones"][indices]
        phonemes = self.__dict__[subset]["phonemes"][indices]
        words = self.__dict__[subset]["words"][indices]
        
        # Find the speaker id
        spkr_id = self.__dict__[subset]["speaker_id"][seq_ids]
        # Find the speaker info
        spkr_info = self.spkrinfo[spkr_id]
        
        # Segment into frames
        wav = segment_axis(wav, frame_length, overlap, axis=1)
        # shape (n_ids, n_frames, frame_length)
        
        # Take the most occurring phone in a sequence
        phones = segment_axis(phones, frame_length, overlap, axis=1)
        phones = scipy.stats.mode(phones, axis=2)[0].reshape(ids.shape[0], \
                    n_frames)
        phones = np.asarray(phones, dtype='int')
        
        # Take the most occurring phone in a sequence
        phonemes = segment_axis(phonemes, frame_length, overlap, axis=1)
        phonemes = scipy.stats.mode(phonemes, axis=2)[0].reshape(ids.shape[0], \
                    n_frames)
        phonemes = np.asarray(phonemes, dtype='int')
        
        # Take the most occurring word in a sequence
        words = segment_axis(words, frame_length, overlap, axis=1)
        words = scipy.stats.mode(words, axis=2)[0].reshape(ids.shape[0], \
                    n_frames)
        words = np.asarray(words, dtype='int')
        
        # Binary variable announcing the end of the word or phoneme
        end_phn = np.zeros_like(phones)
        end_wrd = np.zeros_like(words)
        
        end_phn[:,:-1] = np.where(phones[:,:-1] != phones[:,1:], 1, 0)
        end_wrd[:,:-1] = np.where(words[:,:-1] != words[:,1:], 1, 0)
        
        return [wav, phones, phonemes, end_phn, words, end_wrd, spkr_info]
        
    
    def get_n_fixed_size_seq(self, subset, n_frames, frame_length, overlap, \
                                shuffling = True, end_seq = None):
        """
        Given the subset id, return the number of fixed size sequences in it.
        
        """
        self.check_subset_value(subset)
        self.check_subset_presence(subset)
        self.init_frames_iter(subset, n_frames, frame_length, overlap, shuffling)
        
        if end_seq is None:
            end_seq = -1
        
        return self.frame_seq_intervals[end_seq]
    
    """
    This section is about extracting sequences for each word. 
    
    """
    def init_words_iter(self, subset, shuffling = True):
        """
        Given the subset id, initialize the iterator if need be. 
        
        """
        self.check_subset_value(subset)
        self.check_subset_presence(subset)
        
        # Check if the initialization is needed
        needed = not hasattr(self, "word_to_seq_intervals")
        self.mode = "words"
        
        if not needed:
            needed = (self.shuffle_seq != shuffling)
        
        self.shuffle_seq = shuffling and (subset == "train")
         
        if needed:
            # Compute the required length to build a frame sequence of
            # fixed size
            n_words_per_seq = self.__dict__[subset]["seq_to_words"][:,1] \
                            - self.__dict__[subset]["seq_to_words"][:,0]
            
            if self.shuffle_seq:
                n_words_per_seq = n_words_per_seq[self.shuffling]
            
            self.word_to_seq_intervals = np.zeros((n_words_per_seq.shape[0]+1))
            self.word_to_seq_intervals[1:] = np.cumsum(n_words_per_seq)
            self.word_to_seq_intervals = np.asarray(self.word_to_seq_intervals, dtype="int")
        
    def get_word_seq(self, subset, frame_length, overlap, id, shuffling = True):
        """
        Given the subset id, the number of frames wanted, the frame length, 
        the overlap and the id, return the associated waveform sequence. 
        
        """
        self.init_words_iter(subset, shuffling)
        assert id < self.word_to_seq_intervals[-1]
        
        # Get the sequence
        seq_id = np.digitize([id], self.word_to_seq_intervals)[0] - 1
        if self.shuffle_seq:
            seq_id = self.invert_shuffling[seq_id]
        
        wav_start_in_seq = self.__dict__[subset]["words_intervals"][id, 0]
        wav_end_in_seq = self.__dict__[subset]["words_intervals"][id, 1]
        wav_start = self.__dict__[subset]["intervals"][seq_id] \
                    + wav_start_in_seq
        wav_end = self.__dict__[subset]["intervals"][seq_id] \
                    + wav_end_in_seq
        
        wav = self.__dict__[subset]["wav"][wav_start:wav_end]
        
        # Get the phones, phonemes and words
        phones = self.__dict__[subset]["phones"][wav_start:wav_end]
        phonemes = self.__dict__[subset]["phonemes"][wav_start:wav_end]
        word = self.__dict__[subset]["words_intervals"][id,2]
        
        # Find the speaker id
        spkr_id = self.__dict__[subset]["speaker_id"][seq_id]
        # Find the speaker info
        spkr_info = self.spkrinfo[spkr_id]
        
        # Segment into frames
        wav = segment_axis(wav, frame_length, overlap)
        
        # Take the most occurring phone in a sequence
        phones = segment_axis(phones, frame_length, overlap)
        phones = scipy.stats.mode(phones, axis=1)[0].flatten()
        phones = np.asarray(phones, dtype='int')
        
        # Take the most occurring phone in a sequence
        phonemes = segment_axis(phonemes, frame_length, overlap)
        phonemes = scipy.stats.mode(phonemes, axis=1)[0].flatten()
        phonemes = np.asarray(phonemes, dtype='int')
        
        
        # Binary variable announcing the end of the word or phoneme
        end_phn = np.zeros_like(phones)
        
        for i in range(len(phones) - 1):
            if phones[i] != phones[i+1]:
                end_phn[i] = 1
        
        end_phn[-1] = 1
        
        return [wav, phones, phonemes, end_phn, word, spkr_info]
        
    
    def get_n_words(self, subset, shuffling = True, end_seq = None):
        """
        Given the subset id, return the number of sequence in it.
        
        """
        self.check_subset_value(subset)
        self.check_subset_presence(subset)
        self.init_words_iter(subset, shuffling)
        
        if end_seq is None:
            end_seq = -1

        return self.word_to_seq_intervals[end_seq]
    
