import numpy as np
import os
import os.path
import cPickle
from exceptions import *
from segmentaxis import segment_axis
import scipy.stats

class TIMIT(object):
    """
    This class will encapsulate the interactions that we will have with TIMIT.
    You should have the environment variable MUMBLER_DATA_PATH set. One way to 
    do this is to put 'export MUMBLER_DATA_PATH=/path/to/your/datasets/folder/' 
    in your .bashrc file so that $MUMBLER_DATA_PATH/readable_timit link to 
    /data/lisa/data/timit/readable
    
    """
    def __init__(self, mmap_mode = None):
        """
        Initialize the TIMIT class. 
        """
        timit_path = os.path.join(os.environ["MUMBLER_DATA_PATH"], \
                                  "readable_timit")
        
        if os.path.isdir(timit_path):
            self.timit_path = timit_path
        else:
            raise IOError(timit_path + " is not a valid path !")
        
        self.has_train = False
        self.has_valid = False
        self.has_test = False
        
        spkrinfo_path = os.path.join(self.timit_path, "spkrinfo.npy")
        phns_path = os.path.join(self.timit_path, "reduced_phonemes.pkl")
        wrds_path = os.path.join(self.timit_path, "words.pkl")
        spkrfeat_path = os.path.join(self.timit_path, "spkr_feature_names.pkl")
        spkrid_path = os.path.join(self.timit_path, "speakers_ids.pkl")
        
        for p in [spkrinfo_path, wrds_path, phns_path, spkrfeat_path, \
                  spkrid_path]:
            if not os.path.isfile(p):
                raise IOError(p + " is not a valid path !")
        
        ## Speaker information
        print "Loading speaker information...", 
        self.spkrinfo = np.load(spkrinfo_path).tolist().toarray()
        print "Done !"
        # print str(self.spkrinfo.shape[0]) + " different speakers."
        
        print "Loading speakers list...", 
        self.spkrid = cPickle.load(open(spkrid_path, "r"))
        print "Done !"
        print str(len(self.spkrid)) + " different speakers."
        
        print "Loading speakers list...", 
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
        self.phonemes = np.load(open(phns_path, "r"))
        print "Done !"
        print str(len(self.phonemes)) + " different phonemes."
        
        
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
        raw_wav_path = os.path.join(self.timit_path, subset+"_x_raw.npy")
        phn_path = os.path.join(self.timit_path, subset+"_redux_phn.npy")
        seq_to_phn_path = os.path.join(self.timit_path, \
                                       subset+"_seq_to_phn.npy")
        wrd_path = os.path.join(self.timit_path, subset+"_wrd.npy")
        seq_to_wrd_path = os.path.join(self.timit_path, \
                                       subset+"_seq_to_wrd.npy")
        spkr_path = os.path.join(self.timit_path, subset+"_spkr.npy")
        print "Done !"
        
        # Checking the validity of the paths
        print "Checking path validity...", 
        for p in [raw_wav_path, phn_path, seq_to_phn_path, wrd_path, \
                  seq_to_wrd_path, spkr_path]:
            if not os.path.isfile(p):
                raise IOError(p + " is not a valid path !")
        
        print "Done !"
        
        # Frames
        print "Loading frames...", 
        raw_wav = np.load(raw_wav_path)
        print "Done !"
        print str(raw_wav.shape[0]) + " sentences."
        
        # Side information
        ## Phonemes
        print "Loading phonemes...", 
        phn = np.load(phn_path) 
        seq_to_phn = np.load(seq_to_phn_path)
        print "Done !"
        
        ## Words
        print "Loading words...", 
        wrd = np.load(wrd_path) 
        seq_to_wrd = np.load(seq_to_wrd_path)
        print "Done !"
        
        ## Speaker information
        print "Loading speaker information...", 
        spkr_id = np.asarray(np.load(spkr_path), 'int')
        print "Done !"
        
        
        data = {}
        data[subset+"_raw_wav"] = raw_wav
        data[subset+"_n_seq"] = raw_wav.shape[0]
        data[subset+"_phn"] = phn
        data[subset+"_seq_to_phn"] = seq_to_phn
        data[subset+"_wrd"] = wrd
        data[subset+"_seq_to_wrd"] = seq_to_wrd
        data[subset+"_spkr"] = spkr_id
        
        # Raise the flag advertising the presence of data
        data["has_"+subset] = True
        
        self.__dict__.update(data)
        
        self.sanity_check(subset)
    
    def clear(self, subset):
        """
        Given the subset id, this method will unload the subset from the class. 
        
        """
        self.check_subset_value(subset)
        self.check_subset_presence(subset)
        
        
        del self.__dict__[subset+"_raw_wav"]
        del self.__dict__[subset+"_n_seq"]
        del self.__dict__[subset+"_phn"]
        del self.__dict__[subset+"_seq_to_phn"]
        del self.__dict__[subset+"_wrd"]
        del self.__dict__[subset+"_seq_to_wrd"]
        del self.__dict__[subset+"_spkr"]
        
        # Lower the flag advertising the presence of data
        data["has_"+subset] = False
    
    def check_subset_value(self, subset):
        if subset not in {"train", "valid", "test"}:
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
        for i in range(len(short)):
            if self.__dict__[subset+"_seq_to_"+short[i]][-1,-1] == \
               self.__dict__[subset+"_"+short[i]].shape[0]:
                print "OK for "+long[i]+"."
            else:
                print "KO for "+long[i]+"."
        
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
    
    def get_raw_seq(self, subset, seq_id, framelength, overlap):
        """
        Given an id of the sequence, this method will return a frames sequence 
        from a given set, the associated phoneme sequence and the 
        information vector on the speaker.
        
        """
        self.check_subset_value(subset)
        self.check_subset_presence(subset)
        
        # Check if the id is valid
        n_seq = self.__dict__[subset+"_n_seq"]
        if seq_id >= n_seq:
            raise ValueError("This sequence does not exist.")
        
        # Get the sequence
        wav_seq = self.__dict__[subset+"_raw_wav"][seq_id]
        
        # Get the phonemes
        phn_l_start = self.__dict__[subset+"_seq_to_phn"][seq_id][0]
        phn_l_end = self.__dict__[subset+"_seq_to_phn"][seq_id][1]
        phn_start_end = self.__dict__[subset+"_phn"][phn_l_start:phn_l_end]
        phn_seq = np.zeros_like(wav_seq)
        for (phn_start, phn_end, phn) in phn_start_end:
            phn_seq[phn_start:phn_end] = phn
        
        # Get the words
        wrd_l_start = self.__dict__[subset+"_seq_to_wrd"][seq_id][0]
        wrd_l_end = self.__dict__[subset+"_seq_to_wrd"][seq_id][1]
        wrd_start_end = self.__dict__[subset+"_wrd"][wrd_l_start:wrd_l_end]
        wrd_seq = np.zeros_like(wav_seq)
        # Some timestamp does not correspond to any word so 0 is 
        # the index for "NO_WORD" and the other index are shifted by one
        for (wrd_start, wrd_end, wrd) in wrd_start_end:
            wrd_seq[wrd_start:wrd_end] = wrd+1
        
        # Find the speaker id
        spkr_id = self.__dict__[subset+"_spkr"][seq_id]
        # Find the speaker info
        spkr_info = self.spkrinfo[spkr_id]
        
        # Segment into frames
        wav_seq = segment_axis(wav_seq, framelength, overlap)
        
        # Take the most occurring phoneme in a sequence
        phn_seq = segment_axis(phn_seq, framelength, overlap)
        phn_seq = scipy.stats.mode(phn_seq, axis=1)[0].flatten()
        phn_seq = np.asarray(phn_seq, dtype='int')
        
        # Take the most occurring word in a sequence
        wrd_seq = segment_axis(wrd_seq, framelength, overlap)
        wrd_seq = scipy.stats.mode(wrd_seq, axis=1)[0].flatten()
        wrd_seq = np.asarray(wrd_seq, dtype='int')
        
        return [wav_seq, phn_seq, wrd_seq, spkr_info]
    
    def get_n_seq(self, subset):
        """
        Given the subset id, return the number of sequence in it.
        
        """
        self.check_subset_value(subset)
        self.check_subset_presence(subset)
        return self.__dict__[subset+"_n_seq"]
    
