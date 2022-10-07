import pickle
import pickle5

def dic_save( dic ):
    with open(dic['filename'] + '.pickle', 'wb') as f:
        pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)

def dic_load(filename):
    with open(filename + '.pickle', 'rb') as f:
        return pickle.load(f , encoding = 'latin1' ) #, encoding='bytes' )
        # return pickle.load(f ) #, encoding='bytes' )


def dic_load_new(filename):
    with open(filename + '.pickle', 'rb') as f:
        return pickle5.load(f , encoding='bytes' )