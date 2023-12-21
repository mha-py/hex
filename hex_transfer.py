'''
File for communication between scripts via files.
(c) mha 10.4.2021
'''
import os
import pickle as pkl
from datetime import datetime
from time import sleep
import torch



def timenow():
    # current date and time
    now = datetime.now()
    return now.strftime("%H:%M:%S")


lastloaded = 0
def check_new_model(net, model_fname, verbose=1, harderror=False):
    '''Loads the most recent net parameters into the net'''
    global lastloaded
    time = os.path.getmtime(model_fname)
    if time != lastloaded:
        try:
            net.load_state_dict(torch.load(model_fname))
            lastloaded = time
            if verbose:
                print(f'Loaded new model parameters at time {timenow()}!')
        except Exception as error:
            if verbose:
                print(f'Couldnt load new model parameters at time {timenow()}!')
            if harderror:
                raise Exception('Couldnt load new model parameters!').with_traceback(error.__traceback__)
            



def save_selfplay(game_records, fnmask='temp/selfplay%d.dat'):
    for i in range(1000):
        fn = fnmask % i
        if not os.path.exists(fn):
            break
    with open(fn, 'wb') as f:
        pkl.dump(game_records, f)
    print(f'Saved {len(game_records)} boards to {fn}.')
    
    


def load_selfplay(numacquire = 0, fnmask='temp/selfplay%d.dat', verbose=1):
    '''Loads the most recent net parameters into the net'''
    
    if verbose:
        if numacquire > 0:
            print(f'Acquiring min. {numacquire} board states!')
        else:
            print(f'Acquiring board states!')
        
    records = []
    while True:
        for i in range(100):
            fn = fnmask % i
            if os.path.exists(fn):
                try:
                    with open(fn, 'rb') as f:
                        newrecords = pkl.load(f)
                        records += newrecords
                        print(f'Loaded {len(newrecords)} boards from {fn}.')
                    os.remove(fn)
                except EOFError as e:
                    print(f'Removing {fn} since eof exception!')
                    try: os.remove(fn)
                    except: print('Couldnt remove')
        if len(records) >= numacquire:
            return records
        else:
            sleep(1)
            
            
            
            