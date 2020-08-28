# Testing the Diagno Web App

1. Visit [http://diagno-ui.herokuapp.com/]
2. Download the sample recordings from [here](https://github.com/shehanmunasinghe/diagno/blob/master/Sample%20Recordings). (If you have your own ECG recording device you too can generate JSON Files yourselves. See [JSON File Format](#json-file-format) )
3. Click 'Choose File'/'Browse' and upload any of the sample recordings. 
4. You'll see 12 plots for the visualization of 12-lead ECG signals. 
5. Now click 'Analyse'. You'll get the machine-generated predictions !!



## (OPTIONAL) Generating JSON files yourselves using your own ECG Hardware

### JSON File Format
* 12x4096 JSON Array. 4096 samples per each ECG Lead
* Signals must be normalized (Each data value between 0 and 1 (Float32) )
* Sampling rate 400Hz. If the recorded at a different sampling rate, resample to 400Hz

#### Example:

    '[
        [0.08933808901152494, 0.08622472008049079, 0.08748484483946328, ..., 0.08669382432404785, 0.08727538263812228],
        [0.08681173974874626, 0.0872001459641301, 0.08686349999574926, ..., 0.08716319715879413, 0.0868887008560258],
        [0.0843244646650618, 0.08458883624778389, 0.08436451200063877, ..., 0.08456321887653273, 0.08440538068028715],
        [0.07692143752869449, 0.07906505602469656, 0.07945948802502142, ..., 0.0793007000280487, 0.07941703115399325],
        ........................,
        [0.09979861811156633, 0.099748679613499, 0.09982761038949958, ..., 0.09951139681462995, 0.09730531724294086]
        [0.09711884099772175, 0.09732782051344732, 0.09711130487391956,..., 0.0973777123000035, 0.09690846013666106]
    ]'

#### Python Functions to Resample and Normalise
    import numpy as np
    from scipy import signal

    def resample(input_signal, Fs_in, Fs_out):
        # input_signal=[[],[],[],...] # 12xL
        n_samples_in = input_signal.shape[1]
        n_samples_out =  round(n_samples_in * Fs_out/Fs_in)
        
        output_signals=[]
        for i in range(12):
            output_signals.append(signal.resample(input_signal[i], n_samples_out))
            
        return np.vstack(output_signals)

    def normalize(X):
        Xmin=np.amin(X,keepdims=True,axis=1)
        Xmax=np.amax(X,keepdims=True,axis=1)
        return (X-Xmin)/(Xmax-Xmin)



