Audio classifier
================
A tool for audio classification based on the fourier transformation.
It can currently separate two audio classes, one for samples stored in folder `speech_samples` and one stored in
folder `body_sound_samples`.

The training samples that were used to train the classifier stored in `best_clf.pkl` can't be published due to
licensing issues.

Usage
-----
```
Usage: classifier.py optimize [OPTIONS]

  Use hyper parameter optimization to create an optimal classifier
  parameters and store it. It does the optimization in parallel and uses a
  tmp file for the purpose of communicating the currently best classifier.

Options:
  --dest TEXT              File to store the classifier and the scaler in
  --train_samples INTEGER  Number of samples to use for training
  --test_samples INTEGER   Number of samples to use for testing
  --test_runs INTEGER      Number of independent train / validation runs for
                           each set of parameters
  --iterations INTEGER     Number of different parameter sets that are
                           examined
  --help                   Show this message and exit.
```

```
Usage: classifier.py classify [OPTIONS] INPUT OUTPUT

  Removes or silences the rubbish noise from the input wav file using the
  classifier and outputs the classification results for each sub sample. It
  is more complex than the classify2 method because it does more
  classifications and fading.

Options:
  --clf_file TEXT       File the classifier, scaler and options are stored in
  --rubbish_file TEXT   File to store the rubbish audio in
  --silence / --remove  Silence or remove the rubbish
  --granularity FLOAT   Time between two classified samples in seconds
  --help                Show this message and exit.
```
Dependencies
------------
- unix (needs the `/tmp` folder)
- python3 (>= 3.4)
- click
- optunity
- numpy, scipy
- sklearn


License
-------
MIT