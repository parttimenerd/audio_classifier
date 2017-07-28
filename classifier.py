import multiprocessing
import random, os
import traceback
from collections import namedtuple, defaultdict

import click
import numpy as np
from typing import List, Dict, Tuple, Set, Iterator, NamedTuple

import time

import optunity
import typing
from scipy.io import wavfile
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


@click.group()
def cli():
    pass

SampleOptions = namedtuple('SampleOptions', ['ffts_per_sample', 'fft_length', 'rel_dist_between_ffts', 'freq_bin_width', 'freq_bin_op'])


class Audio:
    """ Represents an audio file """

    def __init__(self, audio: np.array, rate: int):
        """
        Create an audio object
        :param audio: data array, is set read-only
        :param rate: sample rate
        """
        self.audio = audio  # type: np.array
        self.audio.setflags(write=False)
        self.rate = rate

    def translate_seconds(self, second: float) -> int:
        return int(second * self.rate)

    def translate_ticks(self, ticks: int) -> float:
        return ticks * 1.0 / self.rate

    def max_second(self) -> int:
        return int(len(self.audio) / self.rate)

    def _normalized_audio(self, start: int, end: int) -> np.array:
        """
        Get a sample and normalize the maximum of the its audio level to the maximum of the data type
        :param start: start tick
        :param end: end tick
        :return: normalized sample
        """
        sample = self.audio[start:end]  # type: np.array
        assert len(sample) == end - start
        info = np.iinfo(sample.dtype)
        max_level = max(np.max(sample), -np.min(sample))
        max_achievable_level = min(info.max, -info.min)
        if max_level != 0:
            factor = max_achievable_level / max_level
            sample = np.multiply(factor, sample, casting="unsafe")
        return sample

    def fft(self, start: int, end: int, options: SampleOptions) -> np.array:
        """
        Fourier transform the normalized sample for start to end tick and return it
        :param start: start tick
        :param end: end tick
        :param options: options to configure the frequency binning
        :return: discrete fourier transformation
        """
        sample = self._normalized_audio(start, end)
        ret = []
        for x in np.fft.rfftn(sample):
            ret.append(np.abs(x))
            #ret.append(np.abs(x[1]))
        ret_binned = []
        for i in range(0, len(ret), options.freq_bin_width):
            slice = ret[i:i + options.freq_bin_width]
            if options.freq_bin_op == "mean":
                ret_binned.append(np.mean(slice))
            elif options.freq_bin_op == "max":
                ret_binned.append(np.max(slice))
        return np.array(ret_binned)

    def multiple_fft(self, start: int, end: int, options: SampleOptions) -> np.array:
        """
        Fourier transform a number of normalized sub samples between start and end and concatenate the transformation
        :param start: start tick
        :param end: end tick
        :param options: options to configure the number of sub samples and the fourier transformation
        """
        sample = []
        diff = int((end - start) // options.ffts_per_sample * 1 / options.rel_dist_between_ffts)
        #print(diff)
        for i in range(options.ffts_per_sample):
            s = start + int(i * diff * options.rel_dist_between_ffts)
            sample += self.fft(s, s + diff, options).tolist()
        return np.array(sample)

    def random_audio_samples(self, num: int, options: SampleOptions) -> List[List[float]]:
        """
        Returns the fourier transformations of several random samples from this audio
        :param num: number of fourier transformations to return
        :param options: options for the fourier transformation
        :return: list of vectors
        """
        res = []
        sample_length_in_ticks = self.translate_seconds(options.fft_length)
        for i in range(num):
            vec = None
            while vec is None:
                start = random.randrange(0, len(self.audio) - 2 * sample_length_in_ticks)
                vec = self.multiple_fft(start, start + sample_length_in_ticks, options)
            res.append(vec)
        return res

    def get_all_test_samples(self, options: SampleOptions) -> List[Tuple[float, float, np.array]]:
        """
        Get the fourier transformation of all non overlapping samples of this audio object
        :param options: options to configure the fourier transformation
        """
        res = []
        sample_length_in_ticks = self.translate_seconds(options.fft_length)
        start = 0
        while start < len(self.audio) - 2 * sample_length_in_ticks:
            vec = None
            while vec is None:
                vec = self.multiple_fft(start, start + sample_length_in_ticks, options)
            start += sample_length_in_ticks
            res.append((self.translate_ticks(int(start)), self.translate_ticks(int(start + sample_length_in_ticks)), vec))
        return res

    def get_raw_sample(self, start_second: int, end_second: int) -> np.array:
        return self.audio[self.translate_seconds(start_second):self.translate_seconds(end_second)]

    def classify(self, clf_function: typing.Callable[[np.array], int], options: SampleOptions, increment: float) -> Iterator[List[Tuple[float, float, Dict[int, float]]]]:
        """
        Classifies the whole sample using a sliding window approach (with the passed increment).
        Returns the percentage each label has for each increment long slide on the overall number of classifications.
        :param clf_function: function that gets passed the fourier representation of a vector and returns the predicted category
        :param options: options to configure the fourier transformation
        :param increment: increment in seconds from one sample to the next
        :return: Iterator of (start second, end second, {predicted category: probability}) for each sample
        """
        res = []  # type: List[Tuple[float, float, Dict[int, int]]]
        increment_ticks = self.translate_seconds(increment)
        sample_length_in_ticks = self.translate_seconds(options.fft_length)
        start = 0
        cur_res_index = 0

        def process_res(elem: Tuple[float, float, Dict[int, int]]) -> Tuple[float, float, Dict[int, float]]:
            number_of_clfs = sum(elem[2].values())
            for k in elem[2].keys():
                elem[2][k] /= number_of_clfs

        while start < len(self.audio) - 2 * sample_length_in_ticks:
            vec = None
            while vec is None:
                vec = self.multiple_fft(start, start + sample_length_in_ticks, options)
            start += increment_ticks
            clf_res = clf_function(vec)
            i = 0
            while start + i * increment_ticks < start + sample_length_in_ticks:
                if cur_res_index + i >= len(res):
                    res.append(
                        (self.translate_ticks(int(start + i * increment_ticks)),
                         self.translate_ticks(int(start + (i + 1) * increment_ticks)),
                         defaultdict(lambda: 0)))
                res[cur_res_index + i][2][clf_res] += 1
                i += 1
            cur_res_index += 1
            process_res(res[cur_res_index - 1])
            yield res[cur_res_index - 1]
        for i in range(cur_res_index, len(res)):
            process_res(res[i])
            yield res[i]


class SampleLibrary:
    """ Stores samples from different types """

    def __init__(self, audios_for_categories: List[List[str]], train_sample_num: int, test_sample_sum: int):
        """
        Creates a library object
        :param audios_for_categories: List of audios for each category
        :param train_sample_num: number of samples to use for training
        :param test_sample_sum: number of samples to use for testing
        """
        self.audios_for_categories = []  # type: List[Audio]
        for files in audios_for_categories:
            self._add_category(files)
        self.train_sample_num = train_sample_num
        self.test_sample_num = test_sample_sum

    def _add_category(self, files: List[str]):
        vec = []  # type: List[Audio]
        for file in files:
            rate, data = wavfile.read(file)
            data = data if isinstance(data.T[0], np.int16) else data.T[0]
            audio = Audio(data, rate)
            vec.append(audio)
        self.audios_for_categories.append(vec)

    def get_random_sample_tuples(self, count: int, options: SampleOptions) -> Tuple[List[np.array], np.array]:
        """ Returns (Fourier vectors, corresponding categories) """
        X = []
        y = []
        d = int(count / len(self.audios_for_categories))
        for cat in range(len(self.audios_for_categories)):
            num = min(d, count - (cat * d))
            for i in range(num):
                audio = random.choice(self.audios_for_categories[cat])
                for vec in audio.random_audio_samples(1, options):
                    X.append(vec)
                    y.append(cat)

        return np.array(X), np.array(y)

    def get_train_samples(self, options: SampleOptions) -> Tuple[List[np.array], np.array]:
        return self.get_random_sample_tuples(self.train_sample_num, options)

    def get_test_samples(self, options: SampleOptions) -> Tuple[List[np.array], np.array]:
        return self.get_random_sample_tuples(self.test_sample_num, options)

    def get_train_and_test_samples(self, options: SampleOptions) -> Tuple[List[np.array], np.array, List[np.array], np.array]:
        """
        Returns train vectors, train categories, test vectors, test categories
        """
        X_train, y_train = self.get_random_sample_tuples(self.train_sample_num, options)
        X_test, y_test = self.get_random_sample_tuples(self.test_sample_num, options)
        return X_train, y_train, X_test, y_test


class ClassifierScores:
    """
    Container of cross validated scores of a classifier
    """
    def __init__(self, scores: List[float]):
        self.scores = scores
        self.min = min(scores)
        self.max = max(scores)
        self.std = np.std(scores)
        self.mean = np.mean(scores)
        self.score = self.min

    def __str__(self) -> str:
        return "{:.0%}--{:.0%}--{:.0%} (+/- {:.0%})".format(self.min, self.mean, self.max, self.std)


def wav_files_in_dir(dir: str) -> List[str]:
    """
    Returns the wav files in the passed directory
    """
    return  [os.path.join(root, name)
            for root, dirs, files in os.walk(dir)
            for name in files
            if name.endswith('.wav')]

"""
Audio files to consider for the categories, the index corresponds to the category
"""
AUDIO_FILES = [wav_files_in_dir("speech_samples"), wav_files_in_dir("body_sound_samples")]
RUBBISH_CATEGORY = 1


@cli.command()
@click.option("--dest", default="best_clf.pkl", help="File to store the classifier and the scaler in")
@click.option("--train_samples", type=int, default=1000, help="Number of samples to use for training")
@click.option("--test_samples", type=int, default=10000, help="Number of samples to use for testing")
@click.option("--test_runs", type=int, default=3, help="Number of independent train / validation runs "
                                                       "for each set of parameters")
@click.option("--iterations", type=int, default=100, help="Number of different parameter sets that are examined")
def optimize(dest: str, train_samples: int, test_samples: int, test_runs: int, iterations: int):
    """
    Use hyper parameter optimization to create an optimal classifier parameters and store it.
    It does the optimization in parallel and uses a tmp file for the purpose of communicating the currently
    best classifier.
    """
    tmp_file = "/tmp/classifier{}".format(time.time())

    def read_cur_best_accuracy() -> float:
        if os.path.exists(tmp_file):
            with open(tmp_file, "r") as f:
                return float(f.readline().strip())
        else:
            return 0

    def write_cur_best_accuracy(accuracy: float):
        with open(tmp_file, "w") as f:
            print(accuracy, file=f)

    lib = None
    while lib is None:
        lib = SampleLibrary(AUDIO_FILES, train_samples, test_samples)
    print("Initialized sample library")
    space = {'kernel': {'linear': {'C': [0, 10]},
                        'rbf': {'logGamma': [-5, 0], 'C': [0, 50]},
                        'poly': {'degree': [2, 8], 'C': [0, 50], 'coef0': [0, 10]}
                        },
             'fft_length': [0.005, 0.1],
             'ffts_per_sample': {str(i):None for i in range(1, 10)},
             'interlock': [0.5, 2],
             'freq_bin_width': {str(i):None for i in range(1, 20)},
             'freq_bin_op': {"mean":None, "max":None}
             }

    def score(kernel, C, logGamma, degree, coef0, fft_length, ffts_per_sample, interlock, freq_bin_width, freq_bin_op):
        """A generic SVM training function, with arguments based on the chosen kernel."""
        ffts_per_sample = int(ffts_per_sample)
        model = None
        if kernel == 'linear':
            model = SVC(kernel=kernel, C=C)
        elif kernel == 'poly':
            model = SVC(kernel=kernel, C=C, degree=degree, coef0=coef0)
        elif kernel == 'rbf':
            model = SVC(kernel=kernel, C=C, gamma=10 ** logGamma)
        scores, clf, scaler = None, None, None
        options = SampleOptions(ffts_per_sample, fft_length, interlock, int(freq_bin_width), freq_bin_op)
        while scores is None:
            scores, clf, scaler = score_classifier(lib, model, test_runs, options)
        return scores, clf, scaler, options

    def score_(**args):
        global current_max_score, current_max_param_set
        scores, clf, scaler, options = score(**args)
        current_max_score = read_cur_best_accuracy()
        if scores.score > current_max_score:
            current_max_score = scores.score
            write_cur_best_accuracy(scores.score)
            current_max_param_set = args
            print("Current {}: {}".format(scores, current_max_param_set))
            joblib.dump((clf, scaler, options), dest, compress=9)
        else:
            print("Current {}".format(scores))
        return scores.score

    optimal_configuration, info, solver_info = optunity.maximize_structured(score_, search_space=space, num_evals=iterations,
                                                                            pmap=optunity.parallel.create_pmap(multiprocessing.cpu_count()))
    print(optimal_configuration)
    print(info.optimum)
    print(solver_info)
    print('Solution\n========')
    solution = dict([(k, v) for k, v in optimal_configuration.items() if v is not None])
    print("\n".join(map(lambda x: "%s \t %s" % (x[0], str(x[1])), solution.items())))
    df = optunity.call_log2dataframe(info.call_log)
    print(df.sort_values('value', ascending=False)[:10])


def score_classifier(lib: SampleLibrary, classifier: SVC, train_runs: int, options: SampleOptions) -> Tuple[ClassifierScores, SVC, MinMaxScaler]:
    """
    Score the classifier with the passed fourier transformation option
    """
    print("Aquiring test samples...", end="", flush=True)
    scores = []
    best_clf, best_scaler, scaler = None, None, None
    best_clf_score = 0
    X_test_, y_test = lib.get_test_samples(options)
    for i in range(train_runs):
        error_before = False
        print(" Aquiring samples...", end="", flush=True)
        X_train, y_train = lib.get_train_samples(options)
        scaler = MinMaxScaler(copy=True, feature_range=(0.0, 1.0))
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test_)
        print(" Training...", end="", flush=True)
        classifier.fit(X_train, y_train)
        scores.append(classifier.score(X_test, y_test))
        print(" Testing... ", end="", flush=True)
        if best_clf_score < scores[-1]:
            best_clf_score = scores[-1]
            best_clf = classifier
            best_scaler = scaler
    print()
    return ClassifierScores(scores), best_clf, best_scaler


@cli.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(exists=True, dir_okay=False))
@click.option("--clf_file", default="best_clf.pkl", help="file the classifier, scaler and options are stored in")
def classify2(input: str, output: str, clf_file: str):
    """
    Removes the rubbish noise from the input wav file using the classifier and outputs the classification results
    for each sub sample.
    """
    rate, data = wavfile.read(input, mmap=True)
    audio = Audio(data, rate)
    clf, scaler, options = joblib.load(clf_file)
    red = []
    start_ticks = 0
    for start, end, sample in audio.get_all_test_samples(options):
        sample = scaler.transform(np.array([sample]))
        pred = clf.predict(sample)
        print("{:.3f}-{:.3f}: {}".format(start, end, pred))
        is_rubbish = pred == RUBBISH_CATEGORY
        end_ticks = audio.translate_seconds(end)
        if not is_rubbish:
            red += (data[start_ticks:end_ticks]).tolist()
        start_ticks = end_ticks
    if output:
        print("write wav file")
        red_arr = np.array(red, dtype=data.dtype)
        red_sub_arr = red_arr - data[0:len(red_arr)]
        wavfile.write(output, rate, red_arr)


@cli.command()
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(exists=True, dir_okay=False))
@click.option("--clf_file", default="best_clf.pkl", help="File the classifier, scaler and options are stored in")
@click.option("--rubbish_file", default="", help="File to store the rubbish audio in")
@click.option("--silence/--remove", default=False, help="Silence or remove the rubbish")
@click.option("--granularity", type=float, default=0.001, help="Time between two classified samples in seconds")
@click.option("-v/-nv", default=True, help="Verbose output?")
def classify(input: str, output: str, clf_file: str, rubbish_file: str, silence: bool, granularity: float, v: bool):
    """
    Removes or silences the rubbish noise from the input wav file using the classifier and outputs the classification
    results for each sub sample. It is more complex than the classify2 method because it does more classifications
    and fading.
    """
    rate, data = wavfile.read(input, mmap=True)
    audio = Audio(data, rate)
    clf, scaler, options = joblib.load(clf_file)
    red = []
    red_rubbish = []
    REMOVAL_CATEGORY = RUBBISH_CATEGORY
    start_ticks = 0
    for start, end, pred_dict in audio.classify(lambda vec: (clf.predict(scaler.transform([vec]))[0]), options, granularity):
        if v:
            print("{:.3f}-{:.3f}: P[rubbish] = {:3.0%}".format(start, end, pred_dict[RUBBISH_CATEGORY]))
        end_ticks = audio.translate_seconds(end)
        if output and (pred_dict[RUBBISH_CATEGORY] != 1 or silence):
            red += (data[start_ticks:end_ticks] * (1 - pred_dict[RUBBISH_CATEGORY])).tolist()
        if rubbish_file and (pred_dict[0] != 1 or silence):
            red_rubbish += (data[start_ticks:end_ticks] * (1 - pred_dict[0])).tolist()
        start_ticks = end_ticks
    print("write wav file")
    if output:
        red_arr = np.array(red, dtype=data.dtype)
        wavfile.write(output, rate, red_arr)
    if rubbish_file:
        red_arr = np.array(red_rubbish, dtype=data.dtype)
        wavfile.write(rubbish_file, rate, red_arr)

if __name__ == "__main__":
    cli()
