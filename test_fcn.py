import numpy as np
import pandas as pd
from prediction import run_prediction
import load as ld
import mir_eval
import os
import matplotlib.pyplot as plt


root_dataset = 'C:\\Users\\fabio\\Flute Dataset\\traditional-flute-dataset\\'
dataset_file = f"{root_dataset}dataset.csv"
audio_path = f"{root_dataset}\\audio\\"
ground_truth_path = f"{root_dataset}\\ground_truth\\"
score_path = f"{root_dataset}\\score\\"


def test_trial():
    file = "examples/DTB/manual/f0_corrected/4.f0_corrected.sdif"

    # Open the SDIF file in binary mode
    with open(file, 'rb') as file:
        # Read binary data from the file
        sdif_data = file.read()

    # You can then use numpy to manipulate the data as needed
    np_data = np.frombuffer(sdif_data, dtype=np.float32)
    print(np_data)


def test_run_prediction():
    file = "examples/DTB/manual/audio/8kHz/4.orig-8kHz.wav"

    run_prediction([file], output="./fcn_results/", modelTag="1953")


def test_fragment():
    evaluate(filename='allemande_first_fragment_nicolet',
             confidence_levels=np.linspace(0, 0.9, 10), folder="before", force_run=True)


def test_all():
    evaluate(confidence_levels=np.linspace(0, 0.9, 10), folder="before")


def test_print_mir_results():
    save_plots()


def test_print_mir_results_rpa_comparison():
    save_compare_rpa_plots()


def test_print_mir_fragment_results():
    save_plots('allemande_first_fragment_nicolet')


def evaluate(filename="all",
             folder="before",
             use_confidence_level=True,
             confidence_levels=(0.0,),
             tol=50,
             only_non_zeros=True,
             force_run=False):

    fragments = ld.list_of_fragments(dataset_file, filter_file=filename)

    for file in fragments:
        print("\n======================================")
        print(f"Test file: {file}")

        # Get original dataset annotation file
        annotation = f"{ground_truth_path}{file}"
        onset, offset, freq, _ = ld.ground_truth(annotation, 10e-3)

        # Generate new annotation from original annotation file
        ann_file = f"./annotations/{file}_annotation.csv"
        ld.generate_data_file((onset, offset, freq), ann_file)

        # Load intervals from new annotation
        intervals, values = mir_eval.io.load_valued_intervals(ann_file, ',')

        # Get annotation for each time point
        ref_time, ref_freq = mir_eval.util.intervals_to_samples(
            intervals, values, sample_size=1e-3)
        ref_time = np.array(ref_time, dtype=float)
        ref_freq = np.array(ref_freq, dtype=float)

        # Reference correction based on crepe results
        if "fragment" in file:
            # Get annotation for each time point
            _, ref_freq_crepe = mir_eval.util.intervals_to_samples(
                intervals, values, sample_size=10e-3)
            ref_freq_crepe = np.array(ref_freq_crepe, dtype=float)

            result_file = f"{file}.f0.csv"
            _, est_freq_crepe, confidence_crepe = read_crepe_results(
                result_file)

            rf = ref_freq_crepe[np.argmax(confidence_crepe)]
            ef = est_freq_crepe[np.argmax(confidence_crepe)]
            ratio = rf / ef
            ref_freq /= ratio

        # Create a new mir_eval file
        mir_eval_file = f"./mir_results/{file}_mir_results.csv"
        # header = ["recall_rate", "voicing_false_alarm", "overall_accuracy",
        #           "raw_pitch_accuracy", "tol", "confidence"]
        # ld.generate_data_file(columns=None, header=header,
        #                       filename=mir_eval_file)
        if os.path.exists(mir_eval_file):
            os.remove(mir_eval_file)

        # Get fcn estimation
        est_time, est_freq, confidence = get_fcn_estimation(
            file, force_run=force_run, output=f"./fcn_results/{folder}")

        for conf_level in confidence_levels:
            if use_confidence_level:
                est_voicing = (confidence >= conf_level).astype(float)
                est_freq[est_voicing == 0] = 0

            else:
                est_voicing = confidence

            ref_v, ref_c, est_v, est_c = mir_eval.melody.to_cent_voicing(
                ref_time, ref_freq, est_time, est_freq, est_voicing)

            recall_rate, voicing_false = \
                mir_eval.melody.voicing_measures(ref_v, est_v)
            overall_accuracy = \
                mir_eval.melody.overall_accuracy(ref_v, ref_c, est_v, est_c)

            raw_pitch_accuracy = mir_eval.melody.raw_pitch_accuracy(
                ref_v, ref_c, est_v, est_c, cent_tolerance=tol,
                only_non_zeros=only_non_zeros)

            # Save mir eval results
            conf_level = str(int(conf_level * 100))
            ld.generate_data_file(([recall_rate], [voicing_false],
                                   [overall_accuracy], [raw_pitch_accuracy],
                                   [tol], [conf_level]), mir_eval_file, 'a')

            # Save frequencies reference and estimation on same
            raw_results = f"./raw_results/{file}_{conf_level}.csv"
            header = ["time", "ref_freq", "est_freq", "confidence"]
            ld.generate_data_file((est_time, ref_freq, est_freq, confidence),
                                  raw_results, header=header)

            print(f"\nConfidence: {conf_level}")
            print(f"Tolerance: {tol}")
            print(f"Recall Rate: {recall_rate}")
            print(f"Voicing False Alarm: {voicing_false}")
            print(f"Overall Accuracy: {overall_accuracy}")
            print(f"Raw Pitch Accuracy: {raw_pitch_accuracy}")


def get_fcn_estimation(file, output="./fcn_results", delimiter="   ", force_run=False):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fcn_results = f"{output}/{file}.f0.csv"

    if os.path.exists(fcn_results) is False or force_run:
        filepath = f"{audio_path}{file}.wav"
        run_prediction([filepath], output=output, modelTag="1953", verbose=False)

    est_time, est_freq, confidence = ld.load_crepe_results(
        fcn_results, delimiter=delimiter)

    return est_time, est_freq, confidence


def save_plots(filter_file="all"):
    fragments = ld.list_of_fragments(dataset_file, filter_file=filter_file)

    for index, file in enumerate(fragments):
        recall_rate, voicing_false_alarm, overall_accuracy, \
            raw_pitch_accuracy, tolerance, confidence = ld.load_mir_results(
                f'./mir_results/{file}_mir_results.csv')

        print(f"Plotting {file} ({index + 1})")
        plt.figure()

        plt.plot(confidence, recall_rate, label='Recall rate', color='blue',
                 linestyle='-')
        plt.plot(confidence, voicing_false_alarm, label='Voicing False Alarm',
                 color='green', linestyle='--')
        plt.plot(confidence, raw_pitch_accuracy, label='RPA', color='red',
                 linestyle='-.')

        # _, _, conf = get_fcn_estimation(file)
        # max_confidence = round(np.max(conf), 2)

        plt.xlabel('Confidence')
        plt.ylabel('Results')
        plt.title(f"{file}")
        plt.legend()

        for c, rr in zip(confidence, recall_rate):
            plt.text(c, rr, f'{round(rr, 2)}', ha='left', va='bottom')

        for c, vfa in zip(confidence, voicing_false_alarm):
            plt.text(c, vfa, f'{round(vfa, 2)}', ha='right', va='top')

        for c, rpa in zip(confidence, raw_pitch_accuracy):
            plt.text(c, rpa, f'{round(rpa, 2)}', ha='center', va='bottom')

        # # Display the plot
        # plt.show()

        # Save the plot as a PNG file
        plt.savefig(f'{file}.png')

        # Close the plot window (optional)
        plt.close()


def save_compare_rpa_plots(filter_file="all"):
    fragments = ld.list_of_fragments(dataset_file, filter_file=filter_file)

    for index, file in enumerate(fragments):
        _, _, _, raw_pitch_accuracy_before, _, confidence = ld.load_mir_results(
            f'./mir_results/before/{file}_mir_results.csv')

        _, _, _, raw_pitch_accuracy_after, _, _ = ld.load_mir_results(
            f'./mir_results/after/{file}_mir_results.csv')

        print(f"Plotting {file} ({index + 1})")
        plt.figure()

        plt.plot(confidence, raw_pitch_accuracy_before, label='RPA before',
                 color='blue', linestyle='-')
        plt.plot(confidence, raw_pitch_accuracy_after, label='RPA after',
                 color='green', linestyle='--')

        # yticks_values = np.arange(0, 1.1, 0.1)
        # plt.yticks(yticks_values)

        # _, _, conf = get_fcn_estimation(file)
        # max_confidence = round(np.max(conf), 2)

        plt.xlabel('Confidence')
        plt.ylabel('RPA results')
        plt.title(f"{file}")
        plt.legend()

        for c, rr in zip(confidence, raw_pitch_accuracy_before):
            plt.text(c, rr, f'{round(rr, 2)}', ha='left', va='bottom')

        for c, vfa in zip(confidence, raw_pitch_accuracy_after):
            plt.text(c, vfa, f'{round(vfa, 2)}', ha='right', va='top')

        # # Display the plot
        # plt.show()

        # Save the plot as a PNG file
        plt.savefig(f'{file}.png')

        # Close the plot window (optional)
        plt.close()


def read_crepe_results(file_name, folder_path="./crepe_results_10ms"):
    file_path = os.path.join(folder_path, file_name)

    # Read the CSV file using pandas
    df = pd.read_csv(file_path, header=None, names=["time", "frequency", "confidence"])

    # Extract values as separate arrays
    time = df["time"].to_numpy()
    frequency = df["frequency"].to_numpy()
    confidence = df["confidence"].to_numpy()

    return time, frequency, confidence
