#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 19:20:29 2017

@author: jbraga
"""

import librosa as lr
import numpy as np
import scipy.io.wavfile as wav
import csv
import math
import mir_eval
import pandas as pd
import glob
import os


def list_of_fragments(dataset_file, filter_file="all"):
    fragments = []
    cr = csv.reader(open(dataset_file))
    for row in cr:
        fragments.append(row[0])

    if filter_file != "all":
        fragments = [fragment for fragment in fragments
                     if filter_file in fragment]

    return fragments


def score(filename, extension):
    version = filename.split('_')[-1]
    file = filename.split(f"_{version}")[0]

    notes = []
    onset = []
    articulation = []
    articulation_onset = []
    grace_notes = []
    grace_onset = []
    grace_diff = []
    tempo = []
    tempo_onset = []

    with open(f"{file}.{extension}") as csvfile:
        notes_reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        #        onset.append('0.0')
        #        notes.append('0')
        for row in notes_reader:
            if row[1] == 'tempo':
                tempo.append(float(row[2]) / 4)
                tempo_onset.append(row[0])
            if row[1] == 'script':
                articulation_onset.append(row[0])
                articulation.append(row[2])
            #            if row[1] =='breathe':
            #                instants.append(row[0])
            #                events.append('breathe')

            if row[1] == 'note' or row[1] == 'rest':
                if row[0].find('-') > 0:
                    aux1, aux2 = row[0].split('-')
                    grace_onset.append(aux1)
                    grace_diff.append(aux2)
                    grace_notes.append(row[2])

                elif row[2].find('/') > 0:
                    onset.append(row[0])
                    notes.append('0')
                else:
                    onset.append(row[0])
                    if row[1] == 'rest':
                        notes.append('0')
                    else:
                        notes.append(row[2])

    tempo = np.array(tempo, dtype='float32')
    tempo_onset = np.array(tempo_onset, dtype='float32')

    onset = np.array(onset, dtype='float32')
    notes = np.array(notes, dtype='float32')
    #    hole_note=4*60/tempo[1]
    #    onset=onset*hole_note
    hole_note = 1
    for i in range(0, len(tempo)):
        hole_note = (4 * 60 / float(tempo[i])) / hole_note
        onset[onset >= tempo_onset[i]] = onset[onset >= tempo_onset[
            i]] * hole_note

    if len(articulation) == 0:
        articulation = None
        articulation_onset = None
    else:
        articulation_onset = np.array(articulation_onset, dtype='float32')
        #        articulation_onset=articulation_onset*hole_note
        hole_note = 1
        for i in range(0, len(tempo)):
            hole_note = (4 * 60 / float(tempo[i])) / hole_note
            articulation_onset[articulation_onset >= tempo_onset[i]] = \
                articulation_onset[
                    articulation_onset >= tempo_onset[i]] * hole_note

    if len(grace_notes) == 0:
        grace_onset = None
        grace_diff = None
        grace_notes = None
    else:
        grace_onset = np.array(grace_onset, dtype='float32')
        grace_diff = np.array(grace_diff, dtype='float32')
        grace_notes = np.array(grace_notes, dtype='float32')
        #        grace_onset=grace_onset*hole_note
        hole_note = 1
        for i in range(0, len(tempo)):
            hole_note = (4 * 60 / float(tempo[i])) / hole_note
            grace_onset[grace_onset >= tempo_onset[i]] = \
                grace_onset[grace_onset >= tempo_onset[i]] * hole_note
            grace_diff[grace_onset >= tempo_onset[i]] = \
                grace_diff[grace_onset >= tempo_onset[i]] * hole_note

    if articulation is not None:
        for i in range(0, len(articulation)):
            if articulation[i] == 'fermata':
                tempo_aux = tempo[tempo_onset <= articulation_onset[i]][0]
                #                print tempo_aux
                onset[onset > articulation_onset[i]] = \
                    onset[onset > articulation_onset[i]] + \
                    60 * float(4) / tempo_aux  # one bar of fermata
                grace_onset[grace_onset > articulation_onset[i]] = \
                    grace_onset[grace_onset > articulation_onset[i]] + \
                    60 * float(4) / tempo_aux

    if grace_notes is not None:
        #        print len(onset)
        #        for i in range(len(grace_notes)-1, -1, -1):
        for i in range(0, len(grace_notes)):
            notes = np.insert(notes, np.where(onset == grace_onset[i])[0][0],
                              grace_notes[i])
            onset = np.insert(onset, np.where(onset == grace_onset[i])[0][0],
                              grace_onset[i] - grace_diff[i] / 8)

    duration = onset[1:] - onset[:-1]
    return onset[:-1], notes[:-1], duration


def ground_truth(gt_file, adjust=0):
    cr = csv.reader(open(f"{gt_file}.gt"))
    onset = []
    note = []
    duration = []
    for row in cr:
        onset.append(row[0])
        note.append(row[1])
        duration.append(row[2])

    onset = np.array(onset, 'float')
    note = np.array(note, 'float')
    duration = np.array(duration, 'float')
    offset = onset + duration + adjust

    return onset, offset, note, duration


def gen_ground_truth():
    cr = csv.reader(open("allemande.csv"))
    onset = []
    note = []
    duration = []
    for row in cr:
        onset.append(row[0])
        note.append(row[1])
        duration.append(row[2])

    onset = np.array(onset, 'float')
    note = mir_eval.util.midi_to_hz(np.array(note, 'float'))
    duration = np.array(duration, 'float')
    offset = onset + duration # + 10e-3

    return onset, offset, note, duration

def synth_truth(synth_file):
    cr = csv.reader(open(synth_file))
    onset = []
    note = []
    duration = []
    for row in cr:
        onset.append(row[1])
        note.append(row[4])
        duration.append(row[2])

    onset = np.array(onset, 'float')
    note = np.array(note, 'float')
    note = lr.hz_to_midi(note)
    duration = np.array(duration, 'float')

    return onset, note, duration


def get_audio(audio_file):
    fs, audio_samples = wav.read(audio_file)
    t_ticks = np.arange(audio_samples.shape[0]) / fs

    return audio_samples, t_ticks, fs


def audio_notes(audio_file, annotation, hop_size=.01):
    audio_samples, t_ticks, fs = get_audio(audio_file)
    onsets, offsets, notes, durations = ground_truth(annotation)
    audio_length = audio_samples.shape[0]
    audio_info = (audio_samples, t_ticks, fs)

    audio_secs = audio_length / fs
    hop_length = fs * hop_size

    t_note_formatted = \
        np.linspace(0, audio_secs, math.ceil(audio_length / hop_length))
    notes_formatted = np.zeros([len(t_note_formatted), ], 'float')
    index = np.zeros([len(onsets), ])

    j = 0
    for i in range(0, len(onsets)):
        nuevo = True
        while nuevo and t_note_formatted[j] < onsets[i]:
            j = j + 1
        while j < len(t_note_formatted) and \
                onsets[i] <= t_note_formatted[j] < (onsets[i] + durations[i]):
            notes_formatted[j] = notes[i]
            j = j + 1
        index[i] = j

    return t_note_formatted, notes_formatted, index, audio_info


def generate_data_file(columns, filename, mode='w', header=None):

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Write the zipped data to the CSV file
    with open(filename, mode=mode, newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        if header is not None:
            # Insert header
            csv_writer.writerow(header)

        if columns is not None:
            # Write the zipped data row by row
            zipped_data = zip(*columns)

            for row in zipped_data:
                csv_writer.writerow(row)


def load_crepe_results(filename, delimiter=',', comment='#'):
    est_time, est_freq, confidence = \
        mir_eval.io.load_delimited(
            filename,
            [float, float, float],
            delimiter=delimiter,
            comment=comment)

    est_time = np.array(est_time)
    est_freq = np.array(est_freq)
    confidence = np.array(confidence)

    return est_time, est_freq, confidence


def load_mir_results(filename, delimiter=',', comment='#'):
    f = mir_eval.io.load_delimited(
        filename, [float, float, float, float, int, int],
        delimiter=delimiter,
        comment=comment)

    return f


def load_raw_results(filename, delimiter=',', comment='#'):
    f = mir_eval.io.load_delimited(
        filename, [float, float, float, float, int, int],
        delimiter=delimiter,
        comment=comment)

    return f


def merge_csv_files(filename):

    # Get a list of all CSV files with similar names
    csv_files = glob.glob(f'./{filename}/{filename}_*_mir_eval.csv')

    # Initialize an empty list to store DataFrames
    dfs = []

    # Read each CSV file into a DataFrame and append it to the list
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)

    # Concatenate all DataFrames in the list vertically
    merged_df = pd.concat(dfs, ignore_index=True)

    # Save the merged DataFrame to a new CSV file
    os.makedirs(os.path.dirname(f"./mir_eval/"), exist_ok=True)
    file_merged = f'{filename}_mir_eval_merged.csv'
    merged_df.to_csv(f"./mir_eval/{file_merged}", index=False)

    print(f"MIR results saved as '{file_merged}'")


if __name__ == "__main__":
    train, test = list_of_fragments('sequenza')
