import os
import random
import csv
import gc

from scipy.io import wavfile
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split, KFold
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import h5py

# In your code, before creating any TensorFlow operations
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        print("Setting GPU usage")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU Name:", gpu.name)
            print("GPU Device Type:", tf.test.gpu_device_name())
            print("GPU Device name:", tf.config.experimental.get_device_details(gpu)['device_name'])
            print("GPU device compute capability:", tf.config.experimental.get_device_details(gpu)['compute_capability'])
    except RuntimeError as e:
        print(e)

# import wandb
# from wandb.keras import WandbCallback

from models.load_model import load_model

# Load model, including pre trained weights and compile
model_input = 1953
print(f"\nLoad FCN model: {model_input}")
model = load_model(model_input, FULLCONV=False, training=True)

# Initialize W&B
# wandb.init(project='fcn_retraining', resume=True)
# wandb_callback = WandbCallback()

# Set up early stopping callback
early_stopping = EarlyStopping(monitor='loss', patience=32, min_delta=0.001)

# Set up ModelCheckpoint callback to save the best weights
checkpoint_path = f"models/FCN_1953/new_weights.h5"
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss',
                             save_best_only=True, mode='min',
                             restore_best_weights=True)

batch_size = 32
steps_per_epoch = 500
epochs = 500
model_srate = 8000.
step_size = 2.902
hop_length = int(model_srate * step_size / 1000)
num_folds = 5
audio_files_used_to_train = "audio_files_used_to_train.csv"
audio_list = "audio_list.csv"
retry = 1

audio_folder = 'MDB-stem-synth/audio_stems'
annotation_folder = 'MDB-stem-synth/annotation_stems'
frames_target_folder = "D:/MDB-stem-synth/frames_targets"


def test_save_frames_annotations():
    audio_files_list = pd.read_csv(audio_list, header=None, names=['audio_name'])['audio_name'].values
    num_files = len(audio_files_list)

    for idx, audio_file in enumerate(audio_files_list):
        audio_name = audio_file.replace(".wav", "")
        audio_path = os.path.join(audio_folder, audio_file)
        annotation_path = os.path.join(annotation_folder, audio_file.replace(".wav", ".csv"))

        print(f"\nLoading audio: {audio_file} ({idx + 1}/{num_files})")
        audio_data = get_audio(audio_path, model_input)

        print("Loading frequency annotations")
        annotations = pd.read_csv(annotation_path, header=None, names=['timestamp', 'frequency'])

        # Split audio data into frames of 1953 samples
        print("Splitting audio in frames")
        frames = librosa.util.frame(audio_data, model_input, hop_length, axis=0)
        del audio_data

        # Filter frames to get correct annotation for each one
        print("Sync frames with time annotations")
        times = annotations['timestamp'].values
        frame_indexes = librosa.time_to_frames(times, model_srate, hop_length)
        frames = frames[frame_indexes, :]

        # normalize each frame -- this is expected by the model
        print("Normalizing frames")
        frames -= np.mean(frames, axis=1)[:, np.newaxis]
        frames /= np.clip(np.std(frames, axis=1)[:, np.newaxis], 1e-8, None)

        # Get vector targets for each frequency from annotation
        print("Get vector targets for each frequency from annotation")
        frequencies = annotations['frequency'].values
        y_vectors = f0_to_target_vector(frequencies)

        # Create or open the HDF5 file
        with h5py.File(os.path.join(frames_target_folder, "data.h5"), 'a') as hf:
            # Create a group for each audio file (if it doesn't exist)
            if audio_name not in hf:
                group = hf.create_group(audio_name)
            else:
                group = hf[audio_name]

            # Save frames and target vectors to the group
            print("Saving frames and target vector to hdf5 file")
            group.create_dataset('frames', data=frames)
            group.create_dataset('y_vectors', data=y_vectors)


def test_save_frames_annotations_v2():
    audio_files_list = pd.read_csv(audio_list, header=None, names=['audio_name'])['audio_name'].values
    num_files = len(audio_files_list)

    for idx, audio_file in enumerate(audio_files_list):
        audio_path = os.path.join(audio_folder, audio_file)
        annotation_path = os.path.join(annotation_folder, audio_file.replace(".wav", ".csv"))

        print(f"\nLoading audio: {audio_file} ({idx + 1}/{num_files})")
        audio_data = get_audio(audio_path, model_input)

        print("Loading frequency annotations")
        annotations = pd.read_csv(annotation_path, header=None, names=['timestamp', 'frequency'])

        # Split audio data into frames of 1953 samples
        print("Splitting audio in frames")
        frames = librosa.util.frame(audio_data, model_input, hop_length, axis=0)
        del audio_data

        # Filter frames to get correct annotation for each one
        print("Sync frames with time annotations")
        times = annotations['timestamp'].values
        frame_indexes = librosa.time_to_frames(times, model_srate, hop_length)
        frames = frames[frame_indexes, :]

        # normalize each frame -- this is expected by the model
        print("Normalizing frames")
        frames -= np.mean(frames, axis=1)[:, np.newaxis]
        frames /= np.clip(np.std(frames, axis=1)[:, np.newaxis], 1e-8, None)

        # Get vector targets for each frequency from annotation
        print("Get vector targets for each frequency from annotation")
        frequencies = annotations['frequency'].values
        y_vectors = f0_to_target_vector(frequencies)

        # Permutation of frames and target vectors
        print("Permutation of frames and target vectors")
        permutation = np.random.permutation(len(frames))
        frames = frames[permutation]
        y_vectors = y_vectors[permutation]

        # Append frames to h5 file
        print("Append frames to h5 file")
        file = os.path.join(frames_target_folder, "dataset.h5")
        append_audio_frames_to_h5(frames, y_vectors, file)


def test_save_frames_annotations_v3():
    audio_files_list = pd.read_csv(audio_list, header=None, names=['audio_name'])['audio_name'].values
    num_files = len(audio_files_list)

    for idx, audio_file in enumerate(audio_files_list):
        print(f"\nLoading audio: {audio_file} ({idx + 1}/{num_files})")
        frames, y_vectors = load_data_from_hdf5(os.path.join(frames_target_folder, "data.h5"), audio_file.replace(".wav", ""))

        # Append frames to h5 file
        print("Append frames to h5 file")
        file = os.path.join(frames_target_folder, "dataset.h5")
        append_audio_frames_to_h5(frames, y_vectors, file)


def test_train():
    audio_files_list = pd.read_csv(audio_list, header=None, names=['audio_name'])['audio_name'].values
    audio_files_used_list = pd.read_csv(audio_files_used_to_train, header=None, names=['audio_name'])['audio_name'].values

    # Get audio files not used yet
    files_to_train = [audio for audio in audio_files_list if audio not in audio_files_used_list]
    number_audio_files = len(files_to_train)

    # for audio_file in files:
    for audio_file in files_to_train:
        print("\nNumber of audio files remaining:", number_audio_files)
        audio_name = audio_file.replace(".wav", "")

        # Load frames and annotations for training
        print(f"\nLoading audio data: {audio_name}")
        frames, y_vectors = load_data_from_hdf5(os.path.join(frames_target_folder, "data.h5"), audio_name)

        # Create a KFold object
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

        # Loop over folds
        for fold_idx, (train_index, test_index) in enumerate(kf.split(frames)):
            print(f"\nFold {fold_idx + 1}/{num_folds}")

            x_train, x_test = frames[train_index], frames[test_index]
            y_train, y_test = y_vectors[train_index], y_vectors[test_index]

            # Split the training set into train and validation (60/20/20 split)
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

            # Train the model
            try:
                print("Training the model")
                model.fit(
                    x_train, y_train,
                    steps_per_epoch=steps_per_epoch,
                    batch_size=batch_size,
                    validation_data=(x_val, y_val),
                    # callbacks=[early_stopping, checkpoint, wandb_callback]
                    callbacks=[early_stopping, checkpoint]
                    # callbacks=[early_stopping]
                )

                # Evaluate the model on the test set
                print("Evaluating the model")
                test_loss = model.evaluate(x_test, y_test)
                print(f"Test Loss: {test_loss}")
            except:
                raise Exception("Exception during model training")
            finally:
                # Clear the Keras session to release resources
                print("Cleanup")
                tf.keras.backend.clear_session()
                del x_train, x_val, x_test, y_train, y_val, y_test

                # Trigger garbage collection
                gc.collect()

        del frames, y_vectors

        # Save audio file name used to train
        with open(audio_files_used_to_train, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([audio_file])

        number_audio_files -= 1


def test_train_random_batches():
    data_file = os.path.join(frames_target_folder, "data.h5")
    frames_generator = random_frames_batch_generator(data_file)

    # Train the model
    try:
        print("Training the model")
        model.fit(
            frames_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=frames_generator,
            validation_steps=167,
            # callbacks=[early_stopping, checkpoint, wandb_callback]
            callbacks=[early_stopping, checkpoint]
            # callbacks=[early_stopping]
        )

        # Evaluate the model on the test set
        print("Evaluating the model")
        test_loss = model.evaluate(frames_generator, steps=167)
        print(f"Test Loss: {test_loss}")
    except:
        raise Exception("Exception during model training")
    finally:
        # Clear the Keras session to release resources
        print("Cleanup")
        tf.keras.backend.clear_session()

        # Trigger garbage collection
        gc.collect()

def data_generator(audio_files, annotation_files):
    files = list(zip(audio_files, annotation_files))
    random.shuffle(files)

    for audio_file, annotation_file in files:
        audio_path = os.path.join(audio_folder, audio_file)
        annotation_path = os.path.join(annotation_folder, annotation_file)

        print(f"\nLoading audio: {audio_file}")
        audio_data = get_audio(audio_path, model_input)
        annotations = pd.read_csv(annotation_path, header=None, names=['timestamp', 'frequency'])

        # Split audio data into frames of 1953 samples
        print("Splitting audio in frames")
        frames = librosa.util.frame(audio_data, model_input, hop_length, axis=0)

        # Filter frames to get correct annotation for each one
        print("Get frames based on time annotation")
        times = annotations['timestamp'].values
        frame_indexes = librosa.time_to_frames(times, model_srate, hop_length)
        frames = frames[frame_indexes, :]

        # normalize each frame -- this is expected by the model
        print("Normalizing frames")
        frames -= np.mean(frames, axis=1)[:, np.newaxis]
        frames /= np.clip(np.std(frames, axis=1)[:, np.newaxis], 1e-8, None)

        # Get vector targets for each frequency from annotation
        print("Get vector targets for each frequency from annotation")
        frequencies = annotations['frequency'].values
        y_vectors = f0_to_target_vector(frequencies)

        frames_frequencies = list(zip(frames, y_vectors))
        random.shuffle(frames_frequencies)

        total_batches = len(frames_frequencies) // batch_size + 1
        for i in range(0, len(frames_frequencies), batch_size):
            batch_data = frames_frequencies[i:i + batch_size]
            batch_frames, batch_annotations = zip(*batch_data)

            # Get frames batch for training
            yield np.array(batch_frames), np.array(batch_annotations)
            print(f"\nTraining audio: {audio_file}")
            print(f"Batch {i//batch_size + 1} of {total_batches}")


def get_audio(audio_path, model_input_size=993, model_srate=8000.):

    # read sound :
    sr, audio = wavfile.read(audio_path)
    if len(audio.shape) == 2:
        print(f"Making audio mono")
        audio = audio.mean(1)  # make mono
    audio = audio.astype(np.float32)

    sound_duration = len(audio)/sr
    print(f"Duration of sound is {sound_duration}")

    if sr != model_srate: # resample audio if necessary
        print(f"Audio sampling rate: {sr}Hz")
        print(f"Resampling to {model_srate}Hz")
        from resampy import resample
        audio = resample(audio, sr, model_srate)

    # pad so that frames are centered around their timestamps (i.e. first frame is zero centered).
    audio = np.pad(audio, int(model_input_size//2), mode='constant', constant_values=0)

    return audio


def f0_to_target_vector(f0, vecSize = 486, fmin = 30., fmax = 1000., returnFreqs = False) -> np.ndarray:
    '''
    convert from target f0 value to target vector of vecSize pitch classes (corresponding to the values in cents_mapping) that is used as output by the CREPE model
    Unlike the original CREPE model, the first class corresponds to a frequency of 0 (for unvoiced segments).
    If the frequency is 0, all values are 0, except for the 1st value that is = 1.
    For all other cases, the values are gaussian blurred around the target_pitch class, with a maximum value of 1
    :param f0: target f0 value
    :return: target vector of vecSize pitch classes (regularly spaced in cents, from fmin to fmax)
    '''

    fmin_cents = freq2cents(fmin)
    fmax_cents = freq2cents(fmax)
    mapping_cents = np.linspace(fmin_cents, fmax_cents, vecSize)

    # get the idx corresponding to the closest pitch
    f0_cents = freq2cents(f0)

    if isinstance(f0, np.ndarray):
        f0_cents = f0_cents[:, np.newaxis]

    # gaussian-blur the vector auround the taget pitch idx as stated in the paper :
    sigma = 25
    target_vec = np.exp(-((mapping_cents - f0_cents) ** 2) / (2 * (sigma ** 2)))

    if returnFreqs:
        return target_vec, mapping_cents
    else:
        return target_vec


def to_local_average_cents(salience, center=None, fmin=30., fmax=1000., vecSize=486):
    '''
    find the weighted average cents near the argmax bin in output pitch class vector

    :param salience: output vector of salience for each pitch class
    :param fmin: minimum ouput frequency (corresponding to the 1st pitch class in output vector)
    :param fmax: maximum ouput frequency (corresponding to the last pitch class in output vector)
    :param vecSize: number of pitch classes in output vector
    :return: predicted pitch in cents
    '''

    if not hasattr(to_local_average_cents, 'mapping'):
        # the bin number-to-cents mapping
        fmin_cents = freq2cents(fmin)
        fmax_cents = freq2cents(fmax)
        to_local_average_cents.mapping = np.linspace(fmin_cents, fmax_cents, vecSize) # cents values corresponding to the bins of the output vector

    if salience.ndim == 1:
        if center is None:
            center = int(np.argmax(salience)) # index of maximum value in output vector
        start = max(0, center - 4)
        end = min(len(salience), center + 5)
        salience = salience[start:end]
        product_sum = np.sum(
            salience * to_local_average_cents.mapping[start:end])
        weight_sum = np.sum(salience)
        return product_sum / weight_sum
    if salience.ndim == 2:
        return np.array([to_local_average_cents(salience[i, :]) for i in
                         range(salience.shape[0])])

    raise Exception("label should be either 1d or 2d ndarray")


def freq2cents(f0, f_ref=10.):
    """
    Convert a given frequency into its corresponding cents value, according to given reference frequency f_ref
    :param f0: f0 value (in Hz)
    :param f_ref: reference frequency for conversion to cents (in Hz)
    :return: value in cents
    """
    c = 1200 * np.log2(f0/f_ref)
    return c


def calculate_frames(num_samples, frame_size, hop_size):
    num_frames = int((num_samples - frame_size) / hop_size) + 1
    return num_frames


def calculate_total_frames(audio_list, frame_size, hop_size, sr):
    total_frames = 0

    # Iterate through all audio files in the folder
    print("Calculating total number of audio frames of dataset")
    for audio_file in audio_list: # Assuming your files are in WAV format
        audio_path = os.path.join(audio_folder, audio_file)

        # Load audio file and calculate frames
        audio, sr = librosa.load(audio_path, sr=sr)
        num_samples = len(audio)
        frames = calculate_frames(num_samples, frame_size, hop_size)

        # Accumulate the total frames
        total_frames += frames

    return total_frames


def load_data_from_hdf5(hdf5_file, audio_name):
    with h5py.File(hdf5_file, 'r') as hf:
        if audio_name in hf:
            group = hf[audio_name]
            frames = np.array(group['frames'])
            y_vectors = np.array(group['y_vectors'])
            permutation = np.random.permutation(len(frames))
            return frames[permutation], y_vectors[permutation]
        else:
            return None, None


def random_frames_batch_generator(hdf5_file, batch_size=32):
    print("Loading dataset")
    with h5py.File(hdf5_file, 'r') as hf:
        audio_names = list(hf.keys())
        last_selected_indices = {audio_name: [] for audio_name in audio_names}

    while True:
        chosen_audio = np.random.choice(audio_names, size=batch_size, replace=False)
        selected_frames = []
        selected_labels = []

        for audio_name in chosen_audio:
            group = hf[audio_name]
            frames = np.array(group['frames'])
            labels = np.array(group['y_vectors'])

            available_indices = np.setdiff1d(np.arange(len(frames)), last_selected_indices[audio_name])
            if len(available_indices) == 0:
                available_indices = np.arange(len(frames))

            random_index = np.random.choice(available_indices)
            last_selected_indices[audio_name].append(random_index)

            selected_frames.append(frames[random_index])
            selected_labels.append(labels[random_index])

        yield np.array(selected_frames), np.array(selected_labels)


def append_audio_frames_to_h5(frames, labels, filename):
    # Check if the file exists
    if not os.path.isfile(filename):
        # Create a new HDF5 file if it doesn't exist
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset('frames', data=frames, chunks=True, maxshape=(None, *frames.shape[1:]))
            hf.create_dataset('labels', data=labels, chunks=True, maxshape=(None, *labels.shape[1:]))
        print("New HDF5 file created:", filename)

    else:
        # Open HDF5 file in append mode
        with h5py.File(filename, 'a') as hf:
            # Get current length of datasets
            frames_length = hf['frames'].shape[0]
            labels_length = hf['labels'].shape[0]

            # Append frames and labels to the existing datasets
            hf['frames'].resize((frames_length + frames.shape[0]), axis=0)
            hf['frames'][frames_length:] = frames
            hf['labels'].resize((labels_length + labels.shape[0]), axis=0)
            hf['labels'][labels_length:] = labels

        print("Data appended successfully to", filename)


def read_audio_frames_from_h5(filename, index):
    with h5py.File(filename, 'r') as hf:
        # Check if datasets 'frames' and 'labels' exist
        if 'frames' not in hf or 'labels' not in hf:
            raise FileNotFoundError(
                "Dataset 'frames' or 'labels' not found in the specified HDF5 file.")

        # Read frames and labels from the HDF5 file
        frame = hf['frames'][index]
        label = hf['labels'][index]

    return frame, label

def test_save_audio_list():
    audio_files_list = sorted(os.listdir(audio_folder))

    # Write the filenames to a CSV file
    with open('audio_list.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for filename in audio_files_list:
            csv_writer.writerow([filename])


def test_create_h5():
    a = np.array([[1, 2],[3, 4],[5, 6]])
    b = np.array([[7],[8],[9]])

    append_audio_frames_to_h5(a, b, 'test.h5')

    f, l = read_audio_frames_from_h5('test.h5', 0)

    print(f, l)