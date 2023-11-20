from PIL import Image
from PIL import ImageTk
import tkinter as tk
import numpy as np
import cv2 as cv
import neurokit2 as nk
import pandas as pd
import tsfel
import os
from os import listdir
from clustering import Cluster
import matplotlib.pyplot as plt

class Process:
    def __init__(self, images, muscles, muscle_activity="mle"):
        # Member variables to store images, muscles and activity
        self.images = images
        self.muscles = muscles
        self.muscle_activity = muscle_activity

        # Resize all images to a common size
        self.__resize_images(1617, 590)

        # Perform all necessary processing steps
        subject_images = self.__extract_individual_images()
        subject_signals = self.__extract_signals(subject_images)

        # Save overlapped signals
        self.__show_overlapping_signals(subject_signals)

        subject_features = self.__extract_signal_features(subject_signals)
        subject_combined_dataset = self.__form_combined_dataset(
            subject_features)
        subject_combined_signals = self.__get_all_signals(subject_signals)

        # Perform clustering
        self.__perform_clustering(subject_combined_signals,
                                  subject_combined_dataset)
        print("Finished processing")

    def __resize_images(self, max_width, max_height):
        new_size = (max_width, max_height)
        for image in self.images:
            image.thumbnail(new_size, Image.ANTIALIAS)

    def __extract_individual_images(self):
        num_images = len(self.muscles)
        pixels_left_to_truncate = 85
        top_offset = 20
        image_height = 89 - top_offset
        start = top_offset

        images = [np.array(each)[:, :, :3] for each in self.images]

        # Obtain b/w image for each muscle for each subject
        subject_images = []
        for image in images:
            # Convert img to black and white after edge detection
            min_thres = np.mean(image) * 0.66
            max_thres = np.mean(image) * 1.33
            bw_img = cv.Canny(image, min_thres, max_thres)

            signal_images_per_subject = []
            for _ in range(num_images):
                single_signal_image = bw_img[start:start + image_height,
                                             (pixels_left_to_truncate - 1):]
                signal_images_per_subject.append(single_signal_image)
                start = start + image_height + top_offset
            subject_images.append(signal_images_per_subject)

            # Reset start
            start = top_offset

        # for j, images in enumerate(subject_images):
        #     for i, image in enumerate(images):
        #         pil_image = Image.fromarray(image)
        #         pil_image.save(f'{j+1}_{i+1}.jpg', dpi=(300, 300))

        return subject_images

    def __extract_signal_points(self, test_image):
        cols = test_image.shape[1]
        rows = test_image.shape[0]

        signal_points = []
        for col in range(cols):
            row_values = []
            for row in range(rows-1, -1, -1):
                pixel_value = test_image[row][col]
                if pixel_value == 255:
                    row_values.append(row)

            if len(row_values) > 0:
                signal_points.append((rows - row_values[-1]))
            else:
                signal_points.append(0)

        signal_points = np.array(signal_points)
        baseline_correction_value = np.bincount(signal_points).argmax()
        signal_points = signal_points - baseline_correction_value
        signal_points[signal_points < 0] = 0

        return signal_points

    def __extract_signals(self, subject_images):
        # Extract all signals
        extracted_signals_for_subject = []
        for images in subject_images:
            subject_signals = []
            for image in images:
                temp_signal = self.__extract_signal_points(image)

                # Calculate the moving average (keeping same output size)
                moving_averaged_signal = np.convolve(temp_signal, np.ones(5)/5,
                                                     mode='valid')

                # Preprocess moving averaged signal
                processed_data, _ = nk.bio_process(emg=moving_averaged_signal)
                clean_signal = processed_data["EMG_Clean"].values

                # Add signal to subject
                subject_signals.append(clean_signal)
            extracted_signals_for_subject.append(subject_signals)

        return extracted_signals_for_subject

    # Function to normalize a signal to within the range [0, 1]
    def __normalize(self, sig, upper_limit=1, lower_limit=-1):
        maximum = np.max(sig)
        minimum = np.min(sig)

        if (maximum - minimum) > 0:
            normalized = [(((upper_limit - lower_limit)*(point - minimum)) /
                           (maximum - minimum)) + lower_limit for point in sig]
            return np.array(normalized)
        return sig

    def __extract_signal_features(self, subject_signals):
        subject_signal_features = []
        for signals in subject_signals:
            features_list = []
            for sig in signals:
                normalized_sig = self.__normalize(sig)

                # Extracted features
                all_features_dict = tsfel.get_features_by_domain()
                features = tsfel.time_series_features_extractor(
                    all_features_dict, normalized_sig)
                features_list.append(features)
            # print(len(features_list))
            subject_signal_features.append(features_list)
        return subject_signal_features

    # Function to retrieve all '.csv' signals from specified folder
    def __get_all_signals(self, subject_signals):
        path = f"resources/{self.muscle_activity}_emg"
        directories = sorted(listdir(path=path))
        directory_tuples = sorted(
            [(int(directory.split("_")[1]),
              directory.split("_")[0]) for directory in directories])
        directories = ["_".join([directory_tuple[1], str(directory_tuple[0])])
                       for directory_tuple in directory_tuples]

        files = []
        for directory in directories:
            signals = [(path + "/" + directory + "/" + each)
                       for each in listdir(path + "/" + directory)
                       if ".csv" in each]
            files += signals

        muscle_wise_signals = []
        for i, _ in enumerate(self.muscles):
            file_names = files[i::len(self.muscles)]
            # Storing signals into list
            signals = [pd.read_csv(file_name).values
                       for file_name in file_names]
            # Remove second subject
            signals = [each for (i, each) in enumerate(signals) if not i == 1]
            # Normalizing signals
            signals_reshaped = [each.reshape((each.shape[0],))
                                for each in signals]
            signals_reshaped = [self.__normalize(each)
                                for each in signals_reshaped]
            muscle_wise_signals.append(signals_reshaped)

        # Add current subject signals
        subject_combined_signals = []
        for signals in subject_signals:
            temp_muscle_wise_signals = muscle_wise_signals.copy()
            for i, sig in enumerate(signals):
                correct_length = temp_muscle_wise_signals[i][0].shape[0]
                temp_muscle_wise_signals[i].append(sig[:correct_length])
            subject_combined_signals.append(temp_muscle_wise_signals)

        return subject_combined_signals

    def __form_combined_dataset(self, subject_features):
        path = f"resources/{self.muscle_activity}_with_synthetic"

        subject_datasets = []
        for features_list in subject_features:
            muscle_datasets = []
            for i, features in enumerate(features_list):
                muscle = self.muscles[i]
                file_name = path + f"/{muscle}.csv"
                features_df = pd.read_csv(file_name)

                # Remove first column
                features_df.drop(columns=["Unnamed: 0"], inplace=True)

                columns = set(features.columns)
                existing_columns = set(features_df.columns)
                difference = columns.difference(existing_columns)
                temp_features = features.drop(columns=list(difference)).copy()
                combined_df = pd.concat([features_df, temp_features], axis=0)

                # Append to muscle datasets list
                muscle_datasets.append(combined_df)

            subject_datasets.append(muscle_datasets)
        return subject_datasets

    def __perform_clustering(self, subject_signals,
                             subject_features):
        # Define lists to hold dictionary of quitients
        duration_quotients = []
        intensity_quotients = []

        subject_index = 0
        for signals_list, features_list in zip(subject_signals,
                                               subject_features):
            muscle_index = 0
            dict_duration = {}
            dict_intensity = {}
            for signals, features in zip(signals_list, features_list):
                clustering = Cluster(features, signals,
                                     self.muscles[muscle_index],
                                     self.muscle_activity,
                                     subject_index + 1)
                generated_signals = clustering.get_cluster_signals()

                # Generate quotients for generated
                mean_duration_gen, mean_amp_gen = self.__calculate_quotient(
                    generated_signals[-1])

                # Generate quotients for original
                mean_duration_org, mean_amp_org = self.__calculate_quotient(
                    signals[-1])

                # Obtain differences
                diff_duration = np.abs(mean_duration_gen - mean_duration_org)
                diff_amp = np.abs(mean_amp_gen - mean_amp_org)

                dict_duration[self.muscles[muscle_index]] = [diff_duration]
                dict_intensity[self.muscles[muscle_index]] = [diff_amp]

                muscle_index += 1

            duration_quotients.append(dict_duration)
            intensity_quotients.append(dict_intensity)
            subject_index += 1

            # Write quotients to csv files
            self.__write_quotients(duration_quotients,
                                   intensity_quotients)

    def __write_quotients(self, duration_quotients, intensity_quotients):
        for i, dictionary in enumerate(duration_quotients):
            # Create path if not present
            file_path = f"generated_files/subject_{i+1}/"
            if not os.path.exists(file_path):
                # if the demo_folder directory is not present
                # then create it.
                os.makedirs(file_path)
            df = pd.DataFrame(dictionary)
            df.to_csv(file_path + "quotient_duration.csv",
                      index=False)

        for i, dictionary in enumerate(intensity_quotients):
            # Create path if not present
            file_path = f"generated_files/subject_{i+1}/"
            if not os.path.exists(file_path):
                # if the demo_folder directory is not present
                # then create it.
                os.makedirs(file_path)
            df = pd.DataFrame(dictionary)
            df.to_csv(file_path + "quotient_intensity.csv",
                      index=False)

    def __calculate_quotient(self, sig):
        # Find activity mask
        activity = nk.bio_process(emg=sig)[0]["EMG_Activity"].values
        # Find indices where activity starts and ends
        activity_starts = np.where(np.diff(activity) == 1)[0] + 1
        activity_ends = np.where(np.diff(activity) == -1)[0] + 1

        if len(activity_starts) == 0 and len(activity_ends) == 0:
            return 0, 0

        # Ensure the last activity is included if it extends to the end
        if activity[-1] == 1:
            activity_ends = np.append(activity_ends, len(activity))

        # Calculate duration and amplitude for each activity
        durations = activity_ends - activity_starts
        amplitudes = [np.mean(np.abs(sig[start:end]))
                      for start, end in zip(activity_starts,
                                            activity_ends)]

        # Calculate mean duration and mean amplitude
        mean_duration = np.mean(durations)
        mean_amplitude = np.mean(amplitudes)

        return mean_duration, mean_amplitude

    def __show_overlapping_signals(self, subject_signals):
        muscle_index = 0
        muscle_wise_signals = []
        while muscle_index < len(self.muscles):
            signals_for_one_muscle = []
            for signals in subject_signals:
                signals_for_one_muscle.append(signals[muscle_index])
            muscle_wise_signals.append(signals_for_one_muscle)
            muscle_index += 1
        print("Obtained muscle wise signals")

        print(len(muscle_wise_signals))
        print("----")
        # for i, muscle_signals in enumerate(muscle_wise_signals):
        #     legends = []
        #     for j, sig in enumerate(muscle_signals):
        #         # Perform plotting
        #         fig = plt.figure()
        #         plt.plot(range(len(sig)),
        #                  sig)
        #         plt.xlabel("Samples")
        #         plt.ylabel("Magnitude")
        #         plt.title(self.muscles[i])
        #         legends.append(f"Subject {j+1}")
        #     plt.legend(legends)

        # Plot all subjects for each element
        for i, signals_list in enumerate(muscle_wise_signals):
            plt.figure()
            for j, sig in enumerate(signals_list):
                plt.plot(range(len(sig)), sig,
                         label=f'Subject {j+1}')

            plt.title(self.muscles[i])
            plt.xlabel('Samples')
            plt.ylabel('Magnitude')
            plt.legend()

            # Create path if not present
            file_path = "generated_files/overlapped_signals/"
            if not os.path.exists(file_path):
                # if the demo_folder directory is not present
                # then create it.
                os.makedirs(file_path)
            plt.savefig(file_path + f"{self.muscles[i]}.jpg",
                        dpi=300)
            plt.close()