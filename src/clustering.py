from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
import neurokit2 as nk


class Cluster:
    def __init__(self, df, signals,
                 muscle, muscle_activity, subject_number):
        # Initialize member attributes
        self.df = df
        self.signals = signals
        self.muscle = muscle
        self.muscle_activity = muscle_activity
        self.subject_number = subject_number
        self.cluster_signals = None

        # Cluster parameter attributes
        self.parameter_dict = {
            "mmo": {
                "ta_r": {
                    "model": "gmm",
                    "parameters": [2]
                },
                "ta_l": {
                    "model": "dbscan",
                    "parameters": [13.8, 51]
                },
                "mm_r": {
                    "model": "dbscan",
                    "parameters": [13.8, 51]
                },
                "mm_l": {
                    "model": "gmm",
                    "parameters": [2]
                },
                "da_r": {
                    "model": "kmeans",
                    "parameters": [2]
                },
                "da_l": {
                    "model": "dbscan",
                    "parameters": [13.8, 101]
                }
            },
            "mle": {
                "ta_r": {
                    "model": "gmm",
                    "parameters": [3]
                },
                "ta_l": {
                    "model": "gmm",
                    "parameters": [3]
                },
                "mm_r": {
                    "model": "dbscan",
                    "parameters": [13.8, 91]
                },
                "mm_l": {
                    "model": "dbscan",
                    "parameters": [13.8, 71]
                },
                "da_r": {
                    "model": "dbscan",
                    "parameters": [13.8, 51]
                },
                "da_l": {
                    "model": "dbscan",
                    "parameters": [13.8, 61]
                }
            },
            "map": {
                "ta_r": {
                    "model": "gmm",
                    "parameters": [2]
                },
                "ta_l": {
                    "model": "dbscan",
                    "parameters": [13.8, 61]
                },
                "mm_r": {
                    "model": "dbscan",
                    "parameters": [13.8, 51]
                },
                "mm_l": {
                    "model": "dbscan",
                    "parameters": [13.8, 51]
                },
                "da_r": {
                    "model": "kmeans",
                    "parameters": [3]
                },
                "da_l": {
                    "model": "gmm",
                    "parameters": [2]
                }
            },
            "chewing": {
                "ta_r": {
                    "model": "kmeans",
                    "parameters": [2]
                },
                "ta_l": {
                    "model": "dbscan",
                    "parameters": [13.8, 81]
                },
                "mm_r": {
                    "model": "dbscan",
                    "parameters": [13.8, 51]
                },
                "mm_l": {
                    "model": "kmeans",
                    "parameters": [2]
                },
                "da_r": {
                    "model": "dbscan",
                    "parameters": [13.8, 51]
                },
                "da_l": {
                    "model": "dbscan",
                    "parameters": [13.8, 91]
                }
            },
        }

        # Get PCA reduced data
        reduced_x = self.__perform_initial_steps(self.df)

        # Perform clustering
        labels = self.__perform_clustering(reduced_x)
        self.cluster_signals = self.__generate_cluster_centric_signals(labels)

        # Save generated normal signal
        self.__write_generated_signal_for_subject(
            self.cluster_signals[-1])

    def __perform_initial_steps(self, df):
        temp_df = df.copy()

        # Remove second subject
        temp_df.drop([1001], inplace=True)

        # The first 800 (synthetic observations)
        random_sample_indices = random.sample(
            range(temp_df.iloc[:1000, :].shape[0]), k=800)

        # Combining original data with the 800 randomly selected observations
        new_df = np.vstack([temp_df.iloc[random_sample_indices, :],
                            temp_df.iloc[1000:, :]])

        # Perform standard scaling
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(new_df)

        # Perform PCA to reduce to 50 components
        pca_model = PCA(n_components=50)
        reduced_x = pca_model.fit_transform(scaled_data)

        return reduced_x

    def __do_kmeans(self, reduced_x, k):
        final_kmeans = KMeans(n_clusters=k, init="k-means++", max_iter=1000)
        labels_kmeans = final_kmeans.fit_predict(reduced_x)
        return labels_kmeans

    def __do_gmm(self, reduced_x, k):
        final_gmm = GaussianMixture(n_components=k)
        labels_gmm = final_gmm.fit_predict(reduced_x)
        return labels_gmm

    def __do_dbscan(self, reduced_x, epsilon, min_pts):
        dbscan_model = DBSCAN(eps=epsilon, min_samples=min_pts)
        labels_dbscan = dbscan_model.fit_predict(reduced_x)
        return labels_dbscan

    def __perform_clustering(self, reduced_x):
        activity = self.parameter_dict[self.muscle_activity]
        model = activity[self.muscle]["model"]
        parameters = activity[self.muscle]["parameters"]
        if model == "kmeans":
            labels = self.__do_kmeans(reduced_x, parameters[0])
        elif model == "gmm":
            labels = self.__do_gmm(reduced_x, parameters[0])
        else:
            labels = self.__do_dbscan(reduced_x, parameters[0],
                                      parameters[1])
        return labels

    # Align signals and calculate mean
    def __align_and_average(self, signals):
        if len(signals) > 0:
            reference_signal = signals[0].copy()
            for sig in signals[1:]:
                # Calculate the cross-correlation between the two signals
                cross_corr = signal.correlate(reference_signal, sig, mode='full')

                # Find the time lag that maximizes the cross-correlation
                time_lag = np.argmax(cross_corr) - len(reference_signal) + 1

                # Align the second signal to match the first signal
                if time_lag > 0:
                    aligned_signal2 = np.pad(sig, (time_lag, 0),
                                             'constant')[:len(reference_signal)].copy()
                    aligned_signal1 = reference_signal.copy()
                else:
                    aligned_signal1 = np.pad(reference_signal, (-time_lag, 0),
                                             'constant')[:len(sig)].copy()
                    aligned_signal2 = sig.copy()

                # Calculate current mean signal
                reference_signal = np.mean([reference_signal, aligned_signal2],
                                           axis=0).copy()
            return reference_signal
        return signals[0]

    def __generate_cluster_centric_signals(self, labels):
        # Get signals and cluster labels of original observations
        original_labels_gmm = labels[800:]
        print(len(original_labels_gmm))
        print(len(self.signals))

        # Form dictionary of clusters with corresponding indices
        cluster_dict = {}
        for i, label in enumerate(original_labels_gmm):
            if label in cluster_dict:
                cluster_dict[label].append(self.signals[i])
            else:
                cluster_dict[label] = [self.signals[i]]

        signals = []
        generated_signal_dict = {}

        # Form a dictionary of labels and corresponding cluster-centric signal
        for label in cluster_dict:
            if not (label in generated_signal_dict):
                all_signals = cluster_dict[label]
                centric_signal = self.__align_and_average(all_signals)
                # centric_signal = np.mean(all_signals, axis=0)
                generated_signal_dict[label] = centric_signal

        # Assign signals to each original observation
        for label in labels:
            signals.append(generated_signal_dict[label])

        # Clean the signals again to get rid of abrupt baseline shifts
        signals = [nk.bio_process(emg=each)[0]["EMG_Clean"].values
                   for each in signals].copy()

        return signals

    def __write_generated_signal_for_subject(self, generated_signal):
        fig = plt.figure()
        plt.plot(range(len(generated_signal)),
                 generated_signal)
        plt.xlabel("Samples")
        plt.ylabel("Magnitude")
        plt.title("Suggested normal")

        # Create path if not present
        file_path = f"generated_files/subject_{self.subject_number}/"
        if not os.path.exists(file_path):
            # if the demo_folder directory is not present
            # then create it.
            os.makedirs(file_path)
        plt.savefig(file_path + f"{self.muscle}.jpg", dpi=300)
        plt.close(fig)

    def get_cluster_signals(self):
        # obtain cluster centric signals
        return self.cluster_signals
