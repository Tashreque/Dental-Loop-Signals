# Dental Loop Signals

This software allows users to upload EMG signal images, generate normal signals and quotients for duration/intensity. Customizable for muscle selection and activities, it provides CSVs for comparisons, especially useful when processing multiple images.

# What is Dental Loop Signals?

This tool enables users to upload images containing EMG signals from one or more subjects, facilitating the creation of suggested normal EMG signals and associated quotients reflecting both the duration and intensity of EMG activities. Each image includes six specific muscular EMG signals—Temporalis (left and right), Masseter (left and right), and Digastric (left and right). Users have the option to specify the number of muscles in each signal and select muscle activities corresponding to their desired images. After uploading one or more images, the tool allows users to choose and rename muscles, and based on user-defined parameters, it generates overlapped EMG signal images and applies an appropriate clustering algorithm. This process yields normal EMG signal images and corresponding CSV files for each muscle. Alongside the normal EMG signals, individual CSV files are produced for each image, providing quotients for the average duration and intensity of EMG activity for each muscle. This facilitates comprehensive comparisons between the original and suggested normal EMG signals, with the tool offering enhanced analysis by generating overlapped signals for improved data visualization when processing multiple images.

# Features and Functionalities

1. Adjustable Settings: Users have the flexibility to personalize parameters, such as the number of muscles for processing, tailoring the signal analysis to their specific preferences.

2. Activity Specification: Users can define related activities, selecting from choices like Maximum Mouth Opening (MMO), Maximum Lateral Excursion (MLE), Maximum Anterior Protrusion (MAP), and Chewing.

3. Batch Processing: The tool supports the addition and collective processing of multiple EMG images from different subjects, streamlining workflow efficiency.

4. Muscle Selection: Prior to processing, users can choose relevant muscles corresponding to those in the loaded image, ensuring accuracy in data analysis.

5. Customized Labeling: Users can rename or label each muscle according to their choice, and the outputs will reflect these selected names instead of default labels.

## Necessary resources

For the software to operate properly, it is crucial not to modify any subfolders or files within the resources folder.

## External Dependencies

customtkinter, Scikit-Learn, OpenCV, NeuroKit2, TSFEL, Pandas

# Installation

1. Download and install Anaconda. Anaconda can be downloaded from https://www.anaconda.com/download

2. Once installed, follow the instructions laid out in Anaconda to activate a virtual environment

3. Install Scikit-Learn using "pip install scikit-learn" on a terminal (might already be installed within the conda virtual environment)

4. Install OpenCV using "pip install opencv-python" on a terminal

5. Install NeuroKit2 using "pip install neurokit2" on a terminal

6. Install TSFEL using "pip install tsfel" on a terminal

7. Install pandas using "pip install pandas" on a terminal

NOTE: Libraries like pandas and Scikit-Learn may already come by default with Anaconda

**Alternatively, just copy and paste the following into the terminal with the conda virtual environment activated:**

pip install scikit-learn
pip install opencv-python
pip install neurokit2
pip install tsfel
pip install pandas

8. Clone this repository or download the repository in your desired directory

9. Navigate to the src folder within the terminal

10. Run the command **python main.py** on the terminal
