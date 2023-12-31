# Dental Loop Signals

This software allows users to upload EMG signal images containing either unipolar or bipolar EMG signals, generate normal signals and quotients for duration/intensity. Customizable for muscle selection and activities, it provides CSVs for comparisons, especially useful when processing multiple images.

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

3. Install customtkinter using "pip install customtkinter" on a terminal

4. Install Scikit-Learn using "pip install scikit-learn" on a terminal (might already be installed within the conda virtual environment)

5. Install OpenCV using "pip install opencv-python" on a terminal

6. Install NeuroKit2 using "pip install neurokit2" on a terminal

7. Install TSFEL using "pip install tsfel" on a terminal

8. Install pandas using "pip install pandas" on a terminal

NOTE: Libraries like pandas and Scikit-Learn may already come by default with Anaconda

**Alternatively, just copy and paste the following into the terminal with the conda virtual environment activated:**

pip install customtkinter
pip install scikit-learn
pip install opencv-python
pip install neurokit2
pip install tsfel
pip install pandas

9. Clone this repository or download the repository in your desired directory

# Running the software

1. Navigate to the src folder within the terminal

2. Run the command **python main.py** on the terminal

The following parameters can be adjusted:
- **Closest matching muscle activity** - Choose the one which closely resembles the muscle activity in your image(s)
- **Number of muscles** - Specify the number of EMG signals within one image
- **Magnification** - This does not have any impact on the functionality at this moment

3. After adjusting parameters, press the **Add Image** button and select one or more than one image

4. This would load a second window. Here, you can rename each muscle on the left panel. The right side of the window displays all the loaded images.

5. Press the **PROCESS** button to process the images

All the outputs generated will reside within the generated_files folder. Once processing is done, the second window will close and a file explorer with the generated subfolders will appear. Note that each time the **PROCESS** button is pressed, all outputs will be generated within the user defined folder name. This folder name is **task_1** by default. After processing once, if processing is done again for a different set of images and if the folder name is not changed, prior outputs within the previous folder will get renamed. Essentially, the folder name text box allows the user to generate outputs in a separate folder everytime processing is done.

## Examples
2 images for each of the activities of Anterior Protrusion, Lateral Excursion, Mouth Opening, and Chewing have been provided and can be used to test the application.

## Limitations
1. On some occasions, the CSV quotients generated for EMG intensity might show a quotient of 0. This is a limitation of the NeuroKit2 library due to it sometimes not being able to determine regions of EMG activity. This will be addressed in a future release.

2. As of right now, the images used to test the tool had 6 EMG signals in them. While the software may still function properly for images with less than 6 EMG signals, it cannot be guaranteed. This will be addressed on a future release.
