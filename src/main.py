import tkinter as tk
from tkinter import Label
from tkinter import Entry
from tkinter import Button
from tkinter import Frame
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
from processing import Process
import threading


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Generate normal")

        # All text boxes used
        self.num_muscles_text_box = None
        self.mag_text_box = None
        self.option_menu_option = None

        # Control variables
        self.is_new_window_active = False

        # Create variables to keep track of parameters
        self.images = []
        self.muscle_name_text_boxes = []

        # Current muscle activity
        self.activity = "chewing"

        # Member widgets
        self.image_viewer = None

        self.__add_labels()
        self.__add_text_boxes()
        self.__add_buttons()
        self.__add_option_menus()

    def __add_labels(self):
        # Add magnification label
        label_mag = Label(self, text="Magnification")
        label_mag.grid(row=0, column=0)

        # Add image adding label
        label_img = Label(self, text="Insert image")
        label_img.grid(row=1, column=0)

        # Add label for number of muscles
        label_num_muscles = Label(self, text="Number of muscles")
        label_num_muscles.grid(row=2, column=0)

        # Add label for activity type
        label_activity = Label(self, text="Closest matching activity")
        label_activity.grid(row=3, column=0)

    def __add_option_menus(self):
        # Create the list of options 
        options_list = ["Chewing", "Maximum Lateral Excursion (MLE)", 
                        "Maximum Mouth Opening (MMO)",
                        "Maximum Anterior Protrusion (MAP)"]
        self.option_menu_option = tk.StringVar(self)
        self.option_menu_option.set(options_list[0])
        self.option_menu_option.trace("w", self.__select_from_menu)
        menu = tk.OptionMenu(self, self.option_menu_option,
                             *options_list)
        menu.grid(row=3, column=1)

    def __add_text_boxes(self):
        # Add magnification text box
        mag_initial_text = tk.StringVar()
        mag_text_box = Entry(self, textvariable=mag_initial_text)
        mag_initial_text.set("1x")
        mag_text_box.grid(row=0, column=1)
        self.mag_text_box = mag_text_box

        # Add text box for number of muscles
        validation = self.register(self.__validate_input)
        num_muscles_initial_text = tk.StringVar()
        num_muscles_text_box = Entry(self, validate="key",
                                     validatecommand=(validation, '%P'),
                                     textvariable=num_muscles_initial_text)
        num_muscles_initial_text.set("6")
        num_muscles_text_box.grid(row=2, column=1)
        self.num_muscles_text_box = num_muscles_text_box

    def __add_buttons(self):
        # Add button to import image
        add_image_button = Button(self, text="Add image",
                                  command=self.__browse_images)
        add_image_button.grid(row=1, column=1)

        # Add an exit button to close the application
        exit_button = Button(self, text="Exit",
                             command=self.quit)
        exit_button.grid(row=4, column=1)

    def __display_images(self):
        self.muscle_name_text_boxes = []
        if not self.is_new_window_active:
            # Create a new window
            new_window = ctk.CTkToplevel()
            new_window.geometry('1600x900')

            # Add a canvas to the new window
            frame = ctk.CTkScrollableFrame(new_window, orientation="vertical",
                                           width=1600, height=900)
            frame.pack()

            # Add left panel for renaming muscles
            left_frame = Frame(frame)
            left_frame.grid(row=0, column=0)
            num_of_muscles = int(self.num_muscles_text_box.get())
            for i in range(num_of_muscles):
                tmp_lbl = Label(left_frame, text=f"Muscle {i+1} name:")
                tmp_lbl.pack(pady=10)
                tmp_text_box = Entry(left_frame)
                tmp_text_box.pack()
                self.muscle_name_text_boxes.append(tmp_text_box)

            # Add button to process
            # process_button = Button(left_frame, text="PROCESS",
            #                         command=threading.Thread(
            #                             target=self.__process_signals).start)
            process_button = Button(left_frame, text="PROCESS",
                                    command=self.__process_signals)
            process_button.pack(pady=20)

            # Add images
            for i, image in enumerate(self.images):
                label = ctk.CTkLabel(frame, image=image,
                                     text="")
                label.image = image
                # label.pack(pady=40)
                label.grid(row=i, column=1,
                           pady=40, padx=20)

            # Bind the closing event to the on_closing function
            new_window.protocol("WM_DELETE_WINDOW", self.__on_closing)
            self.image_viewer = new_window

            # self.is_new_window_active = True

    def __browse_images(self):
        # Open a file dialog to select image files
        file_paths = filedialog.askopenfilenames(filetypes=[
            ("Image files", "*.png;*.jpg;*.jpeg")
        ])

        # Load and display images
        for file_path in file_paths:
            image = Image.open(file_path)
            photo = ImageTk.PhotoImage(image)
            self.images.append(photo)

        # Display the images
        if len(self.images) > 0:
            if self.image_viewer is None:
                self.__display_images()
            else:
                self.image_viewer.destroy()
                self.__display_images()

    def __on_closing(self):
        self.muscle_name_text_boxes = []
        self.images = []
        self.image_viewer.destroy()
        self.image_viewer = None

    def __select_from_menu(self, *args):
        # Handle option menu selection
        option = self.option_menu_option.get()
        if option == "Maximum Lateral Excursion (MLE)":
            self.activity = "mle"
        elif option == "Chewing":
            self.activity = "chewing"
        elif option == "Maximum Mouth Opening (MMO)":
            self.activity = "mmo"
        else:
            self.activity = "map"

    def __validate_input(self, new_value):
        # Check if the new value is a valid integer
        if new_value == "":
            return True

        try:
            int(new_value)
            return True
        except ValueError:
            return False

    def __process_signals(self):
        print(f"There are {len(self.images)} images")
        muscle_labels = [each.get() for each in self.muscle_name_text_boxes]
        muscle_names = ["ta_r", "ta_l", "mm_r",
                        "mm_l", "da_r", "da_l"]
        print("Current muscle activity:", "chewing")
        print("Muscle names:", muscle_names)
        print("Curr magnification:", self.mag_text_box.get())
        print("Curr num of muscles:", self.num_muscles_text_box.get())
        images = [ImageTk.getimage(each) for each in self.images]
        process = Process(images, muscle_names,
                          self.activity)

# The main UI loop
app = App()
app.mainloop()
