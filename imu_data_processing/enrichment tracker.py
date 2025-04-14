import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import sys
import threading
from processor import Processor

# Import the Dataloader class from the dataloader.py file
# Make sure dataloader.py is in the same directory or in the Python path
from dataloader import Dataloader

class IMUDataVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enrichment Tracking Data Processor")
        self.root.geometry("1920x1080")
        
        # Set default UI scale
        self.ui_scale = "normal"
        self.font_sizes = {
            "normal": {"button": 10, "label": 10, "entry": 10},
            "large": {"button": 14, "label": 14, "entry": 14}
        }
        
        # Default settings
        self.auto_save_raw = False
        self.threshold_value = 0.5
        
        # Create menu bar
        menubar = tk.Menu(root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open Data File", command=self.open_file)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=root.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        root.config(menu=menubar)
        
        # Create top container frame to hold file frame and help button
        self.top_container = tk.Frame(root)
        self.top_container.pack(fill=tk.X, padx=10, pady=10)
        
        # Create frame for file selection (within top container)
        self.file_frame = tk.Frame(self.top_container)
        self.file_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Add open button on the very left
        self.open_button = tk.Button(self.file_frame, text="Open Raw Data File", command=self.open_file_with_progress)
        self.open_button.pack(side=tk.LEFT, padx=5)
        
        # Add file label immediately to the right of the button
        self.file_label = tk.Label(self.file_frame, text="No file selected", anchor='w')
        self.file_label.pack(side=tk.LEFT, padx=5)
        
        # Add settings button to the top right corner
        self.settings_button = tk.Button(self.top_container, text="⚙", font=("Arial", 12, "bold"), 
                                     width=2, height=1, command=self.show_settings)
        self.settings_button.pack(side=tk.RIGHT, padx=5)
        
        # Add help button (i in a circle) to the top right corner
        self.help_button = tk.Button(self.top_container, text="ⓘ", font=("Arial", 12, "bold"), 
                                  width=2, height=1, command=self.show_instructions)
        self.help_button.pack(side=tk.RIGHT, padx=5)
        
        # Create frame for time range selection
        self.time_range_frame = tk.Frame(root)
        self.time_range_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Start and end time labels
        self.start_time_label = tk.Label(self.time_range_frame, text="Start Time: Not available")
        self.start_time_label.pack(side=tk.LEFT, padx=5)
        
        self.end_time_label = tk.Label(self.time_range_frame, text="End Time: Not available")
        self.end_time_label.pack(side=tk.LEFT, padx=5)
        
        # Custom start time entry
        self.custom_start_label = tk.Label(self.time_range_frame, text="Custom Start:")
        self.custom_start_label.pack(side=tk.LEFT, padx=(20, 5))
        
        self.custom_start_entry = tk.Entry(self.time_range_frame, width=20)
        self.custom_start_entry.pack(side=tk.LEFT, padx=5)
        
        # Custom end time entry
        self.custom_end_label = tk.Label(self.time_range_frame, text="Custom End:")
        self.custom_end_label.pack(side=tk.LEFT, padx=(10, 5))
        
        self.custom_end_entry = tk.Entry(self.time_range_frame, width=20)
        self.custom_end_entry.pack(side=tk.LEFT, padx=5)

        # Add hint text to the custom start and end time entries
        self.custom_start_entry.insert(0, "YYYY MM DD HH MM SS")  # Placeholder for start time
        self.custom_end_entry.insert(0, "YYYY MM DD HH MM SS")    # Placeholder for end time
        
        # Bind focus events to clear the placeholder text
        self.custom_start_entry.bind("<FocusIn>", lambda event: self.clear_placeholder(event, "YYYY MM DD HH MM SS"))
        self.custom_start_entry.bind("<FocusOut>", lambda event: self.restore_placeholder(event, "YYYY MM DD HH MM SS"))
        
        self.custom_end_entry.bind("<FocusIn>", lambda event: self.clear_placeholder(event, "YYYY MM DD HH MM SS"))
        self.custom_end_entry.bind("<FocusOut>", lambda event: self.restore_placeholder(event, "YYYY MM DD HH MM SS"))        
        
        # Update time range button
        self.update_range_button = tk.Button(self.time_range_frame, text="Update Time Range", 
                                         command=self.update_time_range, state=tk.DISABLED)
        self.update_range_button.pack(side=tk.LEFT, padx=10)
        
        # Create frame for action buttons
        self.action_frame = tk.Frame(root)
        self.action_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Threshold label and entry
        self.threshold_label = tk.Label(self.action_frame, text="Threshold:")
        self.threshold_label.pack(side=tk.LEFT, padx=5)
        
        self.threshold_var = tk.StringVar(value="0.5")
        self.threshold_entry = tk.Entry(self.action_frame, textvariable=self.threshold_var, width=5)
        self.threshold_entry.pack(side=tk.LEFT, padx=5)

        # Update threshold button
        self.update_threshold_button = tk.Button(self.action_frame, text="Update Threshold",
                            command=lambda: self.plot_imu_data())
        self.update_threshold_button.pack(side=tk.LEFT, padx=5)

        # Process data button
        self.process_button = tk.Button(self.action_frame, text="Process Data", 
                                     command=self.process_with_threshold, state=tk.DISABLED)
        self.process_button.pack(side=tk.LEFT, padx=5)
        
        # Save processed data button
        self.save_processed_button = tk.Button(self.action_frame, text="Save Processed Data", 
                                            command=self.save_processed_data, state=tk.DISABLED)
        self.save_processed_button.pack(side=tk.LEFT, padx=5)
        
        # Save raw data button
        self.save_raw_button = tk.Button(self.action_frame, text="Save Raw Data as CSV", 
                                      command=self.save_raw_data, state=tk.DISABLED)
        self.save_raw_button.pack(side=tk.LEFT, padx=5)
        
        # Plot view selector
        self.view_label = tk.Label(self.action_frame, text="View:")
        self.view_label.pack(side=tk.LEFT, padx=(20, 5))
        
        self.view_var = tk.StringVar(value="raw")
        self.raw_radio = tk.Radiobutton(self.action_frame, text="Raw Data", 
                                      variable=self.view_var, value="raw", 
                                      command=self.switch_view, state=tk.DISABLED)
        self.raw_radio.pack(side=tk.LEFT)
        
        self.processed_radio = tk.Radiobutton(self.action_frame, text="Processed Data", 
                                           variable=self.view_var, value="processed", 
                                           command=self.switch_view, state=tk.DISABLED)
        self.processed_radio.pack(side=tk.LEFT)
        
        # Create frame for plot
        self.plot_frame = tk.Frame(root)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initialize matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(9, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add a status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to load raw data")
        self.status_bar = tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initialize data variables
        self.imu_data = None
        self.imu_data_avg = None
        self.zoodata_dl = None
        self.current_file_path = None
        self.processed_data = None
        self.processor = None
        self.progress_dialog = None
        self.start_time = None
        self.end_time = None
        
    def show_settings(self):
        # Create a new window for settings
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x150")
        settings_window.transient(self.root)  # Set as transient to main window
        
        # Center the window
        self.center_window(settings_window)
        
        # Auto-save settings
        save_frame = tk.LabelFrame(settings_window, text="Auto-save Options")
        save_frame.pack(fill=tk.X, padx=10, pady=10)
        
        auto_save_var = tk.BooleanVar(value=self.auto_save_raw)
        tk.Checkbutton(save_frame, text="Auto-save raw data as CSV when loading", 
                      variable=auto_save_var).pack(anchor=tk.W, padx=10)
        
        # Save settings button
        def save_settings():
            self.auto_save_raw = auto_save_var.get()
            settings_window.destroy()
            
        tk.Button(settings_window, text="Save Settings", command=save_settings).pack(pady=10)
        
    def update_ui_scale(self):
        # Update font sizes based on selected scale
        font_dict = self.font_sizes[self.ui_scale]
        
        # Update buttons
        button_font = ("Arial", font_dict["button"])
        for widget in [self.open_button, self.process_button, self.save_processed_button, 
                     self.save_raw_button, self.update_range_button]:
            widget.config(font=button_font)
        
        # Update labels
        label_font = ("Arial", font_dict["label"])
        for widget in [self.file_label, self.start_time_label, self.end_time_label,
                     self.custom_start_label, self.custom_end_label, self.threshold_label,
                     self.view_label, self.status_bar]:
            widget.config(font=label_font)
        
        # Update entries
        entry_font = ("Arial", font_dict["entry"])
        for widget in [self.custom_start_entry, self.custom_end_entry, self.threshold_entry]:
            widget.config(font=entry_font)
        
        # Update radio buttons
        for widget in [self.raw_radio, self.processed_radio]:
            widget.config(font=label_font)
            
    def show_instructions(self):
        # Create a new window for instructions
        instructions_window = tk.Toplevel(self.root)
        instructions_window.title("Enrichment Tracking Data Processor - Instructions")
        instructions_window.geometry("600x500")
        instructions_window.transient(self.root)  # Set as transient to main window
        
        # Center the window
        self.center_window(instructions_window)
        
        # Create a frame with a scrollbar
        frame = tk.Frame(instructions_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add a scrollable text widget
        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget = tk.Text(frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        text_widget.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)
        
        # Add instructions text
        instructions = """
Enrichment Tracking Data Processor - User Instructions

This application helps you analyze and visualize data collected from the enrichment tracking device.

Basic Steps:
1. Open an IMU data file using the 'Open Raw Data File' button
2. View the raw data visualization
3. Process the data using the 'Process Data' button
4. Switch between raw and processed data views using the radio buttons
5. Save your data (raw or processed) using the corresponding buttons

Detailed Instructions:

LOADING DATA:
- Click the 'Open Raw Data File' button to select and load your IMU data file
- Supported formats include .txt and .csv files
- The file name will appear next to the button once loaded
- Raw data visualization will be displayed automatically

PROCESSING DATA:
- After loading a file, enter a threshold value (default is 0.5)
- Click the 'Process Data' button
- Processing may take a moment depending on the file size
- Once complete, you can view the processed data

TIME RANGE:
- The start and end times of your data will be displayed after loading
- You can enter custom start and end times and click 'Update Time Range' to filter the data

VISUALIZATION:
- Use the 'Raw Data' and 'Processed Data' radio buttons to switch between views
- Raw data view shows IMU data averages and the threshold as a dotted red line
- Processed data view shows minutes of interaction per hour

SAVING DATA:
- 'Save Raw Data' button exports the raw IMU data to CSV format
- 'Save Processed Data' button exports the processed results
- You will be prompted to select an output directory

SETTINGS:
- Click the ⚙ button to access settings
- You can change the UI scale between normal and large
- Enable auto-save to automatically save raw data as CSV when loading

TIPS:
- You can resize the window to get a better view of the plots
- Use the menu bar's File > Exit option to close the application
- The status bar at the bottom shows the current state of the application

        """
        
        text_widget.insert(tk.END, instructions)
        text_widget.config(state=tk.DISABLED)  # Make text read-only
        
        # Add a close button
        close_button = tk.Button(instructions_window, text="Close", command=instructions_window.destroy)
        close_button.pack(pady=10)
        
    def open_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.status_var.set(f"Loading data from {os.path.basename(file_path)}...")
                self.root.update()
                
                # Load data using Dataloader
                self.zoodata_dl = Dataloader.read_file(file_path)
                self.imu_data = np.array(self.zoodata_dl.raw_data)
                self.current_file_path = file_path
                
                # Calculate IMU data averages - taking absolute mean of last 3 values in each row
                self.update_imu_data_avg()
                
                # Get and display time range
                self.update_time_display()
                
                # Plot the data
                self.plot_imu_data()
                
                # Update file label
                self.file_label.config(text=os.path.basename(file_path))
                
                # Update window title
                self.root.title(f"Enrichment Tracking Data Processor - {os.path.basename(file_path)}")
                
                # Enable buttons
                self.process_button.config(state=tk.NORMAL)
                self.save_raw_button.config(state=tk.NORMAL)
                self.raw_radio.config(state=tk.NORMAL)
                self.update_range_button.config(state=tk.NORMAL)
                
                # Auto-save raw data if enabled
                if self.auto_save_raw:
                    self.save_raw_data(auto=True)
                
                self.status_var.set(f"Data loaded successfully. Showing raw data from {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load or process file: {str(e)}")
                self.status_var.set("Error loading data")
    
    def update_imu_data_avg(self):
        # Calculate IMU data averages - taking absolute mean of last 3 values in each row
        self.imu_data_avg = []
        for row in self.imu_data:
            avg = np.abs(np.mean(row[-3:]))
            self.imu_data_avg.append(avg)
        self.imu_data_avg = np.array(self.imu_data_avg)
    
    def update_time_display(self):
        if self.zoodata_dl is not None:
            self.start_time = self.zoodata_dl.get_first_timestamp()
            self.end_time = self.zoodata_dl.get_last_timestamp()
            
            # Update the labels
            self.start_time_label.config(text=f"Start Time: {int(self.start_time[0])}/{int(self.start_time[1]):02d}/"
                                               f"{int(self.start_time[2]):02d} {int(self.start_time[3]):02d}:{int(self.start_time[4]):02d}:{int(self.start_time[5]):02d}")
            self.end_time_label.config(text=f"End Time: {int(self.end_time[0])}/{int(self.end_time[1]):02d}/"
                                             f"{int(self.end_time[2]):02d} {int(self.end_time[3]):02d}:{int(self.end_time[4]):02d}:{int(self.end_time[5]):02d}")
            
            # Format the start and end times as "YYYY MM DD HH MM SS"
            start_time_formatted = f"{int(self.start_time[0])} {int(self.start_time[1]):02d} {int(self.start_time[2]):02d} " \
                                f"{int(self.start_time[3]):02d} {int(self.start_time[4]):02d} {int(self.start_time[5]):02d}"
            end_time_formatted = f"{int(self.end_time[0])} {int(self.end_time[1]):02d} {int(self.end_time[2]):02d} " \
                                f"{int(self.end_time[3]):02d} {int(self.end_time[4]):02d} {int(self.end_time[5]):02d}"        

            # Pre-fill the custom entries with current values
            self.custom_start_entry.delete(0, tk.END)
            self.custom_start_entry.insert(0, start_time_formatted)
            
            self.custom_end_entry.delete(0, tk.END)
            self.custom_end_entry.insert(0, end_time_formatted)
    
    def update_time_range(self):
        if self.zoodata_dl is None:
            messagebox.showwarning("Warning", "No data loaded. Please open a data file first.")
            return

        custom_start = self.custom_start_entry.get().strip()
        custom_end = self.custom_end_entry.get().strip()

        if not custom_start or not custom_end:
            messagebox.showwarning("Warning", "Please enter both start and end times.")
            return

        try:
            # Create and show progress dialog
            self.create_progress_dialog("Updating Time Range", "Updating time range, please wait...")

            # Start the time range update in a separate thread
            threading.Thread(target=lambda: self.update_time_range_thread(custom_start, custom_end), daemon=True).start()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to update time range. Time should be in format YYYY MM DD HH MM SS.")
            self.status_var.set("Error updating time range")
    
    def update_time_range_thread(self, custom_start, custom_end):
        try:
            # Use the Dataloader's crop function to get data within the custom time range
            cropped_data = self.zoodata_dl.crop(custom_start, custom_end)

            # Update our data objects
            self.zoodata_dl = cropped_data
            self.imu_data = np.array(self.zoodata_dl.raw_data)

            # Update IMU data averages
            self.update_imu_data_avg()

            # Update displayed time range and plot
            self.root.after(0, self.update_time_display)
            self.root.after(0, self.plot_imu_data)

            # Reset processed data since the time range has changed
            self.processed_data = None
            self.root.after(0, lambda: self.processed_radio.config(state=tk.DISABLED))
            self.root.after(0, lambda: self.save_processed_button.config(state=tk.DISABLED))

            # Close progress dialog and show success message
            self.root.after(0, self.update_time_range_complete)
            self.root.after(0, lambda: self.status_var.set(f"Time range updated: {custom_start} to {custom_end}"))
            self.root.after(0, lambda: messagebox.showinfo("Success", "Time range updated successfully."))

        except Exception as e:
            self.root.after(0, self.update_time_range_complete)
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to update time range: {str(e)}"))
            self.root.after(0, lambda: self.status_var.set("Error updating time range"))

    def update_time_range_complete(self):
        # Close progress dialog
        if self.progress_dialog:
            self.progress_dialog.destroy()
            self.progress_dialog = None

    def open_file_with_progress(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            # Create and show progress dialog for file loading
            self.create_progress_dialog("Loading Data", "Loading raw data file...")
            
            # Start file loading in a separate thread
            threading.Thread(target=lambda: self.load_file_thread(file_path), daemon=True).start()
    
    def load_file_thread(self, file_path):
        try:
            self.status_var.set(f"Loading data from {os.path.basename(file_path)}...")
            
            # Load data using Dataloader
            self.zoodata_dl = Dataloader.read_file(file_path)
            self.imu_data = np.array(self.zoodata_dl.raw_data)
            self.current_file_path = file_path
            
            # Calculate IMU data averages
            self.update_imu_data_avg()
            
            # Update UI from main thread
            self.root.after(0, lambda: self.file_load_complete(file_path))
            
        except Exception as e:
            # Show error message from main thread
            self.root.after(0, lambda: self.show_load_error(str(e)))
            
    def file_load_complete(self, file_path):
        # Close progress dialog
        if self.progress_dialog:
            self.progress_dialog.destroy()
            self.progress_dialog = None
        
        # Update time display
        self.update_time_display()
            
        # Plot the data
        self.plot_imu_data()
        
        # Update file label
        self.file_label.config(text=os.path.basename(file_path))
        
        # Update window title
        self.root.title(f"Enrichment Tracking Data Processor - {os.path.basename(file_path)}")
        
        # Enable buttons
        self.process_button.config(state=tk.NORMAL)
        self.save_raw_button.config(state=tk.NORMAL)
        self.raw_radio.config(state=tk.NORMAL)
        self.update_range_button.config(state=tk.NORMAL)
        
        self.status_var.set(f"Data loaded successfully. Showing raw data from {os.path.basename(file_path)}")
        
        # Auto-save raw data if enabled
        if self.auto_save_raw:
            self.save_raw_data(auto=True)
        
        # Show success message
        messagebox.showinfo("File Loaded", f"Data loaded successfully from {os.path.basename(file_path)}")
    
    def show_load_error(self, error_msg):
        # Close progress dialog
        if self.progress_dialog:
            self.progress_dialog.destroy()
            self.progress_dialog = None
            
        messagebox.showerror("Error", f"Failed to load file: {error_msg}")
        self.status_var.set("Error loading data")

    def create_progress_dialog(self, title="Processing", message="Please wait..."):
        # Create a progress dialog
        self.progress_dialog = tk.Toplevel(self.root)
        self.progress_dialog.title(title)
        self.progress_dialog.geometry("300x100")
        self.progress_dialog.transient(self.root)
        self.progress_dialog.grab_set()
        
        # Center the dialog
        self.center_window(self.progress_dialog)
        
        # Add progress bar and label
        tk.Label(self.progress_dialog, text=message).pack(pady=(10, 5))
        self.progress_bar = ttk.Progressbar(self.progress_dialog, mode="indeterminate", length=250)
        self.progress_bar.pack(pady=5)
        self.progress_bar.start(10)  # Start the animation
        
    def center_window(self, window):
        window.update_idletasks()
        width = window.winfo_width()
        height = window.winfo_height()
        x = (window.winfo_screenwidth() // 2) - (width // 2)
        y = (window.winfo_screenheight() // 2) - (height // 2)
        window.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    
    def process_with_threshold(self):
        try:
            threshold = float(self.threshold_var.get())
            self.threshold_value = threshold
            self.process_data_with_progress(threshold)
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for the threshold.")
    
    def process_data_with_progress(self, threshold):
        if self.zoodata_dl is None:
            messagebox.showwarning("Warning", "No data loaded. Please open a data file first.")
            return
        
        # Create and show progress dialog
        self.create_progress_dialog(title="Processing Data", message="Processing data, please wait...")
        
        # Start processing in a separate thread to keep UI responsive
        threading.Thread(target=lambda: self.process_data_thread(threshold), daemon=True).start()
    
    def process_data_thread(self, threshold):
        try:
            # Get base name of the file without extension for data_name
            data_name = os.path.splitext(os.path.basename(self.current_file_path))[0]
            
            # Create processor instance and process the data
            self.processor = Processor(self.zoodata_dl)  
            self.processed_data = self.processor.process_imu_data(threshold)
            
            # Update UI from main thread
            self.root.after(0, self.process_complete)
            
        except Exception as e:
            # Show error message from main thread
            self.root.after(0, lambda: self.show_process_error(str(e)))
    
    def process_complete(self):
        # Close progress dialog
        if self.progress_dialog:
            self.progress_dialog.destroy()
            self.progress_dialog = None
        
        # Enable save and view buttons
        self.save_processed_button.config(state=tk.NORMAL)
        self.processed_radio.config(state=tk.NORMAL)
        
        # Show completion message
        messagebox.showinfo("Processing Complete", "Data processing completed successfully!")
        self.status_var.set("Data processing completed successfully")
    
    def show_process_error(self, error_msg):
        # Close progress dialog
        if self.progress_dialog:
            self.progress_dialog.destroy()
            self.progress_dialog = None
            
        messagebox.showerror("Error", f"Failed to process data: {error_msg}")
        self.status_var.set("Error processing data")
    
    def save_raw_data(self, auto=False):
        if self.imu_data is None:
            if not auto:  # Only show warning if not auto-saving
                messagebox.showwarning("Warning", "No data loaded. Please open a data file first.")
            return
            
        # Ask for output directory if not auto-saving
        if auto:
            # Use the same directory as the input file
            output_dir = os.path.dirname(self.current_file_path)
        else:
            output_dir = filedialog.askdirectory(title="Select Output Directory")
            if not output_dir:
                return
        
        # Save raw data as CSV
        output_name = os.path.join(output_dir, f"raw_data_{os.path.splitext(os.path.basename(self.current_file_path))[0]}.csv")
        pd.DataFrame(self.imu_data).to_csv(output_name, index=False, header=False)
        
        self.status_var.set(f"Raw data saved successfully to {output_name}")
        
        if not auto:  # Only show message if not auto-saving
            messagebox.showinfo("Success", f"Raw data saved successfully.\nOutput saved to:\n{output_name}")
    
    def save_processed_data(self):
        if self.processed_data is None or self.processor is None:
            messagebox.showwarning("Warning", "No processed data available. Please process the data first.")
            return
        
        # Ask for output directory
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            return  # User cancelled directory selection
            
        data_name = os.path.splitext(os.path.basename(self.current_file_path))[0]
        
        # Save processed data using the processor's save method
        output_name = self.processor.save_results(output_dir, data_name)
        
        self.status_var.set(f"Data processed successfully. Output saved to {output_name}")
        messagebox.showinfo("Success", f"Data processed successfully.\nOutput saved to:\n{output_name}")
    
    def switch_view(self):
        view_type = self.view_var.get()
        if view_type == "raw":
            self.plot_imu_data()
        else:  # processed
            self.plot_processed_data()
    
    def plot_imu_data(self):
        if self.imu_data_avg is not None:
            # Clear previous plot
            self.ax.clear()
            
            # Create new plot
            self.ax.plot(range(len(self.imu_data_avg)), self.imu_data_avg)
            
            # Add threshold line
            threshold = float(self.threshold_var.get())
            self.ax.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
            
            self.ax.set_xlabel('Time (ms)')
            self.ax.set_ylabel('Acceleration (m/s^2)')
            self.ax.set_title('Raw Enrichment Device Tracker Data')
            self.ax.grid(True)
            self.ax.legend()
            
            # Update canvas
            self.canvas.draw()

    def plot_processed_data(self):
        if self.processed_data is not None:
            # Clear previous plot
            self.ax.clear()
            
            # Create new plot in main figure
            time_labels = [f'{int(year)}-{int(month):02d}-{int(day):02d} {int(hour):02d}:00' 
                        for year, month, day, hour in self.processed_data[:, :4]]
            counts = self.processed_data[:, 4]
            
            # Create a bar plot with reduced font size for x-tick labels
            self.ax.bar(range(len(counts)), counts)
            self.ax.set_xlabel('Time (Year-Month-Day Hour)')
            self.ax.set_ylabel('Minutes of interaction')
            self.ax.set_title('Minutes of interaction per hour')
            self.ax.set_xticks(range(len(counts)))
            self.ax.set_xticklabels(time_labels, rotation=90, fontsize=8)
            
            # Make the plot fit properly in available space
            plt.tight_layout()
            
            # Update canvas
            self.canvas.draw()

        # Add helper methods for managing placeholder text
    def clear_placeholder(self, event, placeholder):
        entry = event.widget
        if entry.get() == placeholder:
            entry.delete(0, tk.END)
            entry.config(fg="black")  # Change text color to black
    
    def restore_placeholder(self, event, placeholder):
        entry = event.widget
        if not entry.get():  # If the entry is empty
            entry.insert(0, placeholder)
            entry.config(fg="gray")  # Change text color to gray

if __name__ == "__main__":
    root = tk.Tk()
    app = IMUDataVisualizerApp(root)
    root.mainloop()