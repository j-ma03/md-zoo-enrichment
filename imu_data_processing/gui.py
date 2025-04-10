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
        self.root.geometry("1000x800")
        
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
        
        # Add help button (i in a circle) to the top right corner
        self.help_button = tk.Button(self.top_container, text="ⓘ", font=("Arial", 12, "bold"), 
                                  width=2, height=1, command=self.show_instructions)
        self.help_button.pack(side=tk.RIGHT, padx=5)
        
        # Create frame for action buttons
        self.action_frame = tk.Frame(root)
        self.action_frame.pack(fill=tk.X, padx=10, pady=5)

        # Process data button
        self.process_button = tk.Button(self.action_frame, text="Process Data", 
                                     command=self.process_data_with_progress, state=tk.DISABLED)
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
- After loading a file, click the 'Process Data' button
- Processing may take a moment depending on the file size
- Once complete, you can view the processed data

VISUALIZATION:
- Use the 'Raw Data' and 'Processed Data' radio buttons to switch between views
- Raw data view shows IMU data averages
- Processed data view shows minutes of interaction per hour

SAVING DATA:
- 'Save Raw Data' button exports the raw IMU data to CSV format
- 'Save Processed Data' button exports the processed results
- You will be prompted to select an output directory

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
                self.imu_data_avg = []
                for row in self.imu_data:
                    avg = np.abs(np.mean(row[-3:]))
                    self.imu_data_avg.append(avg)
                self.imu_data_avg = np.array(self.imu_data_avg)
                
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
                
                self.status_var.set(f"Data loaded successfully. Showing raw data from {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load or process file: {str(e)}")
                self.status_var.set("Error loading data")
    
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
            
            # Calculate IMU data averages - taking absolute mean of last 3 values in each row
            self.imu_data_avg = []
            for row in self.imu_data:
                avg = np.abs(np.mean(row[-3:]))
                self.imu_data_avg.append(avg)
            self.imu_data_avg = np.array(self.imu_data_avg)
            
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
        
        self.status_var.set(f"Data loaded successfully. Showing raw data from {os.path.basename(file_path)}")
        
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
    
    def process_data_with_progress(self):
        if self.zoodata_dl is None:
            messagebox.showwarning("Warning", "No data loaded. Please open a data file first.")
            return
        
        # Create and show progress dialog
        self.create_progress_dialog()
        
        # Start processing in a separate thread to keep UI responsive
        threading.Thread(target=self.process_data_thread, daemon=True).start()
    
    def process_data_thread(self):
        try:
            # Get base name of the file without extension for data_name
            data_name = os.path.splitext(os.path.basename(self.current_file_path))[0]
            
            # Create processor instance and process the data
            self.processor = Processor(self.zoodata_dl)  
            self.processed_data = self.processor.process_imu_data()
            
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
    
    def save_raw_data(self):
        if self.imu_data is None:
            messagebox.showwarning("Warning", "No data loaded. Please open a data file first.")
            return
            
        # Ask for output directory
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            return
        # Save raw data as CSV
        output_name = os.path.join(output_dir, f"raw_data_{os.path.splitext(os.path.basename(self.current_file_path))[0]}.csv")
        pd.DataFrame(self.imu_data).to_csv(output_name, index=False, header=False)
        self.status_var.set(f"Raw data saved successfully to {output_name}")
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
            self.ax.set_xlabel('Time (ms)')
            self.ax.set_ylabel('Acceleration (m/s^2)')
            self.ax.set_title('Raw Enrichment Device Tracker Data')
            self.ax.grid(True)
            
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

if __name__ == "__main__":
    root = tk.Tk()
    app = IMUDataVisualizerApp(root)
    root.mainloop()