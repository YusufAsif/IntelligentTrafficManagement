import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import *
from PIL import Image, ImageTk
import time
import numpy as np
import pandas as pd
from utils import detect_color, get_direction  # Import utility functions
import config  # Import configurations

class CarTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Car Tracking System")
        self.root.geometry("800x600")

        # Initialize UI
        self.canvas = Canvas(root, width=800, height=600)
        self.canvas.pack()
        control_frame = Frame(root)
        control_frame.pack()
        self.load_button = Button(control_frame, text="Load Video", command=self.load_video)
        self.load_button.pack(side=LEFT)
        self.stop_button = Button(control_frame, text="Stop Tracking", command=self.stop_tracking)
        self.stop_button.pack(side=LEFT)
        self.stop_button.config(state=tk.DISABLED)

        # Log area and variables
        self.log_text = Text(root, height=10, width=80)
        self.log_text.pack()
        self.car_data = []
        self.video_path = None
        self.cap = None
        self.is_tracking = False
        self.delay = 30
        self.car_ids = {}
        self.next_car_id = 0
        self.previous_positions = {}
        self.avg_speeds = {}
        self.peak_speeds = {}
        self.directions = {}
        self.colors = {}
        self.draw_gradient()

    def draw_gradient(self):
        for i in range(600):
            color = f'#{int(173 + (135 - 173) * (i / 600)):02x}{int(216 + (240 - 216) * (i / 600)):02x}{int(230):02x}'
            self.canvas.create_line(0, i, 800, i, fill=color)

    def load_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv")])
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open video file.")
                return
            self.is_tracking = True
            self.stop_button.config(state=tk.NORMAL)
            self.process_video()

    def stop_tracking(self):
        self.is_tracking = False
        if self.cap is not None:
            self.cap.release()
        self.stop_button.config(state=tk.DISABLED)
        self.save_car_data()

    def process_video(self):
        fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)
        while self.is_tracking:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (800, 600))
            fgmask = fgbg.apply(frame)
            contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            detected_cars = [(x, y, w, h) for c in contours if cv2.contourArea(c) > 500
                             for x, y, w, h in [cv2.boundingRect(c)] if y + h > frame.shape[0] // 2]
            self.update_car_ids(detected_cars, frame)
            self.calculate_avg_peak_and_direction(frame)
            image = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=image, anchor='nw')
            self.canvas.image = image
            self.log_car_details()
            self.root.update()
            time.sleep(self.delay / 1000)
        self.cap.release()

    def update_car_ids(self, detected_cars, frame):
        current_ids = set(self.car_ids.values())
        detected_ids = set()
        for x, y, w, h in detected_cars:
            found_id = None
            for car_id, (bx, by, bw, bh) in self.car_ids.items():
                if bx < x + w and bx + bw > x and by < y + h and by + bh > y:
                    found_id = car_id
                    break
            if found_id is not None:
                self.car_ids[found_id] = (x, y, w, h)
                detected_ids.add(found_id)
            else:
                self.car_ids[self.next_car_id] = (x, y, w, h)
                detected_ids.add(self.next_car_id)
                self.next_car_id += 1
        for car_id in list(self.car_ids.keys()):
            if car_id not in detected_ids:
                del self.car_ids[car_id]
        for car_id, (bx, by, bw, bh) in self.car_ids.items():
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.putText(frame, f'Car {car_id}', (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            self.colors[car_id] = detect_color(frame[by:by + bh, bx:bx + bw])

    def calculate_avg_peak_and_direction(self, frame):
        for car_id, (bx, by, bw, bh) in self.car_ids.items():
            current_position = (bx + bw // 2, by + bh // 2)
            if car_id in self.previous_positions:
                distance_meters = np.linalg.norm(np.array(current_position) - np.array(self.previous_positions[car_id])) * config.PIXEL_TO_METER
                speed = distance_meters / (1 / config.FRAME_RATE)
                if car_id in self.avg_speeds:
                    self.avg_speeds[car_id].append(speed)
                else:
                    self.avg_speeds[car_id] = [speed]
                self.peak_speeds[car_id] = max(self.peak_speeds.get(car_id, 0), speed)
                self.directions[car_id] = get_direction(self.previous_positions[car_id], current_position)
            self.previous_positions[car_id] = current_position

    def log_car_details(self):
        self.log_text.delete(1.0, END)
        log_details = "ID\tAvg Speed (m/s)\tPeak Speed (m/s)\tDirection\tColor\n"
        for car_id in self.avg_speeds:
            avg_speed = np.mean(self.avg_speeds[car_id])
            log_details += f"{car_id}\t{avg_speed:.2f}\t{self.peak_speeds[car_id]:.2f}\t{self.directions[car_id]}\t{self.colors[car_id]}\n"
        self.log_text.insert(END, log_details)

    def save_car_data(self):
        data_to_save = [{'ID': car_id, 'Avg Speed (m/s)': np.mean(self.avg_speeds[car_id]),
                         'Peak Speed (m/s)': self.peak_speeds[car_id],
                         'Direction': self.directions[car_id], 'Color': self.colors[car_id]}
                        for car_id in self.avg_speeds]
        df = pd.DataFrame(data_to_save)
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if save_path:
            df.to_csv(save_path, index=False)
            messagebox.showinfo("Success", f"Car data saved to {save_path}")
