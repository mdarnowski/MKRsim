import numpy as np
import joblib
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import messagebox, Canvas
from threading import Thread
import time

# Load the saved model, scaler, and label encoder
model = load_model('laundry_sorting_model.h5')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Define the function for predicting the category of a laundry item
def predict_laundry_item(r, g, b):
    new_laundry_item = np.array([[r, g, b]])
    new_laundry_item_scaled = scaler.transform(new_laundry_item)
    prediction = model.predict(new_laundry_item_scaled)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

# Create a Tkinter window
window = tk.Tk()
window.title("Laundry Sorting Simulation")

# Create counters for the bags
dark_counter = 0
white_counter = 0
colored_counter = 0

dark_counter_label = tk.Label(window, text=f"Dark Bag: {dark_counter} items")
dark_counter_label.pack()

white_counter_label = tk.Label(window, text=f"White Bag: {white_counter} items")
white_counter_label.pack()

colored_counter_label = tk.Label(window, text=f"Colored Bag: {colored_counter} items")
colored_counter_label.pack()

# Create canvas to draw the bags
canvas = Canvas(window, width=600, height=400)
canvas.pack()

# Draw the bags
canvas.create_rectangle(50, 50, 200, 350, fill="gray", tags="dark_bag")
canvas.create_text(125, 30, text="Dark Bag")

canvas.create_rectangle(250, 50, 400, 350, fill="lightgray", tags="white_bag")
canvas.create_text(325, 30, text="White Bag")

canvas.create_rectangle(450, 50, 600, 350, fill="orange", tags="colored_bag")
canvas.create_text(525, 30, text="Colored Bag")

# Function to add a new item to the appropriate bag
def add_item_to_bag():
    global dark_counter, white_counter, colored_counter

    while True:
        r = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        b = np.random.randint(0, 256)
        color = f'#{r:02x}{g:02x}{b:02x}'
        predicted_label = predict_laundry_item(r, g, b)

        if predicted_label == 'dark':
            canvas.create_oval(100, 300-dark_counter*10, 150, 350-dark_counter*10, fill=color)
            dark_counter += 1
            dark_counter_label.config(text=f"Dark Bag: {dark_counter} items")
        elif predicted_label == 'white':
            canvas.create_oval(300, 300-white_counter*10, 350, 350-white_counter*10, fill=color)
            white_counter += 1
            white_counter_label.config(text=f"White Bag: {white_counter} items")
        else:
            canvas.create_oval(500, 300-colored_counter*10, 550, 350-colored_counter*10, fill=color)
            colored_counter += 1
            colored_counter_label.config(text=f"Colored Bag: {colored_counter} items")

        time.sleep(1)

# Run the function in a separate thread to avoid blocking the main thread
thread = Thread(target=add_item_to_bag)
thread.daemon = True
thread.start()

# Start the Tkinter event loop
window.mainloop()
