import time
import tkinter as tk
from threading import Thread
from tkinter import Canvas, messagebox

import joblib
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("laundry_sorting_model_final.h5")
scaler = joblib.load("scaler_final.pkl")
label_encoder = joblib.load("label_encoder_final.pkl")


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
canvas = Canvas(window, width=900, height=600)
canvas.pack()

# Draw the bags
dark_bag = canvas.create_rectangle(100, 200, 300, 550, fill="gray", tags="dark_bag")
canvas.create_text(200, 180, text="Dark Bag")

white_bag = canvas.create_rectangle(
    350, 200, 550, 550, fill="lightgray", tags="white_bag"
)
canvas.create_text(450, 180, text="White Bag")

colored_bag = canvas.create_rectangle(
    600, 200, 800, 550, fill="orange", tags="colored_bag"
)
canvas.create_text(700, 180, text="Colored Bag")


# Function to animate the falling of items with better gravity and bouncing
def animate_fall(
    item, start_x, target_x, target_y, bag_left, bag_right, is_white=False
):
    x, y, _, _ = canvas.coords(item)
    dx = (target_x - start_x) / 50
    dy = 0
    gravity = 0.5
    bounce = -0.7
    friction = 0.9
    min_bounce_threshold = 0.5
    floor_level = target_y + 50

    if is_white:
        # Introduce randomness to horizontal movement for white items
        dx += np.random.uniform(-2, 2)

    while True:
        x += dx
        y += dy
        dy += gravity

        # Check for collision with floor
        if y + 50 >= floor_level:
            y = floor_level - 50
            dy *= bounce

            # Add damping to stop bouncing
            if abs(dy) < min_bounce_threshold:
                dy = 0

        # Apply friction to horizontal movement only when the item is on the ground
        if y + 50 >= floor_level - 1:
            dx *= friction

        # Check for collision with side walls
        if x <= bag_left:
            x = bag_left
            dx *= bounce
        elif x + 50 >= bag_right:
            x = bag_right - 50
            dx *= bounce

        canvas.coords(item, x, y, x + 50, y + 50)
        window.update()
        time.sleep(0.001)

        # Stop the loop if both vertical and horizontal movements are minimal
        if abs(dy) < min_bounce_threshold and abs(dx) < min_bounce_threshold:
            break

    # Ensure the item stays within the bag's boundaries after bouncing
    canvas.coords(item, x, floor_level - 50, x + 50, floor_level)


# Function to add a new item to the appropriate bag
def add_item_to_bag():
    global dark_counter, white_counter, colored_counter

    while True:
        r = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        b = np.random.randint(0, 256)
        color = f"#{r:02x}{g:02x}{b:02x}"
        predicted_label = predict_laundry_item(r, g, b)

        # Show the item in the middle above the bags
        item = canvas.create_oval(425, 0, 475, 50, fill=color)
        window.update()
        time.sleep(0.3)

        # Move the item to the appropriate bag
        if predicted_label == "dark":
            dark_counter += 1
            dark_counter_label.config(text=f"Dark Bag: {dark_counter} items")
            animate_fall(item, 425, 200, 500 - dark_counter * 10, 100, 300)
        elif predicted_label == "white":
            white_counter += 1
            white_counter_label.config(text=f"White Bag: {white_counter} items")
            animate_fall(
                item, 425, 450, 500 - white_counter * 10, 350, 550, is_white=True
            )
        else:
            colored_counter += 1
            colored_counter_label.config(text=f"Colored Bag: {colored_counter} items")
            animate_fall(item, 425, 700, 500 - colored_counter * 10, 600, 800)

        time.sleep(0.3)


# Run the function in a separate thread to avoid blocking the main thread
thread = Thread(target=add_item_to_bag)
thread.daemon = True
thread.start()

# Start the Tkinter event loop
window.mainloop()
