import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import joblib

best_model = load_model('laundry_sorting_model.h5')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Generate new dataset
data_size = 100
colors = np.random.randint(0, 256, size=(data_size, 3))
data = pd.DataFrame(colors, columns=['red', 'green', 'blue'])
X = data[['red', 'green', 'blue']]

X_train, X_temp, y_train, y_temp = train_test_split(X, np.zeros(data_size), test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, np.zeros(len(X_temp)), test_size=0.5, random_state=42)

X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

user_labels = []


def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % tuple(rgb)


class ColorClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Color Classifier")
        self.root.geometry("400x300")

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Helvetica', 12), padding=10, background='#ffffff')
        style.configure('TLabel', font=('Helvetica', 14), padding=10)
        style.configure('TFrame', padding=20)

        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.label = ttk.Label(main_frame, text="Classify the color shown")
        self.label.pack(pady=10)

        self.color_box = tk.Canvas(main_frame, width=100, height=100, bd=2, relief='solid')
        self.color_box.pack(pady=10)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)

        self.dark_button = ttk.Button(button_frame, text="Dark", command=lambda: self.classify_color('dark'))
        self.dark_button.grid(row=0, column=0, padx=5)

        self.white_button = ttk.Button(button_frame, text="White", command=lambda: self.classify_color('white'))
        self.white_button.grid(row=0, column=1, padx=5)

        self.colored_button = ttk.Button(button_frame, text="Colored", command=lambda: self.classify_color('colored'))
        self.colored_button.grid(row=0, column=2, padx=5)

        self.end_button = ttk.Button(main_frame, text="End Training", command=self.end_training)
        self.end_button.pack(pady=10)

        self.index = 0
        self.show_next_color()

    def show_next_color(self):
        if self.index < len(colors):
            color = colors[self.index]
            self.color_box.create_rectangle(0, 0, 100, 100, fill=rgb_to_hex(color), outline="")
        else:
            messagebox.showinfo("Info", "All colors have been classified")

    def classify_color(self, label):
        global y_train, X_train_scaled, user_labels
        user_labels.append(label)
        y_train = np.append(y_train, label_encoder.transform([label]))
        X_train_scaled = np.append(X_train_scaled, [scaler.transform([colors[self.index]])[0]], axis=0)

        # Incrementally train the model
        best_model.fit(
            np.array([scaler.transform([colors[self.index]])[0]]),
            np.array([label_encoder.transform([label])]),
            epochs=1,
            verbose=0
        )

        self.index += 1
        self.show_next_color()

    def end_training(self):
        best_model.save('laundry_sorting_model_final.h5')
        joblib.dump(scaler, 'scaler_final.pkl')
        joblib.dump(label_encoder, 'label_encoder_final.pkl')

        loss, accuracy = best_model.evaluate(X_test_scaled, y_test)
        print(f'Test Accuracy: {accuracy:.2f}')

        messagebox.showinfo("Training Ended", f"Model training ended. Test Accuracy: {accuracy:.2f}")
        self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = ColorClassifierApp(root)
    root.mainloop()
