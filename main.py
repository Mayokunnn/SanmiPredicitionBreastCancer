import tkinter as tk
from tkinter import messagebox
from calc import calculate_real_size
from db import save_to_db


def calculate_and_save():
    try:
        username = name_entry.get()
        image_size = float(size_entry.get())
        magnification = float(mag_entry.get())

        real = calculate_real_size(image_size, magnification)
        if isinstance(real, str):
            result_label.config(text=f"Error: {real}")
        else:
            result_label.config(text=f"Real Size: {real} μm")
            save_to_db(username, image_size, magnification, real)
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numbers for size and magnification.")


root = tk.Tk()
root.title("Microscope Specimen Size Calculator")

tk.Label(root, text="Username").pack()
name_entry = tk.Entry(root)
name_entry.pack()

tk.Label(root, text="Image Size (μm)").pack()
size_entry = tk.Entry(root)
size_entry.pack()

tk.Label(root, text="Magnification").pack()
mag_entry = tk.Entry(root)
mag_entry.pack()

tk.Button(root, text="Calculate", command=calculate_and_save).pack(pady=10)

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()
