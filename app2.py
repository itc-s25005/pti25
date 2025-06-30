import tkinter as tk

def dispLabel():
    lbl.configure(text="さようなら")

root = tk.Tk()
root.geometry("200x100")

lbl = tk.Label(text="LABEL")
btn = tk.Button(text="プッシュ", command = dispLabel)

lbl.pack()
btn.pack()
tk.mainloop()

