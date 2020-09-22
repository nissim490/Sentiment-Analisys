import tkinter
from tkinter import messagebox
 #GUI

def show(run_func):
    root = tkinter.Tk()
    root.title('LSN')
    root.geometry('495x200')
    root.resizable(False, False)

    # Input field for tweet
    input_field = tkinter.Entry(root)
    input_field.grid(row=0, column=0, padx=10, pady=30, ipady=30, ipadx=150)
    input_field.focus_set()

    def handle_click():
        sentence = input_field.get()
        result = run_func(sentence)
        messagebox.showinfo('Result', '{}\n{}'.format(result[1], result[0]))

    # Button for running the model
    run_button = tkinter.Button(root, text='Run', width=10, command=handle_click)
    run_button.grid(row=1, column=0, padx=10, pady=0)

    tkinter.mainloop()
