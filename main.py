from gui import *

from tkinter import *

def main():
    print("Starting Program")

    tk = Tk()
    my_gui = MyGui(tk, "Do stuff with pictures.")
    tk.mainloop()

    print("Ending Program")




if __name__ == "__main__":
    main()
