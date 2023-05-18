import tkinter as tk

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title('draw_number')
        
        # tạo giao diện
        self.geometry("400x300")


if __name__=="__main__":
    app=App()
    app.mainloop()