"""
Mục Tiêu:
1 chương trình sẽ có 2 phần, 1 là make trainning data, 2 là learn and test data
----
1: Make trainning data, sẽ vẽ số theo yêu cầu cho trước, sau đó lưu dữ liệu của số đó vào training test data.
Nội dung là dữ liệu dạng ảnh đen trắng, có dữ liệu chỉ là 1 và 0. Tên file sẽ là dạng số_giờ_phút_giây tạo.
----
2: Learn and test:
Learn: sẽ có học k có bộ lọc và thông qua 1 bộ lọc.
Test: Dùng model thu được thử nghiệm.

Chương trình có 2 mode:
mode 1 sẽ là tạo các file data

"""
import tkinter as tk
from turtle import RawTurtle, TurtleScreen, ScrolledCanvas
from PIL import Image, ImageTk
import datetime
import os
from PIL import EpsImagePlugin

EpsImagePlugin.gs_windows_binary= r'D:\Bau\Storage\DevTools\gs10.01.1\bin\gswin64c'

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title('draw_number')
        
        # tạo giao diện
        self.geometry("600x300")

        #khởi tạo các biến để các widgets sử dụng
        self.rdbt_mode_var=tk.IntVar()
        self.entry_value=tk.StringVar()

        #các event phím tắt
        self.bind("<KeyPress>",self.on_key_preesed)

        #khởi tạo các widgets
        self.create_widgets()


    def on_key_preesed(self,event):
        print(f'pressed key: {event.keysym}')
        if event.keysym=='s':
            if not self.entry_value.get() or len(self.entry_value.get())==0:
                print('entry value not set')
                return
            self.save_drawing()
            self.clear_drawing()
            return
        if event.keysym=='c':
            self.clear_drawing()
            return
        print('key not bind!')
        


    def on_mouse_down(self,event):
        self.draw_canvas.x = event.x
        self.draw_canvas.y = event.y

    def on_mouse_drag(self,event):
        x = event.x
        y = event.y
        self.draw_canvas.create_line(self.draw_canvas.x, self.draw_canvas.y, x, y, width=20)
        self.draw_canvas.x = x
        self.draw_canvas.y = y

        
    def create_widgets(self):
        main_frame=tk.Frame(self)
        main_frame.pack(padx=10,pady=10)
        #lưới con 1
        grid1=tk.Canvas(main_frame,bg='light blue')
        grid1.grid(row=0,column=0,padx=(0,5), sticky='nsew')
        main_frame.columnconfigure(0,weight=1)
        main_frame.rowconfigure(0,weight=1)
        #tạo 1 canvas để vẽ
        self.draw_canvas= tk.Canvas(grid1,bg='light gray')
        self.draw_canvas.grid(sticky='nswe')
        self.draw_canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.draw_canvas.bind('<Button-1>', self.on_mouse_down)
        # Lưới con 2
        grid2 = tk.Frame(main_frame)
        grid2.grid(row=0, column=1, padx=(5, 0), sticky='nsew')
        # label thông tin
        lb_info=tk.Label(grid2, text="Handwritten number recognition")
        lb_info.grid(row=0,column=0,sticky='nswe')
        # Tạo LabelFrame với text là tên groupbox
        groupbox = tk.LabelFrame(grid2, text="MODE")
        groupbox.grid(row=1,column=0,sticky='nswe')
        rdbt_mode_1=tk.Radiobutton(groupbox,text="TRAINING",
        variable=self.rdbt_mode_var,
        value=1,
        command=lambda: self.create_widgets_for_mode())
        rdbt_mode_1.grid(row=0,column=0,sticky='w')
        rdbt_mode_2=tk.Radiobutton(groupbox,text="TEST",
        variable=self.rdbt_mode_var,
        value=2,
        command= lambda: self.create_widgets_for_mode())
        rdbt_mode_2.grid(row=1,column=0,sticky='w')
        # Tạo LabelFrame sẽ tương ứng với các mode
        self.mode_box = tk.LabelFrame(grid2)
        self.mode_box.grid(row=2,column=0,sticky='nswe')


    def create_widgets_for_mode(self):
        for widget in self.mode_box.winfo_children():
            widget.destroy()
        if self.rdbt_mode_var.get()==1:
            self.create_mode1_widgets()
        pass
    

    def create_mode1_widgets(self):
        p_frame = self.mode_box
        p_frame['text'] = 'MODE 1: Create Training Data'
        lb_if = tk.Label(p_frame, text='Number')
        lb_if.pack(side='top',fill='x')
        input_textbox = tk.Entry(p_frame,textvariable=self.entry_value)
        input_textbox.pack(side='top',fill='x')
        bt_clear = tk.Button(p_frame, text='Clear Drawing',
        command=lambda: self.clear_drawing())
        bt_clear.pack(side='top',fill='x')
        bt_save = tk.Button(p_frame, text='Save Data', command=lambda: self.save_drawing())
        bt_save.pack(side='top',fill='x')


    def clear_drawing(self):
        print('clearn canvas!')
        self.draw_canvas.delete('all')


    def save_drawing(self):
        epsscript='eps_script.eps'
        self.draw_canvas.postscript(file=epsscript,colormode='color')
        image=Image.open(epsscript)
        print(f'open file eps success!')
        filename=f"./training_data/{self.entry_value.get()}_{datetime.datetime.now().strftime('%H_%M_%S')}.jpg"
        print(f'save file: {filename}')
        image.save(filename,"JPEG")

if __name__=="__main__":
    app= App()
    app.mainloop()