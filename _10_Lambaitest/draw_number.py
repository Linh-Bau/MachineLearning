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

"""
import tkinter as tk

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title('draw_number')
        
        # tạo giao diện
        self.geometry("400x300")

        #khởi tạo các widgets
        self.create_widgets()

    def create_widgets(self):
        main_frame=tk.Frame(self)
        main_frame.pack(padx=10,pady=10)

        #lưới con 1
        grid1=tk.Canvas(main_frame,bg='light blue')
        grid1.grid(row=0,column=0,padx=(0,5), sticky='nsew')
        main_frame.columnconfigure(0,weight=1)
        main_frame.rowconfigure(0,weight=1)

        # Lưới con 2
        grid2 = tk.Frame(main_frame)
        grid2.grid(row=0, column=1, padx=(5, 0))

        #lưới chứa các option, thông tin ... nằm ở trên top
        option_frame=tk.Frame(grid2)
        option_frame.pack(expand=True)

        #frame chứa các button nằm ở dưới cùng
        bt_frame=tk.Frame(grid2)
        bt_frame.pack()
        button1 = tk.Button(bt_frame, text="Button 1")
        button1.pack(pady=10)

        button2 = tk.Button(bt_frame, text="Button 2")
        button2.pack(pady=10)

        






if __name__=="__main__":
    app=App()
    app.mainloop()