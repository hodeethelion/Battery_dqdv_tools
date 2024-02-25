from tkinter import *
from tkinter import ttk
from dq_dv import dqdv
import os

def data_print(*args):
  user_file_name = file_name.get() + '.xlsx'
  user_sheet_name = sheet_name.get()
  user_active_weight = float(active_weight.get())
  org_path = os.getcwd()
  sample_dqdv_set = dqdv(path_excel=org_path+'/'+user_file_name,
                        sheet_name=user_sheet_name,
                        threshold=100
                        )
  sample_dqdv_set.save_excel()
  sample_dqdv_set.check_galvano_profile()
  sample_dqdv_set.check_dqdv()
  for sl_n in [100,125,150,175,200,225,250,275,300]:
    sample_dqdv_set.check_dqdv(slice_number=sl_n)
  

# 처음 windo 잡아 놓기
root = Tk()
root.title("dq/dv Data Sampling Program")
# root.geometry("640x400+100+100") # 사이즈 잡아놓기
root.resizable(True, True) # 사이즈 조정 가능

mainframe = ttk.Frame(root, padding='3 3 12 12')
# grid로 무엇을 잡아 놓는다는 거지?
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

ttk.Label(mainframe, text="파일 이름").grid(column=1, row=1, sticky=W)

# 1. File 이름 정리하기
# creating new instance of stringvar class
file_name = StringVar()
file_entry = ttk.Entry(mainframe, 
                      width=20, 
                      textvariable=file_name)
file_entry.grid(column=2, row=1, sticky=(W, E))

# 2. Sheet 이름 정리하기
ttk.Label(mainframe, text="시트 이름").grid(column=1, row=2, sticky=W)
sheet_name = StringVar()
sheet_entry = ttk.Entry(mainframe, 
                      width=20,
                      textvariable=sheet_name)
sheet_entry.grid(column=2, row=2, sticky=(W, E))

# 2. Active weight 정리하기
ttk.Label(mainframe, text="Active weight(소수점 포함)").grid(column=1, row=3, sticky=W)
active_weight = StringVar()
active_weight_entry = ttk.Entry(mainframe,
                                width=20,
                                textvariable=active_weight)
active_weight_entry.grid(column=2, row=3, sticky=(W, E))

# ttk.Label(mainframe, textvariable=sheet_name).grid(column=2, row=2, sticky=(W, E))

# 버튼 만들기
ttk.Button(mainframe,
          text="데이터 정리하기", 
          command=data_print).grid(column=3, row=4, sticky=W)
for child in mainframe.winfo_children(): 
    child.grid_configure(padx=5, pady=5)
file_entry.focus()

root.mainloop()