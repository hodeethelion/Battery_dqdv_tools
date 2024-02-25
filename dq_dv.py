# dq_dv.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import CubicSpline

# class read_excel로 읽어오는 dataframe 자체
class dqdv:
  # 초기화
  def __init__(self, path_excel='', sheet_name=1, active_weight=3.438, threshold=100):
    ''' 초기화함수의 input 정리
    파일 이름: 엑셀파일 이름과 디렉토리
    파일에 있는 시트 번호: 엑셀 파일에서의 시트 번호
    active weight 실험에서의 무게값: 경은 누나가 말하는 실험 장비에서 무게값을 뽑아 낼 수 있다고 말함
    threshold: 갑자기 바뀌는 정도를 capacity에서 측정할 수 있는 값
    '''
    self.dataname = os.path.splitext(os.path.basename(path_excel))[0]
    self.dataframe = pd.read_excel(path_excel, sheet_name = sheet_name)
    self.dataframe['Capacity(mAh/g)'] = self.dataframe.iloc[:, 1]* (10**6) / active_weight
    self.threshold = threshold
    self.initialize_chargeflags()
    self.make_newdf()
    
  # 자기 자신 확인 
  def __str__(self):
    '''어떤 데이터 파일인지 표시
    '''
    return f"Data from {self.dataname} excel file"
  
  # 주어진 그래프로 Q V 예시 그림 확인하기
  def check_galvano_profile(self, save_path='./output/galvano_sample.png'):
    '''갈바노 스태틱 샘플 그래프
    return: plotting graph & savefigure
    '''
    q_ah = self.dataframe['Capacity(mAh/g)']
    voltage = self.dataframe.iloc[:, 2]
    fig, ax = plt.subplots()
    ax.scatter(q_ah, voltage, marker='o', color='blue', s=8)
    ax.set_xlabel('Capacity(mAh/g)');ax.set_ylabel('Voltage(V)')
    ax.set_title('Galvano profile')
    fig.savefig(save_path)
    return 
  
  # dataframe이 어디서 야무지게 잘리는 지 판별
  def find_changepoint(self, threshold=100):
    ''' 변화포인트 찾기
    return:s int
    '''
    differences = np.abs(np.diff(self.dataframe['Capacity(mAh/g)']))
    change_point = np.where(differences > threshold)[0] + 1
    return change_point[0]
  
  # dataframe에 판별 후 데이터에 flag 맥이는 방식
  def initialize_chargeflags(self):
    change_point = self.find_changepoint()
    self.dataframe['Charge'] = 0
    self.dataframe.loc[:(change_point-1), 'Charge'] = 1
  
  # 새로운 dataframe 만들기
  def make_newdf(self):
    new_df = pd.DataFrame()
    self.seperated_df = new_df
    charge_condition = self.dataframe['Charge'] == 1
    discharge_condition = self.dataframe['Charge'] == 0
    
    # Charging data 넣어놓기
    new_df['|Q|[Ah](charge)'] = self.dataframe[charge_condition]['|Q|(Ah)']
    new_df['Voltage[V](charge)'] = self.dataframe[charge_condition]['Voltage(V)']
    new_df['Capacity[mAh/g](charge)'] = self.dataframe[charge_condition]['Capacity(mAh/g)']
    # Discharging data 한번 바꿔놓기
    discharge_data = self.dataframe[discharge_condition].copy()
    discharge_data.reset_index(drop=True, inplace = True)
    # Discharging data 넣어놓기
    new_df['|Q|[Ah](discharge)'] = discharge_data['|Q|(Ah)']
    new_df['Voltage[V](discharge)'] = discharge_data['Voltage(V)']
    new_df['Capacity[mAh/g](discharge)'] = discharge_data['Capacity(mAh/g)']
  
  # 새로운 엑셀 파일 저장하기
  def save_excel(self):
    output_directory = 'output'
    # 만약에 output 디렉토리가 존재하지 않는다면
    if not os.path.exists(output_directory):
      os.makedirs(output_directory)
    file_path = os.path.join(output_directory, 'charge_discharge.csv')
    to_save_file = self.seperated_df
    to_save_file.to_csv(file_path, index=True)
    
  # 해야될 거리
  # 1. sample dq dv 그래프 그리기 --> linespacing에 따라서 그림 여러 개 뽑아 놓기
  def check_dqdv(self, slice_number=150):
    ############################## data refining ##############################
    charge_df = self.seperated_df.iloc[:, :3]
    discharge_df = self.seperated_df.iloc[:, 3:]
    
    charge_df = charge_df.sort_values(by='Voltage[V](charge)')
    discharge_df = discharge_df.sort_values(by='Voltage[V](discharge)')
    
    charge_df = charge_df.drop_duplicates(subset='Voltage[V](charge)')
    discharge_df = discharge_df.drop_duplicates(subset='Voltage[V](discharge)')
    
    charge_voltage = charge_df['Voltage[V](charge)'].tolist()
    charge_capacity = charge_df['Capacity[mAh/g](charge)'].tolist()
    
    discharge_voltage = discharge_df['Voltage[V](discharge)'].tolist()
    discharge_capacity = discharge_df['Capacity[mAh/g](discharge)'].tolist()
    
    valid_indices = ~np.isnan(discharge_voltage)
    discharge_voltage = np.array(discharge_voltage)[valid_indices]
    discharge_capacity = np.array(discharge_capacity)[valid_indices]
    ##########################################################################
    
    # cubic spline interpolation: voltage & capacity 
    # 첫번째 voltage, capacity 빈공간이 너무 많으니 많이 채워서 만들어 주기
    spl_charge = CubicSpline(charge_voltage, charge_capacity)
    voltage_lin = np.linspace(min(charge_voltage), max(charge_voltage), slice_number)
    capacity_lin = spl_charge(voltage_lin)

    spl_discharge = CubicSpline(discharge_voltage, discharge_capacity)
    voltage_dc_lin = np.linspace(min(discharge_voltage), max(discharge_voltage), slice_number)
    capacity_dc_lin = spl_discharge(voltage_dc_lin)
    
    
    # 두번째 채워진 voltage capacity 그래프를 통해서 미분을 시행함
    spl_charge2 = CubicSpline(voltage_lin, capacity_lin)
    derivative_charge = spl_charge2.derivative(nu=1)
    
    spl_discharge2 = CubicSpline(voltage_dc_lin, capacity_dc_lin)
    derivative_discharge = spl_discharge2.derivative(nu=1)
    
    
    ############################################################
    ################### plotting part ##########################
    ############################################################
    
    fig, ax = plt.subplots()
    ax.scatter(voltage_lin, derivative_charge(voltage_lin), color='orange')
    ax.scatter(voltage_dc_lin, derivative_discharge(voltage_lin), color='red')
    plt.axhline(y=0, color='gray', linewidth=3)
    ax.set_ylabel('dq/dv')
    ax.set_xlabel('voltage')
    
    # save_path=f'./output/dqdv/dqdv_{str(slice_number)}.png'
    
    dqdv_dir = 'output/dqdv'
    
    if not os.path.exists(dqdv_dir):
      os.makedirs(dqdv_dir)
    file_path = os.path.join(dqdv_dir,f'dqdv_{slice_number}.png')
    plt.savefig(file_path)
    return 
  # 2. capacity voltage 그림 그리고 뽑아 놓기 --> 이쁘게
  # 3. capacity voltage 그림에서 추가적으로 데이터 샘플링 해 놓기