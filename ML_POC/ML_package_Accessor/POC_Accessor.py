"""
Author: Barun Prakash
Date:  25/10/2019
objective: poc Handler !!



"""
#!/usr/bin/python3
import xlrd

Xlsxoader ={}
input_Target=[[0]*3]*10  #  3 col ,10 30ws
output_Target=[[0]*2]*10




def read_DATA_FromXLSX():
    
    # Give the location of the file 
    loc = ("BuildAndConfig/EvaluationData.xlsx") 
    wb = xlrd.open_workbook(loc) 
    sheet = wb.sheet_by_index(0) 
    sheet.cell_value(0, 0) 
 
    for i in range(10):
        Xlsxoader[i]= sheet.row_values(i)
        print(sheet.row_values(i)) 
        
#######################################

# Load Data into python
def Xlsxoader_Printer():
    for i in range(10):
        print(Xlsxoader[i])
        


######################################
def loadinputMatrix():
    for x in Xlsxoader:
        for y in Xlsxoader[x]:
            input_Target[x][y]=Xlsxoader[x][y]
            
            
##############################################
def printInputOutputMatrix():
    for x in input_Target:
        for y in input_Target[x]:
            print(y,':',input_Target[x][y])
            
            
##############################################