"""
 POC Composer!!
Date : 10/ 29/2019

"""
#!/usr/bin/python
from ML_package_Accessor import POC_Accessor



def  main():
    POC_Accessor.read_DATA_FromXLSX()
    POC_Accessor.Xlsxoader_Printer()
    POC_Accessor.loadinputMatrix()
    print("Main")

if __name__ == "__main__":
    main()
    


 
    
    