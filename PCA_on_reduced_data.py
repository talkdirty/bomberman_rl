import pandas
import os

def make_dataframe(pickle_data):
    return(pandas.read_pickle(pickle_data, compression='infer', storage_options=None))



directory = os.fsencode("reduced_supervised_data")

frames = []    
for id,file in enumerate(os.listdir(directory)):
     filename = os.fsdecode(file)
     if filename.endswith(".xz"): 
         frames.append(make_dataframe("reduced_supervised_data/"+filename))
         continue
     else:
         continue

print(frames[0])
print(frames[1])