from model import Model
import socket
import os
import sys
import glob
import time

# dataset load arguments
arg_model = sys.argv[1]
arg_shuffle = int(sys.argv[2])
arg_imagesize = int(sys.argv[3])
#arg_feature_dim = int(sys.argv[4])
arg_port = sys.argv[4:]

# get model_list
exec(open('ModelList', 'r').read())

# dataset list load
train_image_list = glob.glob("/home/SOFA/SSD/merge/shuffle/2_train_image" + "_shuffle"*arg_shuffle + "_224"*arg_imagesize + ".dat")
train_image_list.sort()
train_label_list = glob.glob("/home/SOFA/SSD/merge/shuffle/2_train_label" + "_shuffle"*arg_shuffle + ".dat")
train_label_list.sort()
validation_image_list = glob.glob("/home/SOFA/SSD/merge/shuffle/2_validation_image" + "_shuffle"*arg_shuffle + "_224"*arg_imagesize + ".dat")
validation_image_list.sort()
validation_label_list = glob.glob("/home/SOFA/SSD/merge/shuffle/2_validation_label" + "_shuffle"*arg_shuffle + ".dat")
validation_label_list.sort()

train_image_list = train_image_list[:]
train_label_list = train_label_list[:]
validation_image_list = validation_image_list[:]
validation_label_list = validation_label_list[:]

os.system("rm -rf train_feature_7_*.dat")
os.system("rm -rf train_label_7_*.dat")
os.system("rm -rf test_feature_7.dat")
os.system("rm -rf test_label_7.dat")

# server setting
SERVER = "10.150.21.89"

if arg_port:
	PORT = int(arg_port[0])
else:
	PORT = 25258

if not arg_model in model_list:
	raise ValueError(arg_model+' is not supported.')

start_message = '''
=============================================
       Created by JW, Kim and SG, Oh
           _____  ____  ______      
          / ____|/ __ \|  ____/\    
         | (___ | |  | | |__ /  \   
          \___ \| |  | |  __/ /\ \  
          ____) | |__| | | / ____ \ 
         |_____/ \____/|_|/_/    \_\ 
'''
print(start_message)
print("---------------------------------------------")
#print(f'PORT: {PORT}\nDATA SET:{" shuffled "*arg_shuffle}{"and"*arg_shuffle} image size is {"224"*arg_imagesize}{"227"*(1-arg_imagesize)}')
print("=============================================\n")
if arg_imagesize == 1:
    feature_dim = 224
else:
    feature_dim = 227

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((SERVER, PORT))

SALS_value = int(client.recv(4096).decode())
print("SALS_value:", SALS_value)
get_here = f'dir:{os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))}'
client.sendall(get_here.encode())
server_dir = client.recv(4096).decode()

client.sendall("Feature Extraction Started".encode())
data = client.recv(4096)
print("Receive:", data.decode())
#feature_dim = max(extractor.engine.get_binding_shape(1))
#client.sendall(("shape="+str(extractor.engine.get_binding_shape(1))).encode())

train_inference = 0
train_total = 0
train_count = 0

val_inference = 0
val_total = 0
val_count = 0

start = time.perf_counter()

re_mode = False
ex_mode = False

extractor = Model(engine_path="./engine/"+arg_model+".engine", feature_dim = feature_dim, server_dir=server_dir[4:])
print(validation_image_list)
print(train_image_list)
for idx, path in enumerate(validation_image_list):
    print(path)
    class_inference, class_total, class_count = extractor.extract_features(path, validation_label_list[idx], "test_feature_7.dat", "test_label_7.dat", client, append=True, IsExtra=True, extra_num = 3750)
    val_inference += class_inference
    val_total += class_total
    val_count += class_count

for idx, path in enumerate(train_image_list):
    print(path)
    class_inference, class_total, class_count, extra_num, extra_path, re_num, extra_label_path, last_feature_path, last_label_path = extractor.extract_features(path, train_label_list[idx], "None", "None", client, append=True, IsExtra=False, extra_num = 0, split=SALS_value)
    # extra path, extra number get
    train_inference += class_inference
    train_total += class_total
    train_count += class_count

if re_num != 0:
    re_mode = True
if extra_num != 0:
    ex_mode = True
#print("re_mode:", re_mode, "ex_mode:", ex_mode)
if ex_mode == True:
    class_inference, class_total, class_count = extractor.extract_features(extra_path, extra_label_path, last_feature_path, last_label_path, client, append=True, IsExtra=True, extra_num = extra_num, split=1)
    train_inference += class_inference
    train_total += class_total
    train_count += class_count


end = time.perf_counter()

inference = train_inference+val_inference
total = train_total+val_total
count = train_count + val_count
print("=================SOFA SSD Result===============")
print("Inference:", inference, count/inference)
print("Feature Extraction:", total, count/total)
print("Total:", end-start, count/(end-start))
print("===============================================")
extract_time = str(total)
while True:
    data = client.recv(4096)
    data = data.decode()
    if data == "Classifier Transmission Started":
        print("Receive:", data)
        # client.sendall(data.encode())
        break


while True:
    if os.path.isfile('classifier.dat'):
        while True:
            data = client.recv(4096)
            data = data.decode()
            if data == "Classifier Transmission Ended":
                print("Receive:", data)
                break
    else:
        continue
    break

client.sendall(extract_time.encode())
client.close()
extractor.ssh.close()
del extractor

