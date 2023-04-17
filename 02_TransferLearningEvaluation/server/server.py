from classifier import Classifier
from utils.load_module import *
import time, os
import argparse

os.system("rm -rf train_feature_*")
os.system("rm -rf train_label_*")
os.system("rm -rf test_feature_*")
os.system("rm -rf test_label_*")

### Setting GPU number
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

### Get system parameters
parser = argparse.ArgumentParser()
parser.add_argument('--load_balance', '-l', type=str2bool, default=False, help='Distributes the load among each SOFA SSDs\nDefault value is False')
parser.add_argument('--split_number', '-s', type=int, default=1, help='Set the pipelining strength\nDefault value is 1')
parser.add_argument('--num_of_client', '-n', type=int, default=1, help='The number of SOFA SSDs\nDefault value is 1')
parser.add_argument('--port', '-p', type=int, default=25258, help='Set the socket connection port\nDefault value is 25258')
args = parser.parse_args()

### Set communication to clients(SOFA SSDs)
comm = CommUnit(args.load_balance, args.split_number, args.num_of_client, args.port, client=CLIENT)
comm.get_SSD_path()
comm.send_message(f'dir:{os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))}')
comm.get_message("Feature Extraction Started")
comm.load_balancing()

train_feature_path = []
train_label_path = []
test_feature_path = []
test_label_path = []

for i in range(comm.split_number):
    for j in range(comm.number_of_client):
        if i == 0:
            test_feature_path.append(f'test_feature_{comm.SSD_numbers[j]}.dat')
            test_label_path.append(f'test_label_{comm.SSD_numbers[j]}.dat')
        train_feature_path.append(f'train_feature_{comm.SSD_numbers[j]}_{i}.dat')
        train_label_path.append(f'train_label_{comm.SSD_numbers[j]}_{i}.dat')

total_image_path = train_feature_path + test_feature_path
model_path = None
#print(total_image_path)
for i in range(comm.split_number):
    print(i, 'loop...')
    flag = 0
    for j in range(comm.number_of_client):
        past_size = -1
        ### Check ending of feature extraction
        while True:
            if os.path.isfile(train_feature_path[i*comm.number_of_client+j]) and os.path.isfile(train_label_path[i*comm.number_of_client+j]):
                current_size = os.path.getsize(train_feature_path[i*comm.number_of_client+j])
                if i == 0 and flag == 0:
                    time.sleep(0.5)
                    flag = 1
                if current_size == past_size:
                    break
                else:
                    past_size = current_size
                    time.sleep(0.5)

    ### Classifier start
    clsfier = Classifier(num_classes=300, feature_dim=comm.feature_dim,
                        train_feature_path=train_feature_path[i*comm.number_of_client:(i+1)*comm.number_of_client],
                        train_label_path=train_label_path[i*comm.number_of_client:(i+1)*comm.number_of_client],
                        test_feature_path=test_feature_path, test_label_path=test_label_path, lb = comm.split_number, loop = i)
    save_path, accuracy = clsfier.train("classifier.dat", model_path)
    model_path = save_path

### Classifier Transmission
comm.send_message("Classifier Transmission Started")
#for i in range(comm.number_of_client):
#    os.system(f"scp classifier.dat SOFA@{comm.addr[i][0]}:{comm.client_path[i]}")
#print(f"scp clas:qsifier.dat SOFA@{comm.addr[i][0]}:{comm.client_path[i]}")
comm.send_message("Classifier Transmission Ended")

extract_time_list = []
for i in range(comm.number_of_client):
    extract_time_list.append(float((comm.client[i].recv(4096)).decode()))

# result print
print('SOFA (LB+ SALS+) Experiment Information')
print("Feature extraction time (sec):              ", max(extract_time_list)+1)
print("Feature extrainction throughput (image/sec):", 1/((max(extract_time_list)+1)/sum(comm.num_images_list)))
print("Model Accuracy:                             ", str(accuracy*100)+'%')
