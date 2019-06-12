import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import time
from sklearn.decomposition import PCA

def load_data(rows = 710000):
    Attributes = pd.read_csv("amzn-anon-access-samples-2.0.csv",usecols=["PERSON_BUSINESS_TITLE",
    "PERSON_BUSINESS_TITLE_DETAIL","PERSON_COMPANY","PERSON_DEPTNAME","PERSON_ID","PERSON_JOB_CODE","PERSON_JOB_FAMILY"
    ,"PERSON_MGR_ID","PERSON_ROLLUP_1","PERSON_ROLLUP_2","PERSON_ROLLUP_3"])
    history = pd.read_csv("amzn-anon-access-samples-history-2.0.csv",usecols=["ACTION","TARGET_NAME","LOGIN"],nrows=rows)
    df_attributes = pd.DataFrame(Attributes)
    df_history = pd.DataFrame(history)
    df_history["ACTION"] = (df_history["ACTION"] == "add_access") + 0
    df_history.sort_values(by="ACTION",inplace=True)
    df_history.drop_duplicates(subset=["TARGET_NAME","LOGIN"],keep="first",inplace=True)

    row_data = pd.merge(df_history,df_attributes,how="left",left_on="LOGIN",right_on="PERSON_ID")
    row_data.drop(["LOGIN","PERSON_ID"],axis=1,inplace=True)

    negative_num = 0
    for i in list(row_data["ACTION"]):
        if i == 0:
            negative_num += 1
    print("negative_num",negative_num)

    number, attributes_number = row_data.shape
    return number,attributes_number-1,row_data

def process_x_onehot(total_x):
    row,col = total_x.shape
    for i in range(1,col):
        index = 0
        d = dict()
        print("here"+str(i))
        for j in range(row):
            if total_x.iloc[j,i] not in d:
                d[total_x.iloc[j,i]] = index
                total_x.iloc[j,i] = index
                index += 1
            else:
                total_x.iloc[j,i] = d[total_x.iloc[j,i]]
        # pd.DataFrame(d.items()).to_csv("map"+str(i)+".csv")
    total_x.to_csv("row_data.csv")
    total_x.sort_values(by="ACTION", inplace=True)
    total_x.iloc[:9600,:].to_csv("negative_data.csv")

    return

def to_onehot(batch_x):
    one_hot_list = []
    m = batch_x.shape[0]
    for i in range(len(length)):
        batch_x_onehot = torch.zeros(m, length[i]).cuda().scatter_(1, batch_x[:, i].unsqueeze(1), 1)
        one_hot_list.append(batch_x_onehot)
    return one_hot_list

def load_data_new(nrow):
    f = pd.read_csv("row_data.csv",nrows=nrow)
    row_data = pd.DataFrame(f)
    print("data_shape",row_data.shape)
    return row_data.shape[0],row_data.shape[1]-1,row_data


class Net(nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim):
        super(Net,self).__init__()
        self.layer1 = nn.Linear(in_dim,hidden_dim)
        self.layer2 = nn.Linear(hidden_dim,out_dim)
        self.dp = nn.Dropout(0.5)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        # x = self.dp(x)
        x = self.layer2(x)
        x = torch.sigmoid(x)
        return x


if __name__ == "__main__":
    start = time.time()
    number, attributes_number, row_data = load_data_new(20000)
    row_data = shuffle(row_data)

    total_x, total_y = row_data.iloc[:, 1:], row_data[["ACTION"]].values.reshape((number, 1))
    total_y = pd.DataFrame(total_y)

    length = []
    for i in range(attributes_number):
        length.append(len(set(list(total_x.iloc[:, i]))))
    print("length", length)
    print("sum_length",sum(length))

    np.save("length.npy",np.array(length))

    train_x, train_y = total_x.iloc[2000:, :], total_y.iloc[2000:, :]
    test_x, test_y = total_x.iloc[:2000, :], total_y.iloc[:2000, :]

    train_number = train_x.shape[0]

    train_x = torch.from_numpy(np.array(train_x.values,dtype=np.float32)).cuda()
    train_y = torch.from_numpy(np.array(train_y.values,dtype=np.float32)).cuda()
    test_x = torch.from_numpy(np.array(test_x.values,dtype=np.float32)).cuda()
    test_y = torch.from_numpy(np.array(test_y.values,dtype=np.float32)).cuda()

    batch_size = 1000
    learning_rate = 1e-2
    epoches = 100

    net = Net(sum(length),64,1).cuda()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.8)

    torch_dataset = Data.TensorDataset(train_x,train_y)
    loader = Data.DataLoader(dataset=torch_dataset,batch_size=batch_size,shuffle=True)
    best_acc = 0
    for epoch in range(epoches):
        total_loss,count,acc = 0,0,0
        scheduler.step()
        for step,(batch_x,batch_y) in enumerate(loader):
            num = batch_x.shape[0]
            count += 1
            batch_x = batch_x.long()
            batch_y.squeeze(1)
            x = Variable(torch.cat(to_onehot(batch_x),dim=1))
            y = Variable(batch_y)
            optimizer.zero_grad()

            prediction = net(x)
            loss = criterion(prediction,y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            mask = prediction.ge(0.5).float()
            count += (mask == y).sum().item()

        if epoch % 10 == 0:
            acc = count/train_number
            if acc > best_acc:
                best_acc = acc
                torch.save(net.state_dict(), "net.pt")
            print("train  epoch{} : loss = {:.6f}  acc = {:.6f}".format(epoch,total_loss/(train_number//batch_size),acc))

    print("---------------test---------------")

    net.eval()
    correct_num = 0
    test_batch_size = 100
    torch_dataset_test = Data.TensorDataset(test_x, test_y)
    test_loader = Data.DataLoader(dataset=torch_dataset_test,batch_size=test_batch_size,shuffle=False)
    for (x,y) in test_loader:
        x = x.long()
        y.squeeze(1)
        x_onehot = Variable(torch.cat(to_onehot(x),dim=1))
        y = Variable(y)
        out = net(x_onehot)
        # print("out",torch.mean(out))
        mask = out.ge(0.5).float()
        correct_num += (mask == y).sum().item()
    print("test acc = {:.6f}".format(correct_num/test_x.shape[0]))
    end = time.time()
    print("finish time =",end-start)


