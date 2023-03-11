import argparse
from torch import nn as nn
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
import numpy.random as npr
import numpy as np
import sys
from itertools import chain
import copy
import math
from resnet import ResNet18
from torch.utils.data import Dataset
import pickle
device = torch.device("cuda")


class CustomDataset(Dataset):
    def __init__(self) -> None:
        super(CustomDataset, self).__init__()
        self.targets = None
        self.data = None

    def add_items(self, inputs, targets):
        if isinstance(self.targets, np.ndarray) == False:
            self.targets = np.array(targets)
            self.data = np.array(inputs)
        else:
            self.targets = np.append(self.targets, targets)
            self.data = np.append(self.data, inputs, axis = 0)
    
    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    
    def __len__(self):
        return len(self.data)


def build_new_train_dataset(args, train_dataset, trainset_permutation_inds):
    global train_acc_list, train_loss_list, test_acc_list, test_loss_list, threshold_epoch, visited_indexes, min_max_data_nums_per_class, threshold_occupation, epoch_accuracy_matrix, sorted_result_angles,\
            batch_to_filter, filter_button, class_nums_counter, unforgettable_idx, start_new_dataloader_min_nums, threshold_test_accuracy_to_build_new_data, \
            n_samples, n_features, last_selected_train_x_indexes, th_win, th_train_acc, th_stable_acc, th_stable_loss
    
    batch_size = args.batch_size
    new_train_dataset = CustomDataset()
    print("building new train_dataset....")
    for batch_index, batch_start_idx in  enumerate(range(0, len(train_dataset.targets), batch_size)):
        batch_inds = trainset_permutation_inds[batch_start_idx: batch_start_idx + batch_size]
        elem_to_remove = np.setdiff1d(batch_inds.tolist(), list(visited_indexes))
        batch_inds = np.setdiff1d(batch_inds.tolist(), elem_to_remove.tolist()).astype(int)
        inputs = train_dataset.data[batch_inds]
        targets = np.array(train_dataset.targets)[batch_inds]
        new_train_dataset.add_items(inputs, targets)
    
    train_dataset = new_train_dataset
    print(f"len new_train_dataset.targets-data: {len(train_dataset.targets)}-{len(train_dataset.data)} / {len(visited_indexes)}")
    del new_train_dataset
    
    if args.run_mode == "filter" and args.filter_button == True:
        # 保存从全部数据中筛选出来的下标至filtered_index_file_path
        torch.save(list(visited_indexes), args.filtered_index_file_path)
    return train_dataset


def update_dataloader(test_acc, current_epoch_accuracy_detail_lst, train_dataset, trainset_permutation_inds, args):
    global train_acc_list, train_loss_list, test_acc_list, test_loss_list, threshold_epoch, visited_indexes, min_max_data_nums_per_class, threshold_occupation, epoch_accuracy_matrix, sorted_result_angles,\
            batch_to_filter, filter_button, class_nums_counter, unforgettable_idx, start_new_dataloader_min_nums, threshold_test_accuracy_to_build_new_data, \
            n_samples, n_features, last_selected_train_x_indexes, th_win, th_train_acc, th_stable_acc, th_stable_loss
    
    if threshold_epoch != None and test_acc >= threshold_test_accuracy_to_build_new_data and filter_button == True:
        to_build_new_data = False
        if len(visited_indexes) < start_new_dataloader_min_nums:
            filter_button = False
            print(f"not enough filter data: {len(visited_indexes)} using last_index instead, {len(last_selected_train_x_indexes)}")
            visited_indexes = set([i for i in range(len(last_selected_train_x_indexes))]) # not use all_new_train_data_index = last_selected_train_x_indexes
        else:
            filter_button = False
            to_build_new_data = True
            print(f"len(new_visited_indexes) {len(visited_indexes)}")
            # print(f"new_visited_indexes: {visited_indexes}")

        epoch_accuracy_matrix = np.array(epoch_accuracy_matrix)
        print(f"epoch_accuracy_matrix shape: {epoch_accuracy_matrix.shape}")
        epoch_accuracy_matrix = epoch_accuracy_matrix.T[list(visited_indexes)].T
        epoch_accuracy_matrix = epoch_accuracy_matrix.tolist()

        current_epoch_accuracy_detail_lst = np.array(list(chain(*current_epoch_accuracy_detail_lst)))
        current_epoch_accuracy_detail_lst = current_epoch_accuracy_detail_lst[list(visited_indexes)]

        epoch_accuracy_matrix.append(current_epoch_accuracy_detail_lst)

        current_epoch_accuracy_detail_lst = current_epoch_accuracy_detail_lst.tolist()
        print(f"append finished, epoch_accuracy_matrix shape: {np.array(epoch_accuracy_matrix).shape}\n")


        if to_build_new_data: train_dataset = build_new_train_dataset(args, train_dataset, trainset_permutation_inds)
        last_selected_train_x_indexes = visited_indexes
    else:
        current_epoch_accuracy_detail_lst = list(chain(*current_epoch_accuracy_detail_lst))
        epoch_accuracy_matrix.append(current_epoch_accuracy_detail_lst)
        print("\n")


def update_threshold_epoch(epoch):
    global train_acc_list, train_loss_list, test_acc_list, test_loss_list, threshold_epoch, visited_indexes, min_max_data_nums_per_class, threshold_occupation, epoch_accuracy_matrix, sorted_result_angles,\
            batch_to_filter, filter_button, class_nums_counter, unforgettable_idx, start_new_dataloader_min_nums, threshold_test_accuracy_to_build_new_data, \
            n_samples, n_features, last_selected_train_x_indexes, th_win, th_train_acc, th_stable_acc, th_stable_loss
    
    if len(train_acc_list) > th_win and threshold_epoch == None:
        temp_acc_train_list, temp_loss_train_list = sorted(train_acc_list[- th_win - 1: -1]), sorted(train_loss_list[- th_win - 1: -1])
        min_acc_train, max_acc_train, mediean_acc_train = temp_acc_train_list[0], temp_acc_train_list[-1], sum(temp_acc_train_list) / th_win
        min_loss_train, max_loss_train = temp_loss_train_list[0], temp_loss_train_list[-1]
        delta_acc_train = max_acc_train - min_acc_train
        delta_loss_train = max_loss_train - min_loss_train
        print(f"delta_acc_train: {delta_acc_train}\ndelta_loss_train: {delta_loss_train}\nmin_loss_train: {min_loss_train}\nmax_loss_train: {max_loss_train}\nmin_acc_train: {min_acc_train}\nmax_acc_train: {max_acc_train}\nmediean_acc_train: {mediean_acc_train}\n")
        if mediean_acc_train > th_train_acc:
            threshold_epoch = epoch
            print("acc")
        elif delta_acc_train < th_stable_acc and delta_loss_train < th_stable_loss:
            threshold_epoch = epoch
            print("delta")


def check_full():
    global train_acc_list, train_loss_list, test_acc_list, test_loss_list, threshold_epoch, visited_indexes, min_max_data_nums_per_class, threshold_occupation, epoch_accuracy_matrix, sorted_result_angles,\
            batch_to_filter, filter_button, class_nums_counter, unforgettable_idx, start_new_dataloader_min_nums, threshold_test_accuracy_to_build_new_data, \
            n_samples, n_features, last_selected_train_x_indexes, th_win, th_train_acc, th_stable_acc, th_stable_loss
    
    for k, v in class_nums_counter.items():
        if v < min_max_data_nums_per_class[1]:
            return False
    return True


def filter_train_data_by_angle(sorted_result_angle, current_epoch_all_train_y):
    global train_acc_list, train_loss_list, test_acc_list, test_loss_list, threshold_epoch, visited_indexes, min_max_data_nums_per_class, threshold_occupation, epoch_accuracy_matrix, sorted_result_angles,\
            batch_to_filter, filter_button, class_nums_counter, unforgettable_idx, start_new_dataloader_min_nums, threshold_test_accuracy_to_build_new_data, \
            n_samples, n_features, last_selected_train_x_indexes, th_win, th_train_acc, th_stable_acc, th_stable_loss
    ratio = min_max_data_nums_per_class[1] * 10 / 50000
    max_pick_nums = int(ratio * len(sorted_result_angle))
    # sorted_result_angle:
    # v1. (angle1, index_other1, index_me), (angle2, index_other2, index_me) ... sorted angle

    me_index = sorted_result_angle[-1][2]
    cnt_pick_nums = 0
    if me_index in unforgettable_idx: #当前样本是unforgettable，sorted_result_angle里由大到小（从不相似到相似），只要和unforgettable不相似的
        for i in range(len(sorted_result_angle) - 1, 0, -1):
            current_result_index = sorted_result_angle[i][1]
            current_result_real_class = current_epoch_all_train_y[current_result_index]
            if class_nums_counter[current_result_real_class] < min_max_data_nums_per_class[1] and current_result_index not in visited_indexes and cnt_pick_nums < max_pick_nums:
                visited_indexes.add(current_result_index)
                class_nums_counter[current_result_real_class] += 1
                cnt_pick_nums += 1
    print(f"current sorted_angle_list cnt_pick_nums: {cnt_pick_nums} - max_pick_nums: {max_pick_nums}")


def filter_train_data_by_angle_one_batch(sorted_result_angles_one_batch, current_epoch_all_train_y):
    global train_acc_list, train_loss_list, test_acc_list, test_loss_list, threshold_epoch, visited_indexes, min_max_data_nums_per_class, threshold_occupation, epoch_accuracy_matrix, sorted_result_angles,\
            batch_to_filter, filter_button, class_nums_counter, unforgettable_idx, start_new_dataloader_min_nums, threshold_test_accuracy_to_build_new_data, \
            n_samples, n_features, last_selected_train_x_indexes, th_win, th_train_acc, th_stable_acc, th_stable_loss
    
    [filter_train_data_by_angle(sorted_result_angles_one_batch[i], current_epoch_all_train_y)\
    for i in range(len(sorted_result_angles_one_batch))]
  


def calculate_angle_update_visited_index(last_index, current_epoch_accuracy_detail_lst_flattern, batch_index, tot_batches, train_x_shape_0,\
                                        current_epoch_all_train_y, trainset_permutation_inds):
    global train_acc_list, train_loss_list, test_acc_list, test_loss_list, threshold_epoch, visited_indexes, min_max_data_nums_per_class, threshold_occupation, epoch_accuracy_matrix, sorted_result_angles,\
            batch_to_filter, filter_button, class_nums_counter, unforgettable_idx, start_new_dataloader_min_nums, threshold_test_accuracy_to_build_new_data, \
            n_samples, n_features, last_selected_train_x_indexes, th_win, th_train_acc, th_stable_acc, th_stable_loss

    if threshold_epoch != None and last_index != 0 and check_full() == False and filter_button == True:
        epoch_accuracy_matrix = np.array(epoch_accuracy_matrix)
        epoch_accuracy_matrix_ne = copy.deepcopy(epoch_accuracy_matrix.T[:last_index, :]) 
        epoch_accuracy_matrix = epoch_accuracy_matrix.tolist()
        epoch_accuracy_matrix_ne = np.insert(epoch_accuracy_matrix_ne, epoch_accuracy_matrix_ne.shape[1],\
                        values = current_epoch_accuracy_detail_lst_flattern[:last_index], axis = 1)

        current_upper_matrix = torch.tensor(np.array(epoch_accuracy_matrix_ne[:last_index])).half().to(device) #fp16
        current_uppper_train_y = np.array(current_epoch_all_train_y)[:last_index]

        current_compare_matrix = torch.tensor(np.array(epoch_accuracy_matrix_ne[last_index - train_x_shape_0:last_index])).half().to(device)
        current_compare_train_y = np.array(current_epoch_all_train_y)[last_index - train_x_shape_0:last_index]

        sorted_result_angles_one_batch = torch.acos(torch.cosine_similarity(current_compare_matrix.unsqueeze(1), current_upper_matrix.unsqueeze(0),dim=-1))
        sorted_result_angles_one_batch *= 180 / math.pi

        sorted_result_angles_one_batch = sorted_result_angles_one_batch.cpu().numpy()
        l1, l2 = sorted_result_angles_one_batch.shape


        sorted_result_angles_one_batch = [sorted([(sorted_result_angles_one_batch[i][other_index], trainset_permutation_inds[other_index], trainset_permutation_inds[last_index - train_x_shape_0 + i])\
                                    for other_index in range(l2)], key = lambda x: x[0]) for i in range(l1)]

        # shape: batch_size * Angle_list, each element in Angle_list: (angle, other_index, me_index), sorted                                                                                     
        sorted_result_angles.append(sorted_result_angles_one_batch)

        # 删除中间变量释放内存
        del sorted_result_angles_one_batch
        del current_upper_matrix
        del current_compare_matrix
        del epoch_accuracy_matrix_ne
        del current_uppper_train_y
        del current_compare_train_y

        print("calculate angle one batch finished")

        if (batch_to_filter != 0 and batch_index % batch_to_filter == 0) or batch_index == tot_batches:
            print(f"current_batch_index: {batch_index}, tot_bacthes: {tot_batches}")
            
            if batch_index == tot_batches:
                print("last batch!")
            [filter_train_data_by_angle_one_batch(sorted_result_angles_one_batch, current_epoch_all_train_y)\
            for sorted_result_angles_one_batch in sorted_result_angles]
            
            print(f"sorted_result_angles: {len(sorted_result_angles)}")
            print(f"current len visited_indexes: {len(visited_indexes)}. max_new_data: {min_max_data_nums_per_class[1] * 10}.")
            
            del sorted_result_angles
            sorted_result_angles = [] #申请新的列表存储积累的角度结果



def train_one_epoch(args, model, device, train_dataset, optimizer, criterion, epoch):
    global train_acc_list, train_loss_list, test_acc_list, test_loss_list, threshold_epoch, visited_indexes, min_max_data_nums_per_class, threshold_occupation, epoch_accuracy_matrix, sorted_result_angles,\
            batch_to_filter, filter_button, class_nums_counter, unforgettable_idx, start_new_dataloader_min_nums, threshold_test_accuracy_to_build_new_data, \
            n_samples, n_features, last_selected_train_x_indexes, th_win, th_train_acc, th_stable_acc, th_stable_loss
    
    train_loss = 0.
    correct = 0.
    total = 0.

    model.train()

    trainset_permutation_inds = npr.permutation(np.arange(len(train_dataset.targets)))
    
    batch_size = args.batch_size
    
    # init
    current_epoch_all_train_y = []
    current_epoch_accuracy_detail_lst = []
    sorted_result_angles = []
    last_index = 0
    for batch_index, batch_start_idx in  enumerate(range(0, len(train_dataset.targets), batch_size)):
        batch_inds = trainset_permutation_inds[batch_start_idx: batch_start_idx + batch_size]
        transformed_train_dataset = []
        for ind in batch_inds:
            transformed_train_dataset.append(train_dataset.__getitem__(ind)[0])
        inputs = torch.stack(transformed_train_dataset).to(device)
        targets = torch.LongTensor(np.array(train_dataset.targets)[batch_inds].tolist()).to(device)
        for target in targets:
            current_epoch_all_train_y.append(target.item())
    
    batch_cnt = 0
    for batch_index, batch_start_idx in  enumerate(range(0, len(train_dataset.targets), batch_size)):
        batch_inds = trainset_permutation_inds[batch_start_idx: batch_start_idx + batch_size]
        batch_cnt += 1

        transformed_train_dataset = []
        for ind in batch_inds:
            transformed_train_dataset.append(train_dataset.__getitem__(ind)[0])
        
        inputs = torch.stack(transformed_train_dataset).to(device)
        targets = torch.LongTensor(np.array(train_dataset.targets)[batch_inds].tolist()).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        _, predicted = torch.max(outputs.data, 1)
        
        # update model parameters, accuracy, loss
        loss = loss.mean()
        train_loss += loss.item()
        total += targets.size(0)
        result = predicted.eq(targets.data).cpu()
        correct += result.sum()
        current_epoch_accuracy_detail_lst.append(result.tolist())
        current_epoch_accuracy_detail_lst_flattern = np.array(list(chain(*current_epoch_accuracy_detail_lst)))
        loss.backward()
        optimizer.step()
        last_index += inputs.shape[0]
        
        if args.filter_button == True:
            calculate_angle_update_visited_index(last_index, current_epoch_accuracy_detail_lst_flattern, batch_index, len(train_dataset) // batch_size, inputs.shape[0],\
                                        current_epoch_all_train_y, trainset_permutation_inds)
        
        sys.stdout.write('\r')
        sys.stdout.write(
            '| Train | Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%' %
            (epoch, args.epochs, batch_index + 1,
             (len(train_dataset) // batch_size) + 1, loss.item(),
             100. * correct.item() / total))
        sys.stdout.flush()
    
    if isinstance(correct, float) == False:
        correct = correct.item()
    return correct / total, train_loss / batch_cnt, current_epoch_accuracy_detail_lst, trainset_permutation_inds


def test_one_epoch(args, epoch, model, device, test_dataset, criterion):
    global train_acc_list, train_loss_list, test_acc_list, test_loss_list, threshold_epoch, visited_indexes, min_max_data_nums_per_class, threshold_occupation, epoch_accuracy_matrix, sorted_result_angles,\
            batch_to_filter, filter_button, class_nums_counter, unforgettable_idx, start_new_dataloader_min_nums, threshold_test_accuracy_to_build_new_data, \
            n_samples, n_features, last_selected_train_x_indexes, th_win, th_train_acc, th_stable_acc, th_stable_loss
    test_loss = 0.
    correct = 0.
    total = 0.
    test_batch_size = 32

    model.eval()
    batch_cnt = 0
    for batch_index, batch_start_ind in enumerate(range(0, len(test_dataset.targets), test_batch_size)):
        batch_cnt += 1

        transformed_testset = []
        for ind in range(batch_start_ind, min(len(test_dataset.targets), batch_start_ind + test_batch_size)):
            transformed_testset.append(test_dataset.__getitem__(ind)[0])
        
        inputs = torch.stack(transformed_testset).to(device)
        targets = torch.LongTensor(np.array(test_dataset.targets)[batch_start_ind:batch_start_ind + test_batch_size].tolist()).to(device)

        # Forward propagation, compute loss, get predictions
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss = loss.mean()
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write(
            '| Test | Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%' %
            (epoch, args.epochs, batch_index + 1,
             (len(test_dataset) // test_batch_size) + 1, loss.item(),
             100. * correct.item() / total))
        sys.stdout.flush()
    
    if isinstance(correct, float) == False:
        correct = correct.item()
    return correct / total, test_loss / batch_cnt
    
train_acc_list, train_loss_list, test_acc_list, test_loss_list = [], [], [], []
threshold_epoch = None
visited_indexes = set()
min_max_data_nums_per_class = []
threshold_occupation = 0
epoch_accuracy_matrix = []
sorted_result_angles = []
batch_to_filter = 0
filter_button = False
class_nums_counter = {class_name:0 for class_name in range(10)}
unforgettable_idx = set()
start_new_dataloader_min_nums = 0
threshold_test_accuracy_to_build_new_data = 0
n_samples, *n_features = 0, 0
last_selected_train_x_indexes = set([i for i in range(0)])
th_win = 0
th_train_acc = 0
th_stable_acc = 0
th_stable_loss = 0

def main(args):
    global train_acc_list, train_loss_list, test_acc_list, test_loss_list, threshold_epoch, visited_indexes, min_max_data_nums_per_class, threshold_occupation, epoch_accuracy_matrix, sorted_result_angles,\
            batch_to_filter, filter_button, class_nums_counter, unforgettable_idx, start_new_dataloader_min_nums, threshold_test_accuracy_to_build_new_data, \
            n_samples, n_features, last_selected_train_x_indexes, th_win, th_train_acc, th_stable_acc, th_stable_loss
    # model, optim, criterion, scheduler
    model = ResNet18(num_classes = 10).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss().cuda()
    criterion.__init__(reduce=False)
    scheduler = MultiStepLR(
        optimizer, milestones=[60, 120, 160], gamma=0.2)
    
    # load dataset
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    train_transform = transforms.Compose([])
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4)) # data augmentation
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)
    
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = datasets.CIFAR10(
        root='C:\\Users\\GM\\Desktop\\liuzhengchang',
        train=True,
        transform=train_transform,
        download=True)
    test_dataset = datasets.CIFAR10(
        root='C:\\Users\\GM\\Desktop\\liuzhengchang',
        train=False,
        transform=test_transform,
        download=True)
    
    # init
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    npr.seed(args.seed)
    unforgettable_idx = set()
    train_indx = None
    if args.run_mode == "filter":
        train_indx = np.array(range(len(train_dataset.targets)))

        with open(args.sorting_file, 'rb') as fin:
            items = list(pickle.load(fin).items())
            ordered_indx = items[0][-1]
            ordered_values = items[1][-1]
        non_zero_forget_count_index = (np.array(ordered_values) > 0).argmax(axis = 0) # 从有序forget count列表中找到第一个非零元素的下标
        unforgettable_idx = set(np.array(ordered_indx)[:non_zero_forget_count_index]) # 从forget中load unforget examples
        print(f"forget-len(unforgettable_idx): {len(unforgettable_idx)}")
        train_dataset.data = train_dataset.data[train_indx, :, :, :]
        train_dataset.targets = np.array(train_dataset.targets)[train_indx].tolist()
    elif args.run_mode == "pure":
        train_indx = torch.load(args.filtered_index_file_path)
        train_dataset.data = train_dataset.data[train_indx, :, :, :]
        train_dataset.targets = np.array(train_dataset.targets)[train_indx].tolist()
    
    train_acc_list, train_loss_list, test_acc_list, test_loss_list = [], [], [], []
    threshold_epoch = None
    visited_indexes = set()
    min_max_data_nums_per_class = [args.min_data_nums_per_class, args.max_data_nums_per_class]
    threshold_occupation = args.threshold_occupation
    epoch_accuracy_matrix = []
    sorted_result_angles = []
    batch_to_filter = args.batch_to_filter
    filter_button = args.filter_button if args.run_mode == "filter" else False
    class_nums_counter = {class_name:0 for class_name in range(10)}
    start_new_dataloader_min_nums = args.start_new_dataloader_min_nums
    threshold_test_accuracy_to_build_new_data = args.threshold_test_accuracy_to_build_new_data
    n_samples, *n_features = train_dataset.data.shape
    last_selected_train_x_indexes = set([i for i in range(n_samples)])
    th_win = args.th_win
    th_train_acc = args.th_train_acc
    th_stable_acc = args.th_stable_acc
    th_stable_loss = args.th_stable_loss
    print(f"filter_button: {filter_button}, args.filter_button: {args.filter_button}")

    # train test loop
    for epoch in range(args.epochs):
        train_acc, train_loss, current_epoch_accuracy_detail_lst, trainset_permutation_inds = train_one_epoch(args, model, device, train_dataset, optimizer, criterion, epoch)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)

        test_acc, test_loss = test_one_epoch(args, epoch, model, device, test_dataset, criterion)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)

        if filter_button == True: 
            update_dataloader(test_acc, current_epoch_accuracy_detail_lst, train_dataset, trainset_permutation_inds, args)
        if threshold_epoch == None:
            update_threshold_epoch(epoch)
        print(f"epoch: {epoch}, acc_train: {train_acc_list[-1]}, loss_train: {train_loss_list[-1]}, acc_test: {test_acc_list[-1]}, loss_test: {test_loss_list[-1]}")

        scheduler.step()

    print(f"train_acc_list: {train_acc_list}")
    print(f"train_loss_list: {train_loss_list}")
    print(f"test_acc_list: {test_acc_list}")
    print(f"test_loss_list: {test_loss_list}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CNN')
    parser.add_argument("--epochs", type=int, default = 200, help = "epochs")
    parser.add_argument("--batch_size", type=int, default = 128, help = "batch_size for training")
    parser.add_argument("--lr", type=float, default = 0.1, help = "lr")
    parser.add_argument("--seed", type=int, default = 1, help = "seed")
    parser.add_argument("--th_win", type=int, default = 3, help = "th_win")
    parser.add_argument("--th_stable_acc", type=float, default = 1e-3, help = "th_stable_acc")
    parser.add_argument("--th_stable_loss", type=float, default = 1e-3, help = "th_stable_loss")
    parser.add_argument("--th_train_acc", type=float, default = 0.85, help = "th_train_acc")
    parser.add_argument("--filter_button", action = "store_true", help="filter_button")
    parser.add_argument("--batch_to_filter", type=int, default=10, help="batch_to_filter")
    parser.add_argument("--threshold_occupation", type=int, default=2, help="threshold_occupation")
    parser.add_argument("--start_new_dataloader_min_nums", type=int, default=100, help="start_new_dataloader_min_nums")
    parser.add_argument("--threshold_test_accuracy_to_build_new_data", type=float, default=0.6, help="threshold_test_accuracy_to_build_new_data")
    parser.add_argument("--min_data_nums_per_class", type=int, default=10, help="min_data_nums_per_class")
    parser.add_argument("--max_data_nums_per_class", type=int, default=2000, help="max_data_nums_per_class")
    parser.add_argument("--run_mode", type=str, default="filter", help="run_mode") 
    '''
    run_mode:
        "filter": 1.全部数据在转折点之前跑,forget计算原型,angle根据unforgettable example对应挑选,用挑选后的数据跑转折点之后, 
                2.并保存从全部数据中筛选出来的下标至filtered_index_file_path，当前shuffle序列保存至filtered_index_trainset_permutation_inds_file_path
        "pure": 用filter保存下来的下标,在训练之前将原始数据变成指定的数据,用这些数据跑完所有epochs
        全部数据跑完所有epoch: 将filter_button设为False即可
    '''
    parser.add_argument("--filtered_index_file_path", type=str, default="C:\\Users\\GM\\Desktop\\liuzhengchang\\CODE\\filtered_index_file_path.pt", help="filtered_index_file_path")
    parser.add_argument("--sorting_file", type=str, default="C:\\Users\\GM\\Desktop\\liuzhengchang\\CODE\\cifar10_results\\cifar10_sorted.pkl", help="sorting_file")
    args = parser.parse_args()
    main(args)

# filter:
# python cifar10_code.py --epochs 200 --batch_size 128 --lr 0.1 --seed 1 --th_win 3 --th_stable_acc 0.001 --th_stable_loss 0.001 --th_train_acc 0.85 --filter_button --batch_to_filter 10 --threshold_occupation 2 --start_new_dataloader_min_nums 100 --threshold_test_accuracy_to_build_new_data 0.6  --min_data_nums_per_class 10 --max_data_nums_per_class 2000 --run_mode filter --filtered_index_file_path C:\\Users\\GM\\Desktop\\liuzhengchang\\CODE\\filtered_index_file_path.pt --sorting_file C:\\Users\\GM\\Desktop\\liuzhengchang\\CODE\\cifar10_results\\cifar10_sorted.pkl
# pure:
# python cifar10_code.py --epochs 200 --batch_size 128 --lr 0.1 --seed 1 --th_win 3 --th_stable_acc 0.001 --th_stable_loss 0.001 --th_train_acc 0.85 --batch_to_filter 10 --threshold_occupation 2 --start_new_dataloader_min_nums 100 --threshold_test_accuracy_to_build_new_data 0.6  --min_data_nums_per_class 10 --max_data_nums_per_class 2000 --run_mode pure --filtered_index_file_path C:\\Users\\GM\\Desktop\\liuzhengchang\\CODE\\filtered_index_file_path.pt --sorting_file C:\\Users\\GM\\Desktop\\liuzhengchang\\CODE\\cifar10_results\\cifar10_sorted.pkl
# whole data:
# python cifar10_code.py --epochs 200 --batch_size 128 --lr 0.1 --seed 1 --th_win 3 --th_stable_acc 0.001 --th_stable_loss 0.001 --th_train_acc 0.85 --batch_to_filter 10 --threshold_occupation 2 --start_new_dataloader_min_nums 100 --threshold_test_accuracy_to_build_new_data 0.6  --min_data_nums_per_class 10 --max_data_nums_per_class 2000 --run_mode filter --filtered_index_file_path C:\\Users\\GM\\Desktop\\liuzhengchang\\CODE\\filtered_index_file_path.pt --sorting_file C:\\Users\\GM\\Desktop\\liuzhengchang\\CODE\\cifar10_results\\cifar10_sorted.pkl