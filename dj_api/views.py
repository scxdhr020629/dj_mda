from django.conf import settings
from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, JsonResponse
import pandas as pd
import numpy as np
import os
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx

import csv
import logging
import sys
from io import BytesIO

import torch.nn as nn
from jinja2.lexer import TOKEN_DOT
from torch.utils.data import DataLoader, WeightedRandomSampler
from .model.cnn_gcnmulti import GCNNetmuti
# from  model.cnn_gcn import GCNNet
from .utils.utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc

from django.shortcuts import render
from django.http import JsonResponse
from django.core.cache import cache


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                                           'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
                                           'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn',
                                           'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def process_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        else:
            return None
    except:
        return None



def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index


def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN)
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        # print(line[0][0])
        # print(i,"========", ch)
        X[i] = smi_ch_ind[ch]
    return X

    # 这里的seq_dict 在后面


def seq_cat(prot, max_seq_len):
    x = np.zeros(max_seq_len)
    seq_voc = "ACGU"
    seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                         "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                         "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                         "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                         "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                         "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                         "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                         "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64, ">": 65, "<": 66}
def hello(request):
    # return HttpResponse("Hello django. I am comming.")
    return JsonResponse({'hello': 'world'})


def get_user(request):
    if request.method == 'GET':
        usid = request.GET.get('usid', '')
        if usid == '':
            return JsonResponse({'code': 100101, 'msg': '用户id不能为空'})
        if usid == '1':
            return JsonResponse({'code': 100200, 'msg': '查询成功', 'data': {'usid': 1, 'name': 'james', 'age': 36}})
        else:
            return JsonResponse({'code': 100102, 'msg': '未查询到用户数据'})

    else:
        return JsonResponse({'code': 100103, 'msg': '请求方法错误'})


def add_user(request):
    if request.method == 'POST':
        usid = request.POST.get('usid', '')
        name = request.POST.get('name', '')
        print(usid, name)
        if usid == '' or name == '':
            return JsonResponse({'code': 100101, 'msg': '用户id或密码不能为空'})
        if usid == '1' and name == 'james':
            return JsonResponse({'code': 100200, 'msg': '添加成功', 'data': {'usid': 1, 'name': 'james', 'age': 36}})
        else:
            return JsonResponse({'code': 100102, 'msg': '添加失败'})

    else:
        return JsonResponse({'code': 100103, 'msg': '请求方法错误'})






def get_drug_rna_relation(request):
    if request.method == 'POST':

        body = json.loads(request.body.decode('utf-8'))
        drug_sequence = body.get('drug_sequence')
        rna_sequence = body.get('rna_sequence')
        print(rna_sequence)
        cache_key = f"r_sequence_{drug_sequence}_{rna_sequence}"  # 使用 rna_sequence 作为缓存的 key
        cached_result = cache.get(cache_key)
        if cached_result:
            # 如果缓存中有结果，则直接返回缓存的结果
            print("Returning cached result")
            return JsonResponse({'code': 0, 'msg': '查询成功，内容如下', "data": cached_result})

        raw_data = rna_sequence
        # 一些processdata的函数代码


        # -------------------------------------------------------------------------------------------------------------------------------------------------------------
        # 第一部分 对传入的数据进行处理
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------


        CHARISOSMILEN = 66

        # ----------------------------------------------------------------------------------------------------------------------------------------------------
        # 下面这部分代码是 drug 和 mirna 的读取

        # 读取整个 Excel 文件
        import pandas as pd
        # 怎么识别不上


        str_i = '1'

        # 这里先处理一下 用process_smiles

        my_process_smile = []

        result = process_smiles(drug_sequence)
        my_process_smile.append(result)
        compound_iso_smiles = my_process_smile
        rna_list = [raw_data] * 1  # 创建一个和 smiles 列一样长的列表

        # 将 affinity 列全部设置为 0
        affinity = [0] * 1

        # 创建 DataFrame，合并所有数据
        final_df = pd.DataFrame({
            'compound_iso_smiles': compound_iso_smiles,
            'target_sequence': rna_list,
            'affinity': affinity
        })

        # 保存为 CSV 文件
        output_file = 'dj_api/data/processed/last/_mytest' + str_i + '.csv'
        final_df.to_csv(output_file, index=False)

        print(f"CSV 文件已保存至: {output_file}")
        seq_voc = "ACGU"
        seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
        seq_dict_len = len(seq_dict)

        # 药物图处理过程
        opts = ['mytest']
        for i in range(1, 2):
            compound_iso_smiles = []
            for opt in opts:
                df = pd.read_csv('dj_api/data/processed/last/' + '_' + opt + str_i + '.csv')
                compound_iso_smiles += list(df['compound_iso_smiles'])
            compound_iso_smiles = set(compound_iso_smiles)
            smile_graph = {}
            for smile in compound_iso_smiles:
                g = smile_to_graph(smile)
                smile_graph[smile] = g

            # # convert to PyTorch data format
            df = pd.read_csv('dj_api/data/processed/last/' + '_mytest' + str_i + '.csv')
            test_drugs, test_prots, test_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
                df['affinity'])
            XT = [seq_cat(t, 24) for t in test_prots]
            test_sdrugs = [label_smiles(t, CHARISOSMISET, 100) for t in test_drugs]
            test_drugs, test_prots, test_Y, test_seqdrugs = np.asarray(test_drugs), np.asarray(XT), np.asarray(
                test_Y), np.asarray(test_sdrugs)
            test_data = TestbedDataset(root='dj_api/data', dataset='last/' + '_mytest' + str_i, xd=test_drugs, xt=test_prots,
                                       y=test_Y,
                                       z=test_seqdrugs,
                                       smile_graph=smile_graph)

        # ---------------------------------------------------------------------------------------------------------------------------------------------------
        # 第二部分 进行预测
        # ----------------------------------------------------------------------------------------------------------------------------------------------------

        # 删去 rog auc

        def predicting(model, device, loader):
            model.eval()
            total_probs = []
            sample_indices = []
            total_labels = []

            logging.info('Making predictions for {} samples...'.format(len(loader.dataset)))
            with torch.no_grad():
                for batch_idx, data in enumerate(loader):
                    data = data.to(device)
                    output = model(data)
                    probs = output.cpu().numpy()
                    indices = np.arange(len(probs)) + batch_idx * loader.batch_size

                    total_probs.extend(probs)
                    sample_indices.extend(indices)
                    total_labels.extend(data.y.view(-1, 1).cpu().numpy())

            total_probs = np.array(total_probs).flatten()
            sample_indices = np.array(sample_indices).flatten()
            total_labels = np.array(total_labels).flatten()

            return total_probs, sample_indices


        def save_predictions(probs, indices, file_name='all_predict_02.csv'):
            # Convert probabilities and indices to a DataFrame
            predictions_df = pd.DataFrame({
                'Index': indices,  # 保存样本索引
                'Probability': probs  # 保存预测概率
            })

            # Save to CSV without sorting
            predictions_df.to_csv(file_name, index=False)
            logging.info(f'Predictions saved to {file_name}')

        # 检测一下这段代码写的是否正确
        def save_top_30_predictions(probs, indices, file_name='top_30_predictions_02.csv'):
            # Sort by probability (in descending order)
            sorted_indices = np.argsort(probs)[::-1]  # sort in descending order
            sorted_probs = probs[sorted_indices]
            # sorted_labels = labels[sorted_indices]
            sorted_indices = indices[sorted_indices]

            # Create a DataFrame to save the top 30 predictions
            top_30_df = pd.DataFrame({
                'Index': sorted_indices[:30],
                'Probability': sorted_probs[:30],
                # 'True_Label': sorted_labels[:30]
            })

            # Save to CSV
            top_30_df.to_csv(file_name, index=False)
            logging.info(f'Top 30 predictions saved to {file_name}')

        # 保存 rna信息
        import pandas as pd
        modeling = GCNNetmuti

        model_st = modeling.__name__

        cuda_name = "cuda:0"
        if len(sys.argv) > 3:
            cuda_name = "cuda:" + str(int(sys.argv[1]))
        print('cuda_name:', cuda_name)

        # TRAIN_BATCH_SIZE = 64
        TEST_BATCH_SIZE = 64
        LR = 0.0005
        # LOG_INTERVAL = 160
        NUM_EPOCHS = 100

        print('Learning rate: ', LR)
        print('Epochs: ', NUM_EPOCHS)

        # Main program: iterate over different datasets

        print('\nrunning on ', model_st + '_')

        log_filename = f'training_1.log'

        logging.basicConfig(filename=log_filename, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.INFO)

        processed_data_file_test = 'dj_api/data/processed/last/' + '_mytest' + str_i + '.pt'
        if ((not os.path.isfile(processed_data_file_test))):
            print('please run process_data_old.py to prepare data in pytorch format!')
        else:
            test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=False)

            # training the model
            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
            model = modeling().to(device)
            model_path = "dj_api/model_GCNNetmuti_.model"
            model.load_state_dict(torch.load(model_path))

            loss_fn = nn.BCELoss()  # for classification


            probs, indices = predicting(model, device, test_loader)
            print(f'probs: {probs}')
            result_data = {
                "drug_sequence": drug_sequence,  # 这里用你的drug_sequence变量
                "rna_sequence": rna_sequence,
                "probability": float(probs[0] ) # 确保转换为Python原生float，便于JSON序列化
            }
            cache.set(cache_key, result_data, 60 * 60 * 24 * 7)  # 缓存7天
            return JsonResponse({'code':0,'msg':'查询成功，内容如下',"data": result_data})



def get_all_drug_rna_relation(request):
    if request.method == "POST":
        body = json.loads(request.body)
        dataList = body.get("data")
        all_results = []

        for i in range(len(dataList)):
            drug_sequence = dataList[i]["drug_sequence"]
            rna_sequence = dataList[i]["rna_sequence"]

            cache_key = f"r_sequence_{drug_sequence}_{rna_sequence}"  # 使用 rna_sequence 作为缓存的 key
            cached_result = cache.get(cache_key)

            if cached_result:
                print("Returning cached result")
                all_results.append(cached_result)
                continue

            raw_data = rna_sequence

            CHARISOSMILEN = 66

            # 读取整个 Excel 文件


            str_i = '1'

            # 处理 drug_sequence
            my_process_smile = []

            result = process_smiles(drug_sequence)
            my_process_smile.append(result)
            compound_iso_smiles = my_process_smile
            rna_list = [raw_data] * 1  # 创建一个和 smiles 列一样长的列表
            affinity = [0] * 1

            final_df = pd.DataFrame({
                'compound_iso_smiles': compound_iso_smiles,
                'target_sequence': rna_list,
                'affinity': affinity
            })

            output_file = f'dj_api/data/processed/last/_mytest{str_i}.csv'
            final_df.to_csv(output_file, index=False)

            seq_voc = "ACGU"
            seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
            seq_dict_len = len(seq_dict)

            opts = ['mytest']
            for j in range(1, 2):
                compound_iso_smiles = []
                for opt in opts:
                    df = pd.read_csv(f'dj_api/data/processed/last/_{opt}{str_i}.csv')
                    compound_iso_smiles += list(df['compound_iso_smiles'])
                compound_iso_smiles = set(compound_iso_smiles)
                smile_graph = {}
                for smile in compound_iso_smiles:
                    g = smile_to_graph(smile)
                    smile_graph[smile] = g

                df = pd.read_csv(f'dj_api/data/processed/last/_mytest{str_i}.csv')
                test_drugs, test_prots, test_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
                    df['affinity'])
                XT = [seq_cat(t, 24) for t in test_prots]
                test_sdrugs = [label_smiles(t, CHARISOSMISET, 100) for t in test_drugs]
                test_drugs, test_prots, test_Y, test_seqdrugs = np.asarray(test_drugs), np.asarray(XT), np.asarray(
                    test_Y), np.asarray(test_sdrugs)
                test_data = TestbedDataset(root='dj_api/data', dataset=f'last/_mytest{str_i}', xd=test_drugs,
                                           xt=test_prots, y=test_Y, z=test_seqdrugs, smile_graph=smile_graph)

            def predicting(model, device, loader):
                model.eval()
                total_probs = []
                sample_indices = []

                logging.info(f'Making predictions for {len(loader.dataset)} samples...')
                with torch.no_grad():
                    for batch_idx, data in enumerate(loader):
                        data = data.to(device)
                        output = model(data)
                        probs = output.cpu().numpy()
                        indices = np.arange(len(probs)) + batch_idx * loader.batch_size

                        total_probs.extend(probs)
                        sample_indices.extend(indices)

                total_probs = np.array(total_probs).flatten()
                sample_indices = np.array(sample_indices).flatten()

                return total_probs, sample_indices








            modeling = GCNNetmuti

            model_st = modeling.__name__

            cuda_name = "cuda:0"
            if len(sys.argv) > 3:
                cuda_name = "cuda:" + str(int(sys.argv[1]))
            print('cuda_name:', cuda_name)

            TEST_BATCH_SIZE = 64
            LR = 0.0005
            NUM_EPOCHS = 100

            print('Learning rate: ', LR)
            print('Epochs: ', NUM_EPOCHS)

            print('\nrunning on ', model_st + '_')

            log_filename = 'training_1.log'

            logging.basicConfig(filename=log_filename, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s',
                                level=logging.INFO)

            processed_data_file_test = f'dj_api/data/processed/last/_mytest{str_i}.pt'
            if not os.path.isfile(processed_data_file_test):
                print('please run process_data_old.py to prepare data in pytorch format!')
            else:
                test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=False)

                device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
                model = modeling().to(device)
                model_path = "dj_api/model_GCNNetmuti_.model"
                model.load_state_dict(torch.load(model_path))

                loss_fn = nn.BCELoss()

                probs, indices = predicting(model, device, test_loader)

                result_data = {
                    "drug_sequence": drug_sequence,  # 这里用你的drug_sequence变量
                    "rna_sequence": rna_sequence,
                    "probability": float(probs[0]) # 确保转换为Python原生float，便于JSON序列化
                }
                cache.set(cache_key, result_data, timeout=3600)  # 缓存7天
                all_results.append(result_data)

        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df = pd.DataFrame(all_results, columns=["drug_sequence", "rna_sequence","probability"])
            df.to_excel(writer, index=False, sheet_name='Sheet1')

        output.seek(0)
        file_name = "predicted_all_drugs_rnas.xlsx"
        file_path = os.path.join(settings.MEDIA_ROOT, file_name)

        with open(file_path, "wb") as file:
            file.write(output.read())

        download_link = request.build_absolute_uri(settings.MEDIA_URL + file_name)

        return JsonResponse({'code': 0, 'msg': 'xlsx文件发送成功', 'data': download_link})





def get_drugs(request):
    if request.method == 'POST':

        body = json.loads(request.body.decode('utf-8'))
        # drug_sequence = body.get('drug_sequence')
        rna_sequence = body.get('rna_sequence')
        print(rna_sequence)
        cache_key = f"rna_sequence_{rna_sequence}"  # 使用 rna_sequence 作为缓存的 key
        cached_result = cache.get(cache_key)

        if cached_result:
            # 如果缓存中有结果，则直接返回缓存的结果
            print("Returning cached result")
            return JsonResponse({'code': 0, 'msg': '查询成功，内容如下', "data": cached_result})

        raw_data = rna_sequence
        # 一些processdata的函数代码


        # -------------------------------------------------------------------------------------------------------------------------------------------------------------
        # 第一部分 对传入的数据进行处理
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------


        CHARISOSMILEN = 66

        # ----------------------------------------------------------------------------------------------------------------------------------------------------
        # 下面这部分代码是 drug 和 mirna 的读取

        # 读取整个 Excel 文件
        import pandas as pd
        # 怎么识别不上
        drugs = pd.read_excel('dj_api/data/drug_id_smiles.xlsx')
        # rna = pd.read_excel('dj_api/data/miRNA_sequences.xlsx')

        str_i = '1'

        # 这里先处理一下 用process_smiles
        # smiles_list = drugs['smiles'].tolist()
        my_process_smile = []
        # for smiles in smiles_list:
        #     result = process_smiles(smiles)
        #     my_process_smile.append(result)
        ligands = drugs['smiles']
        for d in ligands.keys():
            lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
            print(lg)
            my_process_smile.append(lg)
        compound_iso_smiles = my_process_smile
        # print(compound_iso_smiles)
        rna_list = [raw_data] * len(drugs)  # 创建一个和 smiles 列一样长的列表

        # 将 affinity 列全部设置为 0
        affinity = [0] * len(drugs)

        # 创建 DataFrame，合并所有数据
        final_df = pd.DataFrame({
            'compound_iso_smiles': compound_iso_smiles,
            'target_sequence': rna_list,
            'affinity': affinity
        })

        # 保存为 CSV 文件
        output_file = 'dj_api/data/processed/last/_mytest' + str_i + '.csv'
        final_df.to_csv(output_file, index=False)

        print(f"CSV 文件已保存至: {output_file}")
        seq_voc = "ACGU"
        seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
        seq_dict_len = len(seq_dict)

        # 药物图处理过程
        opts = ['mytest']
        for i in range(1, 2):
            compound_iso_smiles = []
            for opt in opts:
                df = pd.read_csv('dj_api/data/processed/last/' + '_' + opt + str_i + '.csv')
                compound_iso_smiles += list(df['compound_iso_smiles'])
            compound_iso_smiles = set(compound_iso_smiles)
            smile_graph = {}
            for smile in compound_iso_smiles:
                g = smile_to_graph(smile)
                smile_graph[smile] = g

            # # convert to PyTorch data format
            df = pd.read_csv('dj_api/data/processed/last/' + '_mytest' + str_i + '.csv')
            test_drugs, test_prots, test_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
                df['affinity'])
            XT = [seq_cat(t, 24) for t in test_prots]
            test_sdrugs = [label_smiles(t, CHARISOSMISET, 100) for t in test_drugs]
            test_drugs, test_prots, test_Y, test_seqdrugs = np.asarray(test_drugs), np.asarray(XT), np.asarray(
                test_Y), np.asarray(test_sdrugs)
            test_data = TestbedDataset(root='dj_api/data', dataset='last/' + '_mytest' + str_i, xd=test_drugs, xt=test_prots,
                                       y=test_Y,
                                       z=test_seqdrugs,
                                       smile_graph=smile_graph)

        # ---------------------------------------------------------------------------------------------------------------------------------------------------
        # 第二部分 进行预测
        # ----------------------------------------------------------------------------------------------------------------------------------------------------

        # 删去 rog auc

        def predicting(model, device, loader):
            model.eval()
            total_probs = []
            sample_indices = []
            total_labels = []

            logging.info('Making predictions for {} samples...'.format(len(loader.dataset)))
            with torch.no_grad():
                for batch_idx, data in enumerate(loader):
                    data = data.to(device)
                    output = model(data)
                    probs = output.cpu().numpy()
                    indices = np.arange(len(probs)) + batch_idx * loader.batch_size

                    total_probs.extend(probs)
                    sample_indices.extend(indices)
                    total_labels.extend(data.y.view(-1, 1).cpu().numpy())

            total_probs = np.array(total_probs).flatten()
            sample_indices = np.array(sample_indices).flatten()
            total_labels = np.array(total_labels).flatten()

            return total_probs, sample_indices


        def save_predictions(probs, indices, file_name='all_predict_02.csv'):
            # Convert probabilities and indices to a DataFrame
            predictions_df = pd.DataFrame({
                'Index': indices,  # 保存样本索引
                'Probability': probs  # 保存预测概率
            })

            # Save to CSV without sorting
            predictions_df.to_csv(file_name, index=False)
            logging.info(f'Predictions saved to {file_name}')

        # 检测一下这段代码写的是否正确
        def save_top_30_predictions(probs, indices, file_name='top_30_predictions_02.csv'):
            # Sort by probability (in descending order)
            sorted_indices = np.argsort(probs)[::-1]  # sort in descending order
            sorted_probs = probs[sorted_indices]
            # sorted_labels = labels[sorted_indices]
            sorted_indices = indices[sorted_indices]

            # Create a DataFrame to save the top 30 predictions
            top_30_df = pd.DataFrame({
                'Index': sorted_indices[:30],
                'Probability': sorted_probs[:30],
                # 'True_Label': sorted_labels[:30]
            })

            # Save to CSV
            top_30_df.to_csv(file_name, index=False)
            logging.info(f'Top 30 predictions saved to {file_name}')

        # 保存 rna信息
        import pandas as pd

        def map_top_30_to_drugs(top_30_file, output_file='top_30_drugs_predictions.csv'):
            # 1. 读取drugs文件
            # miRNA_df = rna  # 加载miRNA序列文件
            drugs_df = drugs
            # 2. 读取top 30预测文件
            top_30_df = pd.read_csv(top_30_file)  # 加载top 30预测文件

            # 3. 根据Index，将top 30的预测与miRNA_sequences的数据合并
            merged_df = pd.merge(top_30_df, drugs_df, left_on='Index', right_index=True, how='left')

            # 4. 选择需要保存的列
            result_df = merged_df[['DrugBank_ID', 'smiles', 'Probability']]

            # 5. 保存为CSV文件
            result_df.to_csv(output_file, index=False)
            print(f'Top 30 miRNA predictions saved to {output_file}')
            # return result_df








        modeling = GCNNetmuti

        model_st = modeling.__name__

        cuda_name = "cuda:0"
        if len(sys.argv) > 3:
            cuda_name = "cuda:" + str(int(sys.argv[1]))
        print('cuda_name:', cuda_name)

        # TRAIN_BATCH_SIZE = 64
        TEST_BATCH_SIZE = 64
        LR = 0.0005
        # LOG_INTERVAL = 160
        NUM_EPOCHS = 100

        print('Learning rate: ', LR)
        print('Epochs: ', NUM_EPOCHS)

        # Main program: iterate over different datasets

        print('\nrunning on ', model_st + '_')

        log_filename = f'training_1.log'

        logging.basicConfig(filename=log_filename, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.INFO)

        processed_data_file_test = 'dj_api/data/processed/last/' + '_mytest' + str_i + '.pt'
        if ((not os.path.isfile(processed_data_file_test))):
            print('please run process_data_old.py to prepare data in pytorch format!')
        else:
            test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=True)

            # training the model
            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
            model = modeling().to(device)
            model_path = "dj_api/model_GCNNetmuti_.model"
            model.load_state_dict(torch.load(model_path))

            loss_fn = nn.BCELoss()  # for classification


            probs, indices = predicting(model, device, test_loader)

            save_predictions(probs, indices, 'all_predict_0' + str_i + '.csv')
            save_top_30_predictions(probs, indices, 'top_30_predictions_0' + str_i + '.csv')

            # 现在是把前面的数据都给获取到了 但是我们要把数据和mirna对应上
            top_30_file = 'top_30_predictions_0' + str_i + '.csv'
            map_top_30_to_drugs(top_30_file, 'top_30_drug_predictions_0' + str_i + '.csv')


            csv_file_path = 'top_30_drug_predictions_0' + str_i + '.csv'
            data_list = []
            # result_df = merged_df[['DrugBank_ID', 'smiles', 'Probability']]
            # 读取 CSV 文件
            with open(csv_file_path, mode="r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                for index, row in enumerate(reader):
                    if index >= 5:  # 只获取前五条数据
                        break
                    data_list.append({
                        "DrugBank_ID": row.get("DrugBank_ID"),
                        "smiles": row.get("smiles"),
                        "Probability": row.get("Probability")
                    })


            cache.set(cache_key, data_list, timeout=3600)  # 缓存保存1小时
            # 返回 JSON 响应
            return JsonResponse({'code':0,'msg':'查询成功，内容如下',"data": data_list})


def get_all_drugs(request):
    if request.method == "POST":
        body = json.loads(request.body)
        dataList = body.get("data")
        all_results = []

        for i in range(len(dataList)):
            rna_sequence = dataList[i]["sequence"]
            print(rna_sequence)
            cache_key = f"rna_sequence_{rna_sequence}"  # 使用 drug_sequence 作为缓存的 key
            cached_result = cache.get(cache_key)

            if cached_result:
                print("Returning cached result")
                all_results.extend([(rna_sequence, drug["DrugBank_ID"],drug["smiles"],drug["Probability"]) for drug in cached_result])
                continue

            raw_data = rna_sequence

            CHARISOSMILEN = 66

            # 读取整个 Excel 文件
            drugs = pd.read_excel('dj_api/data/drug_id_smiles.xlsx')
            # rna = pd.read_excel('dj_api/data/miRNA_sequences.xlsx')

            str_i = '1'

            # 处理 drug_sequence
            # 这里先处理一下 用process_smiles
            my_process_smile = []
            ligands = drugs['smiles']
            for d in ligands.keys():
                lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
                print(lg)
                my_process_smile.append(lg)
            compound_iso_smiles = my_process_smile
            rna_list = [raw_data] * len(drugs)  # 创建一个和 smiles 列一样长的列表
            affinity = [0] * len(drugs)

            final_df = pd.DataFrame({
                'compound_iso_smiles': compound_iso_smiles,
                'target_sequence': rna_list,
                'affinity': affinity
            })

            output_file = f'dj_api/data/processed/last/_mytest{str_i}.csv'
            final_df.to_csv(output_file, index=False)

            seq_voc = "ACGU"
            seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
            seq_dict_len = len(seq_dict)

            opts = ['mytest']
            for j in range(1, 2):
                compound_iso_smiles = []
                for opt in opts:
                    df = pd.read_csv(f'dj_api/data/processed/last/_{opt}{str_i}.csv')
                    compound_iso_smiles += list(df['compound_iso_smiles'])
                compound_iso_smiles = set(compound_iso_smiles)
                smile_graph = {}
                for smile in compound_iso_smiles:
                    g = smile_to_graph(smile)
                    smile_graph[smile] = g

                df = pd.read_csv(f'dj_api/data/processed/last/_mytest{str_i}.csv')
                test_drugs, test_prots, test_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
                    df['affinity'])
                XT = [seq_cat(t, 24) for t in test_prots]
                test_sdrugs = [label_smiles(t, CHARISOSMISET, 100) for t in test_drugs]
                test_drugs, test_prots, test_Y, test_seqdrugs = np.asarray(test_drugs), np.asarray(XT), np.asarray(
                    test_Y), np.asarray(test_sdrugs)
                test_data = TestbedDataset(root='dj_api/data', dataset=f'last/_mytest{str_i}', xd=test_drugs,
                                           xt=test_prots, y=test_Y, z=test_seqdrugs, smile_graph=smile_graph)

            def predicting(model, device, loader):
                model.eval()
                total_probs = []
                sample_indices = []

                logging.info(f'Making predictions for {len(loader.dataset)} samples...')
                with torch.no_grad():
                    for batch_idx, data in enumerate(loader):
                        data = data.to(device)
                        output = model(data)
                        probs = output.cpu().numpy()
                        indices = np.arange(len(probs)) + batch_idx * loader.batch_size

                        total_probs.extend(probs)
                        sample_indices.extend(indices)

                total_probs = np.array(total_probs).flatten()
                sample_indices = np.array(sample_indices).flatten()

                return total_probs, sample_indices

            def save_predictions(probs, indices, file_name='all_predict_02.csv'):
                predictions_df = pd.DataFrame({
                    'Index': indices,
                    'Probability': probs
                })

                predictions_df.to_csv(file_name, index=False)
                logging.info(f'Predictions saved to {file_name}')

            def save_top_30_predictions(probs, indices, file_name='top_30_predictions_02.csv'):
                sorted_indices = np.argsort(probs)[::-1]
                sorted_probs = probs[sorted_indices]
                sorted_indices = indices[sorted_indices]

                top_30_df = pd.DataFrame({
                    'Index': sorted_indices[:30],
                    'Probability': sorted_probs[:30]
                })

                top_30_df.to_csv(file_name, index=False)
                logging.info(f'Top 30 predictions saved to {file_name}')

            def map_top_30_to_drugs(top_30_file, output_file='top_30_drugs_predictions.csv'):
                # 1. 读取drugs文件
                # miRNA_df = rna  # 加载miRNA序列文件
                drugs_df = drugs
                # 2. 读取top 30预测文件
                top_30_df = pd.read_csv(top_30_file)  # 加载top 30预测文件

                # 3. 根据Index，将top 30的预测与miRNA_sequences的数据合并
                merged_df = pd.merge(top_30_df, drugs_df, left_on='Index', right_index=True, how='left')

                # 4. 选择需要保存的列
                result_df = merged_df[['DrugBank_ID', 'smiles', 'Probability']]

                # 5. 保存为CSV文件
                result_df.to_csv(output_file, index=False)
                print(f'Top 30 miRNA predictions saved to {output_file}')
                # return result_df

            modeling = GCNNetmuti

            model_st = modeling.__name__

            cuda_name = "cuda:0"
            if len(sys.argv) > 3:
                cuda_name = "cuda:" + str(int(sys.argv[1]))
            print('cuda_name:', cuda_name)

            TEST_BATCH_SIZE = 64
            LR = 0.0005
            NUM_EPOCHS = 100

            print('Learning rate: ', LR)
            print('Epochs: ', NUM_EPOCHS)

            print('\nrunning on ', model_st + '_')

            log_filename = 'training_1.log'

            logging.basicConfig(filename=log_filename, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s',
                                level=logging.INFO)

            processed_data_file_test = f'dj_api/data/processed/last/_mytest{str_i}.pt'
            if not os.path.isfile(processed_data_file_test):
                print('please run process_data_old.py to prepare data in pytorch format!')
            else:
                test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=True)

                device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
                model = modeling().to(device)
                model_path = "dj_api/model_GCNNetmuti_.model"
                model.load_state_dict(torch.load(model_path))

                loss_fn = nn.BCELoss()

                probs, indices = predicting(model, device, test_loader)

                save_predictions(probs, indices, f'all_predict_0{str_i}.csv')
                save_top_30_predictions(probs, indices, f'top_30_predictions_0{str_i}.csv')

                top_30_file = f'top_30_predictions_0{str_i}.csv'
                map_top_30_to_drugs(top_30_file, 'top_30_drug_predictions_0' + str_i + '.csv')

                csv_file_path = 'top_30_drug_predictions_0' + str_i + '.csv'
                data_list = []

                with open(csv_file_path, mode="r", encoding="utf-8") as file:
                    reader = csv.DictReader(file)
                    for index, row in enumerate(reader):
                        if index >= 5:  # 只获取前五条数据
                            break
                        data_list.append({
                            "DrugBank_ID": row.get("DrugBank_ID"),
                            "smiles": row.get("smiles"),
                            "Probability": row.get("Probability")
                        })

                cache.set(cache_key, data_list, timeout=3600)
                all_results.extend([(rna_sequence, drug["DrugBank_ID"], drug["smiles"], drug["Probability"]) for drug in
                                    data_list])

        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df = pd.DataFrame(all_results, columns=["RNA Sequence", "DrugBank_ID", "smiles","Probability"])
            df.to_excel(writer, index=False, sheet_name='Sheet1')

        output.seek(0)
        file_name = "predicted_drugs.xlsx"
        file_path = os.path.join(settings.MEDIA_ROOT, file_name)

        with open(file_path, "wb") as file:
            file.write(output.read())

        download_link = request.build_absolute_uri(settings.MEDIA_URL + file_name)

        return JsonResponse({'code': 0, 'msg': 'xlsx文件发送成功', 'data': download_link})










# test mda
def get_rnas(request):
    if request.method == 'POST':

        body = json.loads(request.body.decode('utf-8'))
        drug_sequence = body.get('drug_sequence')
        print(drug_sequence)
        cache_key = f"drug_sequence_{drug_sequence}"  # 使用 drug_sequence 作为缓存的 key
        cached_result = cache.get(cache_key)

        if cached_result:
            # 如果缓存中有结果，则直接返回缓存的结果
            print("Returning cached result")
            return JsonResponse({'code': 0, 'msg': '查询成功，内容如下', "data": cached_result})

        raw_data = drug_sequence
        # 一些processdata的函数代码


        # -------------------------------------------------------------------------------------------------------------------------------------------------------------
        # 第一部分 对传入的数据进行处理
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------


        CHARISOSMILEN = 66

        # ----------------------------------------------------------------------------------------------------------------------------------------------------
        # 下面这部分代码是 drug 和 mirna 的读取

        # 读取整个 Excel 文件
        import pandas as pd
        # 怎么识别不上
        drugs = pd.read_excel('dj_api/data/drug_id_smiles.xlsx')
        rna = pd.read_excel('dj_api/data/miRNA_sequences.xlsx')

        str_i = '1'

        # 这里先处理一下 用process_smiles
        my_process_smile = process_smiles(raw_data)

        compound_iso_smiles = [my_process_smile] * len(rna)  # 创建一个和 Sequence 列一样长的列表

        # 将 affinity 列全部设置为 0
        affinity = [0] * len(rna)

        # 创建 DataFrame，合并所有数据
        final_df = pd.DataFrame({
            'compound_iso_smiles': compound_iso_smiles,
            'target_sequence': rna['Sequence'],
            'affinity': affinity
        })

        # 保存为 CSV 文件
        output_file = 'dj_api/data/processed/last/_mytest' + str_i + '.csv'
        final_df.to_csv(output_file, index=False)

        print(f"CSV 文件已保存至: {output_file}")
        seq_voc = "ACGU"
        seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
        seq_dict_len = len(seq_dict)

        # 药物图处理过程
        opts = ['mytest']
        for i in range(1, 2):
            compound_iso_smiles = []
            for opt in opts:
                df = pd.read_csv('dj_api/data/processed/last/' + '_' + opt + str_i + '.csv')
                compound_iso_smiles += list(df['compound_iso_smiles'])
            compound_iso_smiles = set(compound_iso_smiles)
            smile_graph = {}
            for smile in compound_iso_smiles:
                g = smile_to_graph(smile)
                smile_graph[smile] = g

            # # convert to PyTorch data format
            df = pd.read_csv('dj_api/data/processed/last/' + '_mytest' + str_i + '.csv')
            test_drugs, test_prots, test_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
                df['affinity'])
            XT = [seq_cat(t, 24) for t in test_prots]
            test_sdrugs = [label_smiles(t, CHARISOSMISET, 100) for t in test_drugs]
            test_drugs, test_prots, test_Y, test_seqdrugs = np.asarray(test_drugs), np.asarray(XT), np.asarray(
                test_Y), np.asarray(test_sdrugs)
            test_data = TestbedDataset(root='dj_api/data', dataset='last/' + '_mytest' + str_i, xd=test_drugs, xt=test_prots,
                                       y=test_Y,
                                       z=test_seqdrugs,
                                       smile_graph=smile_graph)

        # ---------------------------------------------------------------------------------------------------------------------------------------------------
        # 第二部分 进行预测
        # ----------------------------------------------------------------------------------------------------------------------------------------------------

        # 删去 rog auc

        def predicting(model, device, loader):
            model.eval()
            total_probs = []
            sample_indices = []
            total_labels = []

            logging.info('Making predictions for {} samples...'.format(len(loader.dataset)))
            with torch.no_grad():
                for batch_idx, data in enumerate(loader):
                    data = data.to(device)
                    output = model(data)
                    probs = output.cpu().numpy()
                    indices = np.arange(len(probs)) + batch_idx * loader.batch_size

                    total_probs.extend(probs)
                    sample_indices.extend(indices)
                    total_labels.extend(data.y.view(-1, 1).cpu().numpy())

            total_probs = np.array(total_probs).flatten()
            sample_indices = np.array(sample_indices).flatten()
            total_labels = np.array(total_labels).flatten()

            return total_probs, sample_indices


        def save_predictions(probs, indices, file_name='all_predict_02.csv'):
            # Convert probabilities and indices to a DataFrame
            predictions_df = pd.DataFrame({
                'Index': indices,  # 保存样本索引
                'Probability': probs  # 保存预测概率
            })

            # Save to CSV without sorting
            predictions_df.to_csv(file_name, index=False)
            logging.info(f'Predictions saved to {file_name}')

        # 检测一下这段代码写的是否正确
        def save_top_30_predictions(probs, indices, file_name='top_30_predictions_02.csv'):
            # Sort by probability (in descending order)
            sorted_indices = np.argsort(probs)[::-1]  # sort in descending order
            sorted_probs = probs[sorted_indices]
            # sorted_labels = labels[sorted_indices]
            sorted_indices = indices[sorted_indices]

            # Create a DataFrame to save the top 30 predictions
            top_30_df = pd.DataFrame({
                'Index': sorted_indices[:30],
                'Probability': sorted_probs[:30],
                # 'True_Label': sorted_labels[:30]
            })

            # Save to CSV
            top_30_df.to_csv(file_name, index=False)
            logging.info(f'Top 30 predictions saved to {file_name}')

        # 保存 rna信息
        import pandas as pd

        def map_top_30_to_rna(top_30_file, output_file='top_30_miRNA_predictions.csv'):
            # 1. 读取miRNA_sequences文件
            miRNA_df = rna  # 加载miRNA序列文件

            # 2. 读取top 30预测文件
            top_30_df = pd.read_csv(top_30_file)  # 加载top 30预测文件

            # 3. 根据Index，将top 30的预测与miRNA_sequences的数据合并
            merged_df = pd.merge(top_30_df, miRNA_df, left_on='Index', right_index=True, how='left')

            # 4. 选择需要保存的列
            result_df = merged_df[['RNA_ID', 'Sequence', 'Probability']]

            # 5. 保存为CSV文件
            result_df.to_csv(output_file, index=False)
            print(f'Top 30 miRNA predictions saved to {output_file}')
            # return result_df


        modeling = GCNNetmuti

        model_st = modeling.__name__

        cuda_name = "cuda:0"
        if len(sys.argv) > 3:
            cuda_name = "cuda:" + str(int(sys.argv[1]))
        print('cuda_name:', cuda_name)

        # TRAIN_BATCH_SIZE = 64
        TEST_BATCH_SIZE = 64
        LR = 0.0005
        # LOG_INTERVAL = 160
        NUM_EPOCHS = 100

        print('Learning rate: ', LR)
        print('Epochs: ', NUM_EPOCHS)

        # Main program: iterate over different datasets

        print('\nrunning on ', model_st + '_')

        log_filename = f'training_1.log'

        logging.basicConfig(filename=log_filename, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.INFO)

        processed_data_file_test = 'dj_api/data/processed/last/' + '_mytest' + str_i + '.pt'
        if ((not os.path.isfile(processed_data_file_test))):
            print('please run process_data_old.py to prepare data in pytorch format!')
        else:
            test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=True)

            # training the model
            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
            model = modeling().to(device)
            model_path = "dj_api/model_GCNNetmuti_.model"
            model.load_state_dict(torch.load(model_path))

            loss_fn = nn.BCELoss()  # for classification


            probs, indices = predicting(model, device, test_loader)

            save_predictions(probs, indices, 'all_predict_0' + str_i + '.csv')
            save_top_30_predictions(probs, indices, 'top_30_predictions_0' + str_i + '.csv')

            # 现在是把前面的数据都给获取到了 但是我们要把数据和mirna对应上
            top_30_file = 'top_30_predictions_0' + str_i + '.csv'
            map_top_30_to_rna(top_30_file, 'top_30_miRNA_predictions_0' + str_i + '.csv')


            csv_file_path = 'top_30_miRNA_predictions_0' + str_i + '.csv'
            data_list = []

            # 读取 CSV 文件
            with open(csv_file_path, mode="r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                for index, row in enumerate(reader):
                    if index >= 5:  # 只获取前五条数据
                        break
                    data_list.append({
                        "RNA_ID": row.get("RNA_ID"),
                        "Sequence": row.get("Sequence"),
                        "Probability": row.get("Probability")
                    })


            cache.set(cache_key, data_list, timeout=3600)  # 缓存保存1小时
            # 返回 JSON 响应
            return JsonResponse({'code':0,'msg':'查询成功，内容如下',"data": data_list})


def get_all_rnas(request):
    if request.method == "POST":
        body = json.loads(request.body)
        dataList = body.get("data")
        all_results = []

        for i in range(len(dataList)):
            drug_sequence = dataList[i]["sequence"]
            print(drug_sequence)
            cache_key = f"drug_sequence_{drug_sequence}"  # 使用 drug_sequence 作为缓存的 key
            cached_result = cache.get(cache_key)

            if cached_result:
                print("Returning cached result")
                all_results.extend([(drug_sequence, rna["RNA_ID"],rna["Sequence"],rna["Probability"]) for rna in cached_result])
                continue

            raw_data = drug_sequence

            CHARISOSMILEN = 66

            # 读取整个 Excel 文件
            drugs = pd.read_excel('dj_api/data/drug_id_smiles.xlsx')
            rna = pd.read_excel('dj_api/data/miRNA_sequences.xlsx')

            str_i = '1'

            # 处理 drug_sequence
            my_process_smile = process_smiles(raw_data)

            compound_iso_smiles = [my_process_smile] * len(rna)
            affinity = [0] * len(rna)

            final_df = pd.DataFrame({
                'compound_iso_smiles': compound_iso_smiles,
                'target_sequence': rna['Sequence'],
                'affinity': affinity
            })

            output_file = f'dj_api/data/processed/last/_mytest{str_i}.csv'
            final_df.to_csv(output_file, index=False)

            seq_voc = "ACGU"
            seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
            seq_dict_len = len(seq_dict)

            opts = ['mytest']
            for j in range(1, 2):
                compound_iso_smiles = []
                for opt in opts:
                    df = pd.read_csv(f'dj_api/data/processed/last/_{opt}{str_i}.csv')
                    compound_iso_smiles += list(df['compound_iso_smiles'])
                compound_iso_smiles = set(compound_iso_smiles)
                smile_graph = {}
                for smile in compound_iso_smiles:
                    g = smile_to_graph(smile)
                    smile_graph[smile] = g

                df = pd.read_csv(f'dj_api/data/processed/last/_mytest{str_i}.csv')
                test_drugs, test_prots, test_Y = list(df['compound_iso_smiles']), list(df['target_sequence']), list(
                    df['affinity'])
                XT = [seq_cat(t, 24) for t in test_prots]
                test_sdrugs = [label_smiles(t, CHARISOSMISET, 100) for t in test_drugs]
                test_drugs, test_prots, test_Y, test_seqdrugs = np.asarray(test_drugs), np.asarray(XT), np.asarray(
                    test_Y), np.asarray(test_sdrugs)
                test_data = TestbedDataset(root='dj_api/data', dataset=f'last/_mytest{str_i}', xd=test_drugs,
                                           xt=test_prots, y=test_Y, z=test_seqdrugs, smile_graph=smile_graph)

            def predicting(model, device, loader):
                model.eval()
                total_probs = []
                sample_indices = []

                logging.info(f'Making predictions for {len(loader.dataset)} samples...')
                with torch.no_grad():
                    for batch_idx, data in enumerate(loader):
                        data = data.to(device)
                        output = model(data)
                        probs = output.cpu().numpy()
                        indices = np.arange(len(probs)) + batch_idx * loader.batch_size

                        total_probs.extend(probs)
                        sample_indices.extend(indices)

                total_probs = np.array(total_probs).flatten()
                sample_indices = np.array(sample_indices).flatten()

                return total_probs, sample_indices

            def save_predictions(probs, indices, file_name='all_predict_02.csv'):
                predictions_df = pd.DataFrame({
                    'Index': indices,
                    'Probability': probs
                })

                predictions_df.to_csv(file_name, index=False)
                logging.info(f'Predictions saved to {file_name}')

            def save_top_30_predictions(probs, indices, file_name='top_30_predictions_02.csv'):
                sorted_indices = np.argsort(probs)[::-1]
                sorted_probs = probs[sorted_indices]
                sorted_indices = indices[sorted_indices]

                top_30_df = pd.DataFrame({
                    'Index': sorted_indices[:30],
                    'Probability': sorted_probs[:30]
                })

                top_30_df.to_csv(file_name, index=False)
                logging.info(f'Top 30 predictions saved to {file_name}')


            def map_top_30_to_rna(top_30_file, output_file='top_30_miRNA_predictions.csv'):
                miRNA_df = rna

                top_30_df = pd.read_csv(top_30_file)

                merged_df = pd.merge(top_30_df, miRNA_df, left_on='Index', right_index=True, how='left')

                result_df = merged_df[['RNA_ID', 'Sequence', 'Probability']]

                result_df.to_csv(output_file, index=False)
                print(f'Top 30 miRNA predictions saved to {output_file}')

            modeling = GCNNetmuti

            model_st = modeling.__name__

            cuda_name = "cuda:0"
            if len(sys.argv) > 3:
                cuda_name = "cuda:" + str(int(sys.argv[1]))
            print('cuda_name:', cuda_name)

            TEST_BATCH_SIZE = 64
            LR = 0.0005
            NUM_EPOCHS = 100

            print('Learning rate: ', LR)
            print('Epochs: ', NUM_EPOCHS)

            print('\nrunning on ', model_st + '_')

            log_filename = 'training_1.log'

            logging.basicConfig(filename=log_filename, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s',
                                level=logging.INFO)

            processed_data_file_test = f'dj_api/data/processed/last/_mytest{str_i}.pt'
            if not os.path.isfile(processed_data_file_test):
                print('please run process_data_old.py to prepare data in pytorch format!')
            else:
                test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=True)

                device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
                model = modeling().to(device)
                model_path = "dj_api/model_GCNNetmuti_.model"
                model.load_state_dict(torch.load(model_path))

                loss_fn = nn.BCELoss()

                probs, indices = predicting(model, device, test_loader)

                save_predictions(probs, indices, f'all_predict_0{str_i}.csv')
                save_top_30_predictions(probs, indices, f'top_30_predictions_0{str_i}.csv')

                top_30_file = f'top_30_predictions_0{str_i}.csv'
                map_top_30_to_rna(top_30_file, f'top_30_miRNA_predictions_0{str_i}.csv')

                csv_file_path = f'top_30_miRNA_predictions_0{str_i}.csv'
                data_list = []

                with open(csv_file_path, mode="r", encoding="utf-8") as file:
                    reader = csv.DictReader(file)
                    for index, row in enumerate(reader):
                        if index >= 5:
                            break
                        data_list.append({
                            "RNA_ID": row.get("RNA_ID"),
                            "Sequence": row.get("Sequence"),
                            "Probability": row.get("Probability")
                        })

                cache.set(cache_key, data_list, timeout=3600)

                all_results.extend([(drug_sequence, rna["RNA_ID"],rna["Sequence"],rna["Probability"]) for rna in data_list])

        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df = pd.DataFrame(all_results, columns=["Drug Sequence", "RNA_ID", "Sequence","Probability"])
            df.to_excel(writer, index=False, sheet_name='Sheet1')

        output.seek(0)
        file_name = "predicted_rnas.xlsx"
        file_path = os.path.join(settings.MEDIA_ROOT, file_name)

        with open(file_path, "wb") as file:
            file.write(output.read())

        download_link = request.build_absolute_uri(settings.MEDIA_URL + file_name)

        return JsonResponse({'code': 0, 'msg': 'xlsx文件发送成功', 'data': download_link})





def test_redis_cache(request):
    # 获取缓存中的数据
    cached_data = cache.get('test_key')

    if cached_data is None:
        # 如果缓存中没有数据，设置缓存并返回响应
        cache.set('test_key', 'This is a test value', timeout=60)
        response = {
            'status': 'success',
            'message': 'No data found in cache. Data has been cached.',
            'cached_data': 'This is a test value'
        }
    else:
        # 如果缓存中有数据，直接返回缓存的数据
        response = {
            'status': 'success',
            'message': 'Data fetched from cache.',
            'cached_data': cached_data
        }

    return JsonResponse(response)