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