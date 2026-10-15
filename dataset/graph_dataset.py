import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from featurizer.pdb_feature import make_feature
from featurizer.graph_feature import construct_graph_data_single
from utils.common import to_tensor

class GraphDataset(Dataset):
    def __init__(self, data_path, data_config):
        """
        Args:
            data_path: CSV 数据路径 (split_data.csv)
            data_config: 配置字典
        """
        super(GraphDataset, self).__init__()

        self.data = pd.read_csv(data_path)
        self.data_config = data_config
        self.graph_feat_config = data_config.get('graph_featurizer', {})
        self.name_idx = self.data['pdb_id'].to_list()

        self.use_ss_feature = self.data_config.get('use_ss_feature', False)

        # LLM Embedding 存储目录 (预计算好的 .pt 文件)
        self.embedding_dir = self.data_config.get('llm_embedding_dir')

        if 'secondary_structure' in list(self.data.columns):
            self.secondary_structure = self.data['secondary_structure'].to_list()
        else:
            self.secondary_structure = None

        print(f"GraphDataset initialized. Size: {len(self.name_idx)}")

    def __len__(self):
        return len(self.name_idx)

    def __getitem__(self, index):
        if index is None:
            return None

        pdb_id = self.name_idx[index]

        # ---------------------------------------------------------------------
        # 1. 获取基础结构特征 (Ground Truth & Input Backbone)
        # ---------------------------------------------------------------------
        # get_structure 返回: coords_coarse, coords_gt, angles_gt, seq(index), mask 等
        try:
            if self.secondary_structure is not None:
                ret = make_feature(
                    os.path.join(self.data_config['pdb_data_dir'], f'{pdb_id}.pdb'),
                    self.secondary_structure[index]
                )
            else:
                ret = make_feature(
                    os.path.join(self.data_config['pdb_data_dir'], f'{pdb_id}.pdb')
                )
            if ret is None:
                return None
        except Exception as e:
            print(f"Error loading structure {pdb_id}: {e}")
            return None

        # ---------------------------------------------------------------------
        # 2. 获取 RNA-FM Embedding (Condition / Repa)
        # ---------------------------------------------------------------------
        emb_path = os.path.join(self.embedding_dir, f"{pdb_id}.pt")
        #print("emb_path: ", emb_path)
        if os.path.exists(emb_path):
            llm_emb = torch.load(emb_path).float()  # [L, 640]
        else:
            # Fallback: 如果没有 embedding，给个零矩阵 (或在预处理时保证都有)
            # print(f"Warning: No embedding for {pdb_id}")
            L_struct = ret['coords_coarse'].shape[0]
            llm_emb = torch.zeros((L_struct, 1))

        #print("llm_emb: ", llm_emb.shape, llm_emb.dtype)

        # ---------------------------------------------------------------------
        # 3. 关键步骤：长度对齐 (Alignment)
        # ---------------------------------------------------------------------
        # 必须保证 PDB特征 和 Embedding 长度严格一致，否则 PyG Batch 会报错
        L_coords = ret['coords_coarse'].shape[0]
        L_emb = llm_emb.shape[0]

        if L_coords != L_emb:
            raise ValueError("embedding length do not match!!!!")

        # ---------------------------------------------------------------------
        # 4. 构建图特征 (Graph Construction)
        # ---------------------------------------------------------------------
        # 这里的输入是 Coarse Backbone，用于计算 VAE Encoder 需要的几何标量/向量特征

        # 读取配置
        num_posenc = self.graph_feat_config.get('num_posenc', 16)
        num_rbf = self.graph_feat_config.get('num_rbf', 16)
        knn_num = self.graph_feat_config.get('knn_num', 10)

        # 调用 graph_feature.py
        data = construct_graph_data_single(
            coords=ret['coords_coarse'],
            seq=ret['seq'],
            mask=ret['mask_coarse'][:, 1],  # C1' mask
            ss_pairs=ret.get('ss_pairs', None),
            num_posenc=num_posenc,
            num_rbf=num_rbf,
            knn_num=knn_num,
            use_ss_feature=self.use_ss_feature,
        )

        # print("ss_pairs: ", ret['ss_pairs'])

        # ---------------------------------------------------------------------
        # 5. 组装 Data 对象
        # ---------------------------------------------------------------------
        graph = Data(
            # === Node Features (VAE Input) ===
            # 这里 graph_feature 返回的 data['seq'] 是 one-hot [L, 4]
            x=data['seq'],

            # extra_x: 几何标量特征 (Internal Coords: Angles, Bond lengths) [L, 9]
            extra_x=data["extra_x"],

            # node_v: 几何向量特征 (Frames) [L, 3, 3]
            node_v=data["node_v"],

            # === Edge Features ===
            edge_index=data["edge_index"],
            edge_attr=data["edge_attr"],  # RBF + PosEnc
            edge_v=data["edge_v"],

            # === Condition / Auxiliary Features ===
            # llm_opt: 这里放 LLM Embedding，作为“外挂”特征
            llm_opt=llm_emb,

            # === Ground Truth (用于 Loss 计算) ===
            pos=to_tensor(data["pos"], torch.float32),  # C1' 坐标 (Graph Node Pos)
            coords_coarse=to_tensor(ret['coords_coarse'], torch.float32),  # Input Backbone

            # 预测目标：全原子坐标 & 扭转角
            coords_gt=to_tensor(ret['coords_gt'], torch.float32),
            mask_gt=to_tensor(ret['mask_gt'], torch.bool),

            angles_gt=to_tensor(ret['angles_gt'], torch.float32),
            mask_angles=to_tensor(ret['mask_angles'], torch.bool),

            # 碰撞检测
            coords_obstacle=to_tensor(ret['coords_obstacle'], torch.float32),

            # Meta info
            seq_idx=to_tensor(ret['seq'], torch.int64),  # 整数索引序列
            name=pdb_id
        )

        return {"graph": graph}

    def collate_fn(self, batch):
        batch_graph_list = []
        for info in batch:
            if info is None:
                continue
            batch_graph_list.append(info['graph'])

        if len(batch_graph_list) == 0:
            return None

        return Batch.from_data_list(batch_graph_list)