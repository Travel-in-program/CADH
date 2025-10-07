import argparse
import torch
import numpy as np
import os
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import random
from load_data import get_source_data, get_target_data
from model import ProjectionResidualFusion, hash_encoder, Classifier,share_DomainClassifier,AllModel
from loss import *  # 你可以根据自己定义的损失函数替换
from tqdm import tqdm
from torchvision import transforms
from feature_extra import CustomImageDataset,TransformerEncoder
from sklearn.model_selection import train_test_split
import scipy.io as sio
#from test import performance_eval_real#performance_eval,evaluate,build_codes_and_labels,mean_average_precision1,
from mask import FeatureMasker
from torch.optim.lr_scheduler import CosineAnnealingLR
def seed_setting(seed=2025):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
#seed_setting(seed=2025)
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Office-Home', choices=['Office-Home', 'Office-31', 'Digits'])
    parser.add_argument('--nbit', type=int, default=64, choices=[16, 32, 64, 128])
    parser.add_argument('--batchsize', type=int, default=256) # 20230725 128 -》256
    parser.add_argument('--num_epoch', type=int, default=50) # 50
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--w', type=float, default=1e-4)
    #parser.add_argument("--n_iter_per_epoch", type=int, default=20, help="Used in Iteration-based training")
    parser.add_argument('--alpha', type=float, default=0.001)
    parser.add_argument('--mask_ratio', type=float, default=0.05)#0.3
    parser.add_argument('--block_size', type=float, default=8) # 1
    parser.add_argument('--cls', type=float, default=1)
    parser.add_argument('--dmc', type=float, default=0)
    parser.add_argument('--doma', type=float, default=0)
    parser.add_argument('--hash', type=float, default=2.0)
    parser.add_argument('--mask', type=float, default=0.05)
    parser.add_argument('--coral', type=float, default=0)
    parser.add_argument('--kernel_mul', type=float, default=1.5)
    parser.add_argument('--kernel_num', type=int, default=7)
    parser.add_argument('--domain', type=str, default='ArtToReal_World') # ArtToReal_World AmazonToDslr
    # print
    args = parser.parse_args()
    print(args)

    return args
def performance_eval_real(model, database_loader, query_loader):

    model.eval().to(device)
    re_BI, re_L, qu_BI, qu_L = compress1(database_loader, query_loader, model)
    print('re_BI.shape',re_BI.shape)
    print('qu_BI.shape',qu_BI.shape)
    ## Save
    _dict = {
        'retrieval_B': re_BI,
        'L_db':re_L,
        'val_B': qu_BI,
        'L_te':qu_L,
    }
    #print(_dict['L_te'],qu_L.shape)
    sava_path = 'hashcode/HASH1_' + args.dataset + '_'  + str(args.nbit)  + 'bits.mat'#'hashcode/HASH_' + 'Office-Home' + '_' + str(64) + 'bits.mat' +str(args.domain)
    sio.savemat(sava_path, _dict)

    return 0

def compress1(database_loader, query_loader, model):

    # retrieval
    re_BI = list([])
    re_L = list([])
    for _, (data_I, data_L, _) in enumerate(database_loader):
        with torch.no_grad():
            var_data_I = data_I.to(device)
            code_I = model.predict(var_data_I.to(torch.float))['hash_code']
        code_I = torch.sign(code_I)
        re_BI.extend(code_I.cpu().data.numpy())
        re_L.extend(data_L.cpu().data.numpy())

    # query
    qu_BI = list([])
    qu_L = list([])
    for _, (data_I, data_L, _) in enumerate(query_loader):
        with torch.no_grad():
            var_data_I = data_I.to(device)
            code_I = model.predict(var_data_I.to(torch.float))['hash_code']
        code_I = torch.sign(code_I)
        qu_BI.extend(code_I.cpu().data.numpy())
        qu_L.extend(data_L.cpu().data.numpy())

    re_BI = np.array(re_BI)
    re_L = np.array(re_L)

    qu_BI = np.array(qu_BI)
    qu_L = np.array(qu_L)

    return re_BI, re_L, qu_BI, qu_L

#伪标签生成策略：先根据共享特征提取器生成的特征图提取中心，对标到伪标签上
def compute_centers(x, psedo_labels, num_cluster):
    n_samples = x.size(0)
    #print(n_samples)
    if len(psedo_labels.size()) > 1:
        # 如果伪标签 psedo_labels 是一个二维张量，则将其转置得到一个形状为 (n_samples, num_cluster) 的张量
        weight = psedo_labels.T
        #print(weight)
    else:
        # 如果伪标签 psedo_labels 是一个一维张量，则创建一个形状为 (num_cluster, n_samples) 的零张量 weight，
        # 并将其中第 psedo_labels[i] 行、第 i 列的元素设置为 1，表示第 i 个样本属于第 psedo_labels[i] 个聚类
        weight = torch.zeros(num_cluster, n_samples).to(x)  # L, N
        weight[psedo_labels, torch.arange(n_samples)] = 1

    weight = weight.float()
    # 对 weight 进行 L1 归一化，即将每一行的元素值都除以该行的元素之和，以确保每个聚类的权重之和为 1
    weight = F.normalize(weight, p=1, dim=1)  # l1 normalization
    # 通过矩阵乘法 torch.mm(weight, x) 将 weight 和 x 相乘，得到每个聚类的样本特征的加权平均值，即聚类中心
    centers = torch.mm(weight, x)
    # 对聚类中心进行 L2 归一化，以确保每个聚类中心向量的长度为1
    centers = F.normalize(centers, dim=1)
    return centers
@torch.no_grad()
def psedo_labeling(num_cluster, batch_features, centers, batch_target):
    l2_normalize =True
    torch.cuda.empty_cache()
    # (1) 进行L2归一化
    if l2_normalize:
        batch_features = F.normalize(batch_features, dim=1)
        batch_features_cpu = batch_features.cpu()
    # (2) 计算batch_feat和source_centers的相似度矩阵
    # 为了使用F.cosine_similarity函数,需要将cluster_centers和centers分别扩展成大小为(c, 1, d)和(1, n, d)的三维张量,以便在第二维上进行广播操作
    # todo ok
    centers_cpu = centers.cpu()
    btarget_cen_similarity = F.cosine_similarity(batch_features_cpu.unsqueeze(1), centers_cpu.unsqueeze(0), dim=2)

    # (3) 得到源域和目标域标签的对应关系
    relation = torch.zeros(num_cluster, dtype=torch.int64) - 1
    # 对similarity中的每一行进行排序,并按照从大到小的顺序记录索引
    sorted_indices = torch.argsort(btarget_cen_similarity, dim=1, descending=True)
    new_cluster_labels = sorted_indices[:, 0]

    return new_cluster_labels
#test
def train(args,model, source_loader, target_loader,test_loader,num_classes, device, optimizer, epochs,loss_weights):
    model.train()
    len_source_loader = len(source_loader)  # 获取源域数据加载器中batch的数量 19
    len_target_loader = len(target_loader)  # 31
    memory_source_centers = torch.zeros(num_classes, 4096).to(device)
    memory_target_centers = torch.zeros(num_classes, 4096).to(device)
#迭代数据
    steps = min(len_source_loader, len_target_loader)  
    i = 0
    for epoch in range(epochs): # 50
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        total_loss_epoch = 0.0   # 每个 epoch 初始化
        total_correct = 0
        iter_source, iter_target = iter(source_loader), iter(target_loader)
        if max(len_target_loader, len_source_loader) != 0:
            iter_source, iter_target = iter(source_loader), iter(target_loader)

        #pbar = tqdm(range(steps), desc=f"Epoch {epoch+1}/{epochs}")
        
        for abcd in range(steps):#pbar
            source_data, source_labels, _= next(iter_source)              
            target_data, target_label, _= next(iter_target)   
            
            source_data, source_labels = source_data.to(device), source_labels.to(device)
            target_data = target_data.to(device)
            target_label = target_label.to(device)
            optimizer.zero_grad()
            '''p = float(i + epoch * steps) / epochs / steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1'''
            all = model(source_data,target_data,lambda_=1)
            # shared_source_data,shared_target_data,\
            # hash_source_data,hash_target_data,\
            # = all['shared_feat_source'],all['shared_feat_target'],\
            #                  all['hash_code_source'],all['hash_code_target'],\
                             
            shared_source_data,shared_target_data,domain_source_data,domain_target_data,\
            proj_source_data,proj_target_data,hash_source_data,hash_target_data,\
            class_logits_source,class_logits_target,dom_src,dom_tgt,dom_spe_src,dom_spe_tag= all['shared_feat_source'],all['shared_feat_target'],all['domain_feat_source'],all['domain_feat_target'],\
                           all['fused_feat_source'],all['fused_feat_target'],all['hash_code_source'],all['hash_code_target'],\
                           all['class_logits_source'],all['class_logits_target'],all['dom_src'],all['dom_tgt'],all['dom_spe_src'],all['dom_spe_tag']
            
            #生成伪标签c
            #计算源域的原型
            
            batch_source_centers = compute_centers(shared_source_data, source_labels.squeeze(), num_classes)
            memory_source_centers = memory_source_centers.detach() + batch_source_centers
            # 对聚类中心进行 L2 归一化，以确保每个聚类中心向量的长度为1
            memory_source_centers = F.normalize(memory_source_centers, dim=1)
            #计算目标域的伪标签和原型
            with torch.no_grad():
            #不能持续使用源域，修改成目标域
                target_plabel = psedo_labeling(num_classes, shared_target_data, memory_source_centers,target_label)
                target_plabel = target_plabel.to(device)
                #print("%%%%")
                #print(target_plabel.shape)

                batch_target_centers = compute_centers(shared_target_data, target_plabel, num_classes)

                memory_target_centers = memory_target_centers.detach() + batch_target_centers
                memory_target_centers = F.normalize(memory_target_centers, dim=1)
                target_plabel = psedo_labeling(num_classes, shared_target_data, memory_target_centers,target_label)

            # 计算损失
            #域损失domain_loss为交叉熵损失     ys与yt均为one-hot编码的标签
            # 域分类损失
            pred_tgt = class_logits_target
            #目标域的伪标签
            pseudo_labels1 = target_plabel.to(device)
            #print(pseudo_labels1)
            #计算的个数
            correct_sample = torch.sum(pred_tgt.argmax(dim=1) == target_label.squeeze())
            correct += correct_sample
            pred_src = class_logits_source
            source_labels = source_labels.squeeze()
            
            loss_dmc = domain_classification_loss(dom_spe_src, source_labels, dom_spe_tag, pseudo_labels1, num_classes)
            total += target_label.size(0)

            #域对齐损失

            #loss_domain = domain_adversarial_loss(dom_src, dom_tgt)

        #掩码损失#设置掩码块，并初始化一个mask的实例
            '''mask_ratio =args.mask_ratio
            if epoch>=40:
                mask_ratio = 0.3
            mask_generator = FeatureMasker(mask_ratio=mask_ratio, block_size=args.block_size)#111111111111111
            loss_mask = mask_consistency_loss(model, class_logits_target,target_data, mask_generator,pseudo_labels1, num_classes)'''
            mask_generator = FeatureMasker(mask_ratio=args.mask_ratio, block_size=args.block_size)#111111111111111
            loss_mask = mask_consistency_loss(model, class_logits_target,target_data, mask_generator, pseudo_labels1,num_classes)
        #分类损失
            loss_cls = classification_loss(pred_src, source_labels,pred_tgt, pseudo_labels1)
            #print(loss_cls)
            # 计算哈希损失
            #编码器损失
         #mmd损失
            #mmd_loss = MMDLoss(kernel_type='rbf', kernel_mul=args.kernel_mul, kernel_num=args.kernel_num)#11111111111111    1.5    7
            #loss_coral = mmd_loss(shared_source_data,shared_target_data)#mmd_loss+(domain_source_data,domain_target_data)
            #源域的one—hot编码标签
            source_labels_onehot = F.one_hot(source_labels, num_classes=num_classes).float()
        # 计算哈希对齐损失
        #参考崔老师的程序，对照一边
            #print(pseudo_labels1.shape,source_labels_onehot.shape)
            sim_matrix = torch.mm(source_labels_onehot,F.one_hot(pseudo_labels1,num_classes=num_classes).float().t())  # 计算相似度矩阵
            #print(hash_source_data)#不是哈希码

            #print(hash_source_code)
            #量化损失可以多加入一个目标与的损失
            loss_hash = hash_pairwise_loss(hash_source_data, hash_target_data, sim_matrix, alpha=args.alpha)#  0.0008               11111111111111
            #loss_quant = quantization_loss(hash_source_data)

            total_loss = (
                loss_weights["cls"] * loss_cls +
                loss_weights["dmc"] * loss_dmc +
                #loss_weights["domain"] * loss_domain +
                loss_weights["hash"] * loss_hash 
                +loss_weights["mask"] * loss_mask
                #+loss_weights["coral"] * loss_coral
                #+loss_weights["quant"] * loss_quant
            )
            #清空旧梯度
            #optimizer.zero_grad()
            #计算梯度
            total_loss.backward()
            #更新参数
            #print('lr:',optimizer.param_groups[0])
            
            optimizer.step()
            #scheduler.step() 
            # 统计
            total_loss_epoch += total_loss.item()
        if (epoch + 1) % 10 == 0 and (abcd + 1) == steps:
            print({
            "loss_cls": f"{loss_cls.item():.4f}",
            "loss_dmc": f"{loss_dmc.item():.4f}",
            #"loss_domain": f"{loss_domain.item():.4f}",
            "loss_hash": f"{loss_hash.item():.4f}",
            "loss_mask": f"{loss_mask.item():.4f}",
            #"loss_coral": f"{loss_coral.item():.4f}",
            #"loss_quant": f"{loss_quant.item():.4f}",
            "total": f"{total_loss.item():.4f}"
        })
            # 打印 step 级别损失
            '''pbar.set_postfix({
                "loss_cls": f"{loss_cls.item():.4f}",
                "loss_dmc": f"{loss_dmc.item():.4f}",
                "loss_domain": f"{loss_domain.item():.4f}",
                "loss_hash": f"{loss_hash.item():.4f}",
                "loss_mask": f"{loss_mask.item():.4f}",
                #"loss_coral": f"{loss_coral.item():.4f}",
                #"loss_quant": f"{loss_quant.item():.4f}",
                "total": f"{total_loss.item():.4f}"
            })'''
        #缩进
            with torch.no_grad():
            # 每个 epoch 打印一次平均损失
                print(f"Epoch [{epoch+1}/{epochs}] - Avg Total Loss: {total_loss_epoch/steps:.4f}")
                acc = 100. * correct / total
                print('total acc: %.8f' %(acc))
        i+=1
    performance_eval_real(model, source_loader,test_loader)
    save_path = '/media/abc/ware/xsh/uad/program5_tiv/file/mnist/ptor2.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")



    
if __name__ == "__main__":
    device = torch.device("cuda:1")#cuda:1
    args = get_args()
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准均值和标准差
    ])
    #duqu txt文件中的路径和标签
    # image_paths_source = []


    labels_source = []
    batchsize = 256



    # base_path  = '/media/abc/ware/xsh/uad/data/use_data_tiv/mnist/'#'/media/abc/ware/xsh/uad/data/use_data_tiv/office_home/''/media/abc/ware/xsh/uad/data/use_data/'
    # source_domain = 'usps_mat_vit_b16'#'Real_World_feature_mat   usps_mat_vit_b16'
    # print(source_domain)
    # target_domain = 'Product_feature_mat'#'mnist_mat_vit_b16'
    '''source_loader, n_class, dim1 = get_source_data(base_path, source_domain,batchsize)
    print(n_class,dim1)
    target_loader = get_target_data(base_path, target_domain,batchsize)
    #训练集的数据量3995
    target_train_loader = target_loader['train']
    target_test_loader = target_loader['query']'''



    source_domain = args.domain.split('To')[0]#
    target_domain = args.domain.split('To')[1]#
    if args.dataset == 'Office-Home':  # (1) Art (2) Clipart (3) Product (4) Real_World
        base_path = '/media/abc/ware/xsh/uad/data/use_data_tiv/office_home/'#'/media/abc/ware/xsh/uad/data/use_data_tiv/office_home/'
        source_loader, n_class, dim1 = get_source_data(base_path, source_domain,args.batchsize)
        target_loader = get_target_data(base_path, target_domain,args.batchsize)
        print(len(target_loader))
        target_train_loader = target_loader['train']
        target_test_loader = target_loader['query']

    elif args.dataset == 'Office-31':  # (1) Amazon (2) Dslr (3) Webcam
        base_path = '/media/abc/ware/xsh/uad/data/use_data_tiv/office31/'
        source_loader, n_class, dim1 = get_source_data(base_path, source_domain,args.batchsize)
        target_loader = get_target_data(base_path, target_domain,args.batchsize)
        target_train_loader = target_loader['train']
        target_test_loader = target_loader['query']

    elif args.dataset == 'Digits':
        base_path = '/media/abc/ware/xsh/uad/data/use_data_tiv/mnist/'
        source_loader, n_class, dim1 = get_source_data(base_path, source_domain,args.batchsize)
        target_loader = get_target_data(base_path, target_domain,args.batchsize)
        target_train_loader = target_loader['train']
        target_test_loader = target_loader['query']
    else:
        raise Exception('No this dataset!')
    # 模型构建
    model = AllModel(feature_dim=4096,
                 nbits=args.nbit,
                 num_classes=n_class,                    #11111111111111#65
                 device=device).to(device)
    print(args.nbit)
    #model = nn.DataParallel(model)
    model = model.to(device)
    loss_weights = {
                        "cls": args.cls,      
                        "dmc": args.dmc,
                        "domain": args.doma,
                        "hash": args.hash,  
                        "mask": args.mask,
                        #"coral":args.coral,
                        #"quant":0.5
                    }
    # 优化器与损失
    optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=args.lr, weight_decay=args.w
                                )
    #lr=5e-5, weight_decay=1e-4
    #scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=3e-5)
    # 训练模型
    train(args,
            model,
            source_loader,
            target_train_loader, 
            target_test_loader,
            n_class,
            device, 
            optimizer,  
            epochs=args.num_epoch,#50
            loss_weights=loss_weights)

    

