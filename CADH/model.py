import torch
import torch.nn as nn
#from feature_extra import TransformerEncoder
#from feature_extra import ShareExtractor, Domain_Extractor
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cuda:1")
'''nn.Linear(768, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),'''
# class domain_extractor(nn.Module):
#     def __init__(self,domain_dim):
#         super(domain_extractor, self).__init__()
#         self.domain_extractor = nn.Sequential(
#             nn.Linear(768, 2048),
#             #nn.BatchNorm1d(2048),
#             nn.ReLU(),
#             nn.Linear(2048, 512),
#             nn.BatchNorm1d(512),
#             nn.Dropout(0.5),
#             nn.Linear(512, domain_dim),
            
#         )
#     def forward(self,x):
#         x = self.domain_extractor(x)
#         return x
class domain_extractor(nn.Module):
    def __init__(self,domain_dim):
        super(domain_extractor, self).__init__()
        self.domain_extractor = nn.Sequential(
            
            nn.Linear(768, 2048),
            #nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, domain_dim),
        )
    def forward(self,x):
        x = self.domain_extractor(x)
        return x
class shared_extractor(nn.Module):
    def __init__(self,share_domain_dim):
        super(shared_extractor, self).__init__()
        self.shared_extractor = nn.Sequential(
            
            nn.Linear(768, 2048),
            #nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024,share_domain_dim),

        )

        
    def forward(self,x):
        x = self.shared_extractor(x)
        return x 
# class shared_extractor(nn.Module):
#     def __init__(self,share_domain_dim):
#         super(shared_extractor, self).__init__()
#         self.shared_extractor = nn.Sequential(
#             nn.Linear(768, 2048),
#             #nn.BatchNorm1d(2048),
#             nn.ReLU(),
#             nn.Linear(2048, 1024),
#             nn.BatchNorm1d(1024),
#             nn.Dropout(0.5),
#             nn.Linear(1024,share_domain_dim),

#         )
        
#     def forward(self,x):
#         x = self.shared_extractor(x)
#         return x 

#投影层
class ProjectionResidualFusion(nn.Module):
    def __init__(self, input_dim_e, input_dim_a, proj_output_dim):
        super().__init__()                                                                                                                                                  
        self.proj_head = nn.Sequential(
            nn.Linear(input_dim_e + input_dim_a, 2048),
            #nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, proj_output_dim)
        )

    def forward(self, e, a):
        #print(e.size(), a.size())
        # 拼接：[a, e]
        concat = torch.cat([a, e], dim=1)  # (B, D1 + D
        # 投影 + 残差
        projected = self.proj_head(concat)  # (B, D_proj)
        #print(projected.size())
        fused = projected + e # 残差融合，要求 projected 和 e 维度一致   + e

        return fused 
class hash_encoder(nn.Module):
    def __init__(self,nbits,dim1):
        super(hash_encoder, self).__init__()
        self.encoder= nn.Sequential(

            nn.Linear(dim1, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=False),
            nn.Linear(1024, nbits),
            nn.Tanh()
        )
    def forward(self,x):
        x = self.encoder(x)
        return x
#拼接使用concat = torch.cat([a, b], dim=1) 

class GradientReversalFunction(torch.autograd.Function):
    #
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        #print("GRL backward grad mean:", grad_output.mean().item())
        return ctx.lambda_ * grad_output.neg(), None

def grad_reverse(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)

#分类器
#假设输入特征维度为4096，输出类别数为num_classes
class Classifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        #self.fc = nn.Linear(in_features, num_classes)
        self.classifier_net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.classifier_net(x)
#共享编码器域分类器
class share_DomainClassifier(nn.Module):
    def __init__(self, feat_dim, hidden_size=100):   # 800 是 CNN 出来的特征维度
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, 2)          # 2 个输出 → 源域/目标域
            ,nn.LogSoftmax(dim=1)        # 输出 log-prob
        )
    def forward(self, x, lambda_=1.0):
        x = grad_reverse(x, lambda_)
        return self.net(x)  
#专用编码器语义分类器
class DomainSpecific(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super().__init__()

        # 2) 域特有特征提取器（两个轻量 MLP 即可）
        self.fs_head = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.BatchNorm1d(1024),
            
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
            )
    def forward(self, x):
        logits = self.fs_head(x)
        return logits


#定义总类
class AllModel(nn.Module):
    def __init__(self, feature_dim,nbits, num_classes, device):
        super(AllModel, self).__init__()
        self.shared_extractor = shared_extractor(feature_dim).to(device)
        self.domain_src_extractor = domain_extractor(feature_dim).to(device)#区分了源域和目标域的权重
        self.domain_tag_extractor = domain_extractor(feature_dim).to(device)

        #投影层可以不一样
        self.proj_fusion_src = ProjectionResidualFusion(input_dim_e=feature_dim, input_dim_a=feature_dim, proj_output_dim=feature_dim).to(device)
        self.proj_fusion_tag = ProjectionResidualFusion(input_dim_e=feature_dim, input_dim_a=feature_dim, proj_output_dim=feature_dim).to(device)
        print(nbits)
        self.hash_encoder = hash_encoder(nbits=nbits, dim1=4096).to(device)


        self.classifier = Classifier(in_features=4096, num_classes=num_classes).to(device)

        self.share_domain_classifier = share_DomainClassifier(feat_dim=4096, hidden_size=800).to(device)
        #尝试一个或者两个的分类器(DANN的域初始化就初始化了一个)
        self.domainspecific_src = DomainSpecific(feat_dim=feature_dim,num_classes = num_classes).to(device)
        self.domainspecific_tag = DomainSpecific(feat_dim=feature_dim,num_classes = num_classes).to(device)
        
    def forward(self, source,target, lambda_=1.0):
        # 共享特征提取器
        shared_feat_source = self.shared_extractor(source)  
        shared_feat_target = self.shared_extractor(target)
        #grad_reverse(shared_feat_source, lambda_)
        #域特征提取器
        domain_feat_source = self.domain_src_extractor(source)  
        domain_feat_target = self.domain_src_extractor(target)
        #投影残差融合
        fused_feat_source = self.proj_fusion_src(shared_feat_source, domain_feat_source)  
        fused_feat_target = self.proj_fusion_tag(shared_feat_target, domain_feat_target)
        # 哈希编码器
        hash_code_source = self.hash_encoder(shared_feat_source)  
        hash_code_target = self.hash_encoder(shared_feat_target)
        # 分类器
        class_logits_source = self.classifier(fused_feat_source)
        class_logits_target = self.classifier(fused_feat_target)
        # 共享域分类器  用来求解域对齐 2分类
        dom_src = self.share_domain_classifier(shared_feat_source, lambda_)#共享域分类器修改过
        dom_tgt = self.share_domain_classifier(shared_feat_target , lambda_)
        # 专用域分类器
        dom_spe_src = self.domainspecific_src(domain_feat_source)
        dom_spe_tag = self.domainspecific_tag(domain_feat_target)
        return {
            'shared_feat_source': shared_feat_source,
            'shared_feat_target': shared_feat_target,
            'domain_feat_source': domain_feat_source,
            'domain_feat_target': domain_feat_target,
            'fused_feat_source': fused_feat_source,
            'fused_feat_target': fused_feat_target,
            'hash_code_source': hash_code_source,
            'hash_code_target': hash_code_target,
            # 分类器输出
            'class_logits_source': class_logits_source,
            'class_logits_target': class_logits_target,

            'dom_src': dom_src,
            'dom_tgt': dom_tgt,

            'dom_spe_src':dom_spe_src,
            'dom_spe_tag':dom_spe_tag

        } 
    @torch.no_grad()
    def predict(self, x):
        shared_feat = self.shared_extractor(x)
        domain_feat = self.domain_tag_extractor(x)
        fused_feat = self.proj_fusion_tag(shared_feat, domain_feat)
        hash_code = self.hash_encoder(shared_feat)
        class_logits = self.classifier(fused_feat)
        return {
            'shared_feat': shared_feat,
            'domain_feat': domain_feat,
            'fused_feat': fused_feat,
            'hash_code': hash_code,
            'class_logits': class_logits
        }