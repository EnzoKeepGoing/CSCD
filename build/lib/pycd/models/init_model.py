# pycd/models/init_model.py
import torch

def create_model(args, concept_count, exercise_count, user_count):
    """
    根据模型名称创建相应的模型实例和参数

    参数:
        args: 命令行参数
        concept_count: 知识点数量
        exercise_count: 习题数量
        user_count: 用户数量

    返回:
        model: 模型实例
        model_params: 用于记录的模型参数
        optimizer: 优化器实例
    """
    model_name = args.model_name

    if model_name == 'dina':
        from pycd.models.dina import DINA
        
        model = DINA(
            user_num=user_count,
            item_num=exercise_count,
            hidden_dim=args.hidden_dim,
            concept_dim=concept_count,
            ste=bool(args.ste)
        )

        model_params = {
            'hidden_dim': args.hidden_dim,
            'concept_dim': concept_count,
            'ste': bool(args.ste),
            'max_slip': args.max_slip if hasattr(args, 'max_slip') else 0.4,
            'max_guess': args.max_guess if hasattr(args, 'max_guess') else 0.4,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'fold': args.fold
        }
        
        optimizer = torch.optim.Adam(model.dina_net.parameters(), lr=args.lr)

    elif model_name == 'irt':
        from pycd.models.irt import IRT
        
        model = IRT(
            n_students=user_count,
            n_exercises=exercise_count,
            value_range=args.value_range,
            a_range=args.a_range
        )
        
        model_params = {
            'value_range': args.value_range,
            'a_range': args.a_range,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'fold': args.fold
        }
        
        optimizer = torch.optim.Adam(model.irt_net.parameters(), lr=args.lr)

    elif model_name == 'mirt':
        from pycd.models.mirt import MIRT

        model = MIRT(
            n_students=user_count,
            n_exercises=exercise_count,
            n_concepts=args.latent_dim,
            a_range=args.a_range
        )

        model_params = {
            'latent_dim': args.latent_dim,
            'a_range': args.a_range,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'fold': args.fold
        }

        optimizer = torch.optim.Adam(model.mirt_net.parameters(), lr=args.lr)

    elif model_name == 'neuralcdm':
        from pycd.models.neuralcdm import NeuralCDM
    
        # ✅ 新增：安全地从 args 读取 a_dim
        a_dim = getattr(args, "a_dim", 0)
    
        model = NeuralCDM(
            n_concepts=concept_count,
            n_exercises=exercise_count,
            n_students=user_count,
            hidden_dims=(args.hidden_dims1, args.hidden_dims2),
            dropout=(args.dropout1, args.dropout2),
            a_dim=a_dim,   # ✅ 现在不会报错
        )

        model_params = {
            'hidden_dims': (args.hidden_dims1, args.hidden_dims2),
            'dropout': (args.dropout1, args.dropout2),
            'lr': args.lr,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'fold': args.fold
        }

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    elif model_name == 'kancd':
        from pycd.models.kancd import KaNCD
        a_dim = getattr(args, 'a_dim', 0)
        model = KaNCD(
            n_concepts=concept_count,
            n_exercises=exercise_count,
            n_students=user_count,
            emb_dim=args.emb_dim,
            mf_type=args.mf_type,
            hidden_dims=(args.hidden_dims1, args.hidden_dims2),
            dropout=(args.dropout1, args.dropout2),
            a_dim=a_dim,  # ← 加上
        )

        # model_params = {
        #     'hidden_dims': (args.hidden_dims1, args.hidden_dims2),
        #     'dropout': (args.dropout1, args.dropout2),
        #     'emb_dim': args.emb_dim,
        #     'mf_type': args.mf_type,
        #     'lr': args.lr,
        #     'batch_size': args.batch_size,
        #     'epochs': args.epochs,
        #     'fold': args.fold
        # }

        model_params = {
            'hidden_dims': (args.hidden_dims1, args.hidden_dims2),
            'dropout': (args.dropout1, args.dropout2),
            'emb_dim': args.emb_dim,
            'mf_type': args.mf_type,
            'a_dim': a_dim,  # ← 记录
            'lr': args.lr,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'fold': args.fold
        }

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    elif model_name == 'kscd':
        from pycd.models.kscd import KSCD
        a_dim = getattr(args, 'a_dim', 0)
        print('a_dim')
        print(a_dim)
        model = KSCD(
            n_students=user_count,
            n_exercises=exercise_count,
            n_concepts=concept_count,
            # emb_dim=64,
            emb_dim=args.emb_dim,
            a_dim=a_dim   # ✅ 新增参数,
            # disable_q=False
        )
        # a_dim = getattr(args, 'a_dim', 0)
        model_params = {
            'emb_dim': args.emb_dim,
            'a_dim': a_dim,  # ✅ 记录 a_dim
            'lr': args.lr,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'fold': args.fold
        }
    
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    elif model_name == 'rcd':  # Mingliang Hou, 2025-05-25, add the rcd model import.
        from pycd.models.rcd import RCD
        from data.graph_utils import construct_local_map,ensure_rcd_graph_files
        ensure_rcd_graph_files(args.dataset, args.base_dir)
        args.gpu = 0 
        local_map = construct_local_map(args.dataset, args.base_dir)

        model = RCD(args, local_map)

        model_params = {
            'knowledge_n': args.knowledge_n,
            'exer_n': args.exer_n,
            'student_n': args.student_n,
            'emb_dim': args.emb_dim,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'fold': args.fold
        }

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    elif model_name == 'icdm':
        from pycd.models.icdm import ICDM

        model = ICDM(stu_num=user_count, 
                    prob_num=exercise_count, 
                    know_num=concept_count,
                    dim=args.dim, 
                    device=args.device, 
                    gcn_layers=args.gcnlayers,
                    weight_reg=args.weight_reg,
                    graph=args.graph_dict, 
                    agg_type=args.agg_type, 
                    cdm_type=args.cdm_type,
                    khop=args.khop
                    )

        model_params = {
            'dim': args.dim,
            'lr': args.lr,
            'gcn_layers':args.gcnlayers,
            'weight_reg':args.weight_reg,
            'agg_type':args.agg_type,
            'cdm_type':args.cdm_type,
            'khop':args.khop,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'fold': args.fold
            
        }
 
        optimizer = None 

    elif model_name == 'scd':
        from pycd.models.scd import SymbolicCDM
        a_dim = getattr(args, 'a_dim', 0)
        model = SymbolicCDM(
                    q_matrix=args.q_matrix,
                    student_number=user_count, 
                    question_number=exercise_count, 
                    knowledge_number=concept_count,
                    train_size=args.train_size,
                    train_set=args.train_loader,
                    valid_set=args.valid_loader,
                    test_set=args.test_loader,
                    a_dim=a_dim
        )

        model_params = {
            "parameter_epochs": args.parameter_epochs, 
            "interaction_epochs": args.interaction_epochs, 
            "population_size": args.population_size, 
            "ngen": args.ngen, 
            "cxpb": args.cxpb, 
            "mutpb": args.mutpb,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'fold': args.fold,
            'a_dim': a_dim
        }
 
        optimizer = None 
    
    elif model_name == 'hypercdm':
        from pycd.models.hypercdm import HyperCDM

        model = HyperCDM(
            student_num=user_count,
            exercise_num=exercise_count,
            knowledge_num=concept_count,
            feature_dim=args.feature_dim,
            emb_dim=args.emb_dim,
            layers=args.layers,
            device=args.device
        )

        model_params = {
            'feature_dim': args.feature_dim,
            'emb_dim': args.emb_dim,
            'layers': args.layers,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'fold': args.fold
        }

        optimizer = None

    elif model_name == 'disengcd':
        from pycd.models.disengcd import DisenGCD
        model = DisenGCD(args=args)
        model_params = {
            'lr': args.lr,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'fold': args.fold
        }
        optimizer=[torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay = args.weight_decay

    ),
    torch.optim.Adam(model.FusionLayer1.alphas(), lr=args.alr)]

    elif model_name == 'orcdf':
        from pycd.models.orcdf import ORCDF

        model = ORCDF(
            student_num=user_count,
            exercise_num=exercise_count,
            knowledge_num=concept_count,
            latent_dim=args.latent_dim,
            hidden_dims=[args.hidden_dims1, args.hidden_dims2],
            ssl_temp=args.ssl_temp,
            ssl_weight=args.ssl_weight,
            flip_ratio=args.flip_ratio,
            gcn_layers=args.gcn_layers,
            keep_prob=args.keep_prob,
            dtype= args.dtype,
            if_type=args.if_type, #Interaction Function, here only use kancd
            device=args.device
        )

        model_params = {
            "latent_dim": args.latent_dim,
            "hidden_dims": [args.hidden_dims1, args.hidden_dims2],
            "ssl_temp": args.ssl_temp,
            "ssl_weight": args.ssl_weight,
            "flip_ratio": args.flip_ratio,
            'gcn_layers': args.gcn_layers,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'fold': args.fold
        }

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    else:
        raise ValueError(f"不支持的模型: {model_name}")

    return model, model_params, optimizer
