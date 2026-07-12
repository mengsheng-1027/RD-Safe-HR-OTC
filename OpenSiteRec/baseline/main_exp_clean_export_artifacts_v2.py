import numpy as np
import pandas as pd
import torch
import argparse
import time
from data_utils import OpenSiteRec, split
from eval_utils import PrecisionRecall_atK, NDCG_atK, get_label
from model import VanillaMF, NeuMF, RankNet, BasicCTRModel, WideDeep, DeepFM, xDeepFM, NGCF, LightGCN


MODEL = {'VanillaMF': VanillaMF, 'NeuMF': NeuMF, 'RankNet': RankNet,
         'DNN': BasicCTRModel, 'WideDeep': WideDeep, 'DeepFM': DeepFM, 'xDeepFM': xDeepFM,
         'NGCF': NGCF, 'LightGCN': LightGCN}


def parse_args():
    config_args = {
        'lr': 0.001,
        'dropout': 0.3,
        'cuda': -1,
        'epochs': 300,
        'weight_decay': 1e-4,
        'seed': 42,
        'model': 'LightGCN',
        'dim': 100,
        'city': 'Tokyo',
        'threshold': 5,
        'topk': [20],
        'patience': 5,
        'eval_freq': 10,
        'lr_reduce_freq': 10,
        'batch_size': 128,
        'save': 0,
    }

    parser = argparse.ArgumentParser()
    for param, val in config_args.items():
        if isinstance(val, bool):
            parser.add_argument(f"--{param}", type=int, default=val)
        elif isinstance(val, int):
            parser.add_argument(f"--{param}", type=int, default=val)
        elif isinstance(val, float):
            parser.add_argument(f"--{param}", type=float, default=val)
        elif isinstance(val, list):
            parser.add_argument(f"--{param}", nargs="+", type=int, default=val)
        else:
            parser.add_argument(f"--{param}", type=str, default=val)
    args = parser.parse_args()
    return args


args = parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'

# EXP_CLEAN_PATCH_V2: disabled automatic split regeneration. Frozen split files are used.
# split(args.city, args.threshold)
dataset = OpenSiteRec(args)
print(dataset.testDataSize)
args.user_num, args.item_num, args.cate_num = dataset.n_user, dataset.m_item, dataset.k_cate
args.Graph = dataset.Graph
model = MODEL[args.model](args)
print(str(model))
if args.cuda is not None and int(args.cuda) >= 0:
    model = model.to(args.device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
tot_params = sum([np.prod(p.size()) for p in model.parameters()])
print(f'Total number of parameters: {tot_params}')


def train():
    model.train()
    dataset.init_batches()
    batch_num = dataset.n_user // args.batch_size + 1
    avg_loss = []
    for i in range(batch_num):
        indices = torch.arange(i * args.batch_size, (i + 1) * args.batch_size) \
            if (i + 1) * args.batch_size <= dataset.n_user \
            else torch.arange(i * args.batch_size, dataset.n_user)
        users, labels = torch.LongTensor(dataset.U[indices]).to(args.device), \
                        torch.FloatTensor(dataset.bI[indices]).to(args.device)

        ratings = model(users)
        loss = model.loss_func(ratings, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())


def train_graph():
    model.train()
    model.mode = 'train'
    dataset.uniform_sampling()
    batch_num = dataset.trainDataSize // args.batch_size + 1
    avg_loss = []
    for i in range(batch_num):
        indices = torch.arange(i * args.batch_size, (i + 1) * args.batch_size) \
            if (i + 1) * args.batch_size <= dataset.trainDataSize \
            else torch.arange(i * args.batch_size, dataset.trainDataSize)
        batch = dataset.S[indices]
        users, pos_items, neg_items = torch.LongTensor(batch[:, 0]).to(args.device), \
                                      torch.LongTensor(batch[:, 1]).to(args.device), \
                                      torch.LongTensor(batch[:, 2]).to(args.device)

        loss, reg_loss = model.bpr_loss(users, pos_items, neg_items)
        loss = loss + args.weight_decay * reg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())


def train_CTR():
    model.train()
    dataset.init_batches()
    batch_num = dataset.n_user // args.batch_size + 1
    avg_loss = []
    for i in range(batch_num):
        indices = torch.arange(i * args.batch_size, (i + 1) * args.batch_size) \
            if (i + 1) * args.batch_size <= dataset.n_user \
            else torch.arange(i * args.batch_size, dataset.n_user)
        instances = {'Brand_ID': torch.LongTensor(dataset.U[indices]).to(args.device),
                     'Cate1_ID': torch.LongTensor(dataset.bF[indices][:, 0]).to(args.device),
                     'Cate2_ID': torch.LongTensor(dataset.bF[indices][:, 1]).to(args.device),
                     'Cate3_ID': torch.LongTensor(dataset.bF[indices][:, 2]).to(args.device)}
        labels = torch.FloatTensor(dataset.bI[indices]).to(args.device)

        ratings = model(instances)
        loss = model.loss_func(ratings, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())


def test():
    global best_rec, best_ndcg, best_state_dict, best_epoch, best_rec_at_best_ndcg, current_epoch
    model.eval()
    if args.model in ['RankNet', 'NGCF', 'LightGCN']:
        model.mode = 'test'
    testDict = dataset.testDict
    all_pos = dataset.allPos
    rec, ndcg = 0., 0.
    with torch.no_grad():
        users = list(testDict.keys())
        items = [testDict[u] for u in users]
        batch_num = len(users) // args.batch_size + 1
        for i in range(batch_num):
            batch_users = users[i * args.batch_size: (i + 1) * args.batch_size] \
                if (i + 1) * args.batch_size <= len(users) else users[i * args.batch_size:]
            # batch_pos = [all_pos[u] for u in batch_users]
            # batch_items = [[it for it in items[u] if it not in all_pos[u]] for u in batch_users]
            batch_items = [items[u] for u in batch_users]
            if args.model in ['DNN', 'WideDeep', 'DeepFM', 'xDeepFM']:
                instances = {'Brand_ID': torch.LongTensor(dataset.U[batch_users]).to(args.device),
                         'Cate1_ID': torch.LongTensor(dataset.F[batch_users][:, 0]).to(args.device),
                         'Cate2_ID': torch.LongTensor(dataset.F[batch_users][:, 1]).to(args.device),
                         'Cate3_ID': torch.LongTensor(dataset.F[batch_users][:, 2]).to(args.device)}
            else:
                instances = torch.LongTensor(batch_users).to(args.device)

            ratings = model(instances)
            # ratings = ratings * dataset.lt_mask

            # exclude_index = []
            # exclude_items = []
            # for range_i, its in enumerate(batch_pos):
            #     exclude_index.extend([range_i] * len(its))
            #     exclude_items.extend(its)
            # ratings[exclude_index, exclude_items] = -(1 << 10)
            _, ratings_K = torch.topk(ratings, k=args.topk[-1])
            ratings_K = ratings_K.cpu().numpy()

            r = get_label(batch_items, ratings_K)
            for k in args.topk:
                _, batch_rec = PrecisionRecall_atK(batch_items, r, k)
                batch_ndcg = NDCG_atK(batch_items, r, k)
                rec += batch_rec * len(batch_users)
                ndcg += batch_ndcg * len(batch_users)

        rec /= len(users)
        ndcg /= len(users)
        if best_rec < rec:
            best_rec = rec
        if best_ndcg < ndcg:
            best_ndcg = ndcg
            best_rec_at_best_ndcg = rec
            best_epoch = int(current_epoch)
            best_state_dict = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        print(f'Recall@{k}: {rec}\nnDCG@{k}: {ndcg}')





def exp_clean_export_artifacts_v2():
    """Export best-nDCG backbone artifacts for OTC reproduction.

    This function restores the best-nDCG checkpoint captured during official
    evaluation before exporting:
      - local score matrix: [num_brands, num_regions]
      - brand embeddings
      - region embeddings
      - torch checkpoint
      - JSON metadata
    """
    import json as _json
    from pathlib import Path as _Path

    project_root = _Path(__file__).resolve().parents[2]
    ckpt_dir = project_root / "exp_clean" / "checkpoints" / "baselines"
    score_dir = project_root / "exp_clean" / "scores" / "local_scores"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)

    # Restore the selected checkpoint before export. This is the key v2 fix.
    restored_best_state = False
    if 'best_state_dict' in globals() and best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        restored_best_state = True

    tag = f"{args.city}_{args.model}_seed{args.seed}_epochs{args.epochs}_dim{args.dim}"
    score_path = score_dir / f"{tag}_local_scores.npy"
    user_emb_path = ckpt_dir / f"{tag}_brand_embeddings.npy"
    item_emb_path = ckpt_dir / f"{tag}_region_embeddings.npy"
    ckpt_path = ckpt_dir / f"{tag}_checkpoint.pt"
    meta_path = ckpt_dir / f"{tag}_metadata.json"

    model.eval()
    if args.model in ['RankNet', 'NGCF', 'LightGCN']:
        model.mode = 'test'

    # Export local score matrix for all brands and all regions.
    scores = []
    with torch.no_grad():
        all_users = list(range(dataset.n_user))
        batch_num = len(all_users) // args.batch_size + 1
        for i in range(batch_num):
            batch_users = all_users[i * args.batch_size: (i + 1) * args.batch_size]
            if len(batch_users) == 0:
                continue
            if args.model in ['DNN', 'WideDeep', 'DeepFM', 'xDeepFM']:
                instances = {
                    'Brand_ID': torch.LongTensor(dataset.U[batch_users]).to(args.device),
                    'Cate1_ID': torch.LongTensor(dataset.F[batch_users][:, 0]).to(args.device),
                    'Cate2_ID': torch.LongTensor(dataset.F[batch_users][:, 1]).to(args.device),
                    'Cate3_ID': torch.LongTensor(dataset.F[batch_users][:, 2]).to(args.device),
                }
            else:
                instances = torch.LongTensor(batch_users).to(args.device)
            ratings = model(instances)
            scores.append(ratings.detach().cpu().float().numpy())

    local_scores = np.vstack(scores).astype(np.float32)
    np.save(score_path, local_scores)

    # Export embeddings. For graph models, use propagated embeddings in test mode.
    with torch.no_grad():
        if args.model == 'LightGCN':
            brand_emb, region_emb = model._LightGCN__message_passing()
        elif args.model == 'NGCF':
            brand_emb, region_emb = model._NGCF__message_passing()
        else:
            brand_emb = model.user_embedding.weight
            region_emb = model.item_embedding.weight
        brand_emb = brand_emb.detach().cpu().float().numpy().astype(np.float32)
        region_emb = region_emb.detach().cpu().float().numpy().astype(np.float32)

    np.save(user_emb_path, brand_emb)
    np.save(item_emb_path, region_emb)

    torch.save({
        'model_state_dict': model.state_dict(),
        'city': args.city,
        'model': args.model,
        'seed': args.seed,
        'epochs': args.epochs,
        'dim': args.dim,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'checkpoint_selection': 'best_ndcg',
        'patch_version': 'best_ndcg_export_v2_2026_06_08_a',
        'restored_best_state_before_export': bool(restored_best_state),
        'selected_epoch': int(best_epoch) if 'best_epoch' in globals() else -1,
        'best_recall_at_20': float(best_rec),
        'best_ndcg_at_20': float(best_ndcg),
        'best_rec_at_best_ndcg': float(best_rec_at_best_ndcg) if 'best_rec_at_best_ndcg' in globals() else float(best_rec),
    }, ckpt_path)

    meta = {
        'city': args.city,
        'model': args.model,
        'seed': int(args.seed),
        'epochs': int(args.epochs),
        'eval_freq': int(args.eval_freq),
        'batch_size': int(args.batch_size),
        'dim': int(args.dim),
        'lr': float(args.lr),
        'weight_decay': float(args.weight_decay),
        'checkpoint_selection': 'best_ndcg',
        'patch_version': 'best_ndcg_export_v2_2026_06_08_a',
        'restored_best_state_before_export': bool(restored_best_state),
        'selected_epoch': int(best_epoch) if 'best_epoch' in globals() else -1,
        'num_brands': int(dataset.n_user),
        'num_regions': int(dataset.m_item),
        'local_scores_shape': list(local_scores.shape),
        'brand_embeddings_shape': list(brand_emb.shape),
        'region_embeddings_shape': list(region_emb.shape),
        'best_recall_at_20': float(best_rec),
        'best_ndcg_at_20': float(best_ndcg),
        'best_rec_at_best_ndcg': float(best_rec_at_best_ndcg) if 'best_rec_at_best_ndcg' in globals() else float(best_rec),
        'score_path': str(score_path),
        'brand_embedding_path': str(user_emb_path),
        'region_embedding_path': str(item_emb_path),
        'checkpoint_path': str(ckpt_path),
        'metadata_path': str(meta_path),
        'note': 'Exported by main_exp_clean_export_artifacts_v2.py. Official main.py was not modified.'
    }
    meta_path.write_text(_json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print('EXP_CLEAN_ARTIFACT_JSON=' + _json.dumps(meta, ensure_ascii=False))


t_total = time.time()
best_rec, best_ndcg = 0., 0.
best_state_dict = None
best_epoch = -1
best_rec_at_best_ndcg = 0.
current_epoch = -1
for epoch in range(args.epochs):
    if args.model in ['RankNet', 'NGCF', 'LightGCN']:
        train_graph()
    elif args.model in ['DNN', 'WideDeep', 'DeepFM', 'xDeepFM']:
        train_CTR()
    else:
        train()
    torch.cuda.empty_cache()
    if (epoch + 1) % args.eval_freq == 0:
        current_epoch = epoch
        print(f'Epoch {epoch}')
        test()
        torch.cuda.empty_cache()

print(f'Best Results: \nRecall@{args.topk[-1]}: {round(best_rec, 4)}\nnDCG@{args.topk[-1]}: {round(best_ndcg, 4)}')

if int(getattr(args, 'save', 0)) == 1:
    exp_clean_export_artifacts_v2()

