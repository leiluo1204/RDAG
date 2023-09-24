import os
import threading

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import IEMOCAPDataset
from model import *
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, \
    precision_recall_fscore_support
from trainer import train_or_eval_model, save_badcase
from dataset import IEMOCAPDataset
from dataloader import get_IEMOCAP_loaders
from dataloader import get_IEMOCAP_loaders_v2
from transformers import AdamW
import copy
import tkinter as tk
from tkinter import filedialog
from functools import partial
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# from tkintertable import TableCanvas,TableModel

# We use seed = 100 for reproduction of the results reported in the paper.
seed = 100


def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def switch_case(value):
    switcher = {
        0: "中立",
        1: "幸福",
        2: "悲伤",
        3: "愤怒",
        4: "惊讶",
        5: "恐惧",
        6: "厌恶",
    }
    return switcher.get(value, 'wrong value')


def switch_case_v2(value):
    switcher = {
        264: "A",
        265: "B",
        284: "C",
        285: "D",
    }
    return switcher.get(value, 'wrong value')


def evaluate(model, dataloader, cuda, args, speaker_vocab, label_vocab):
    preds, labels = [], []
    scores, vids = [], []
    dialogs = []
    speakers = []
    content = []
    utt = []

    model.eval()

    for data in dataloader:

        features, label, adj, s_mask, s_mask_onehot, lengths, speaker, utterances, spea = data
        if cuda:
            features = features.cuda()
            label = label.cuda()
            adj = adj.cuda()
            s_mask_onehot = s_mask_onehot.cuda()
            s_mask = s_mask.cuda()
            lengths = lengths.cuda()

        log_prob = model(features, adj, s_mask, s_mask_onehot, lengths)  # (B, N, C)

        label = label.cpu().numpy().tolist()  # (B, N)
        pred = torch.argmax(log_prob, dim=2).cpu().numpy().tolist()  # (B, N)
        preds += pred
        labels += label
        dialogs += utterances
        speakers += speaker

    if preds != []:
        new_preds = []
        new_labels = []
        for i, label in enumerate(labels):
            for j, l in enumerate(label):
                if l != -1:
                    # print(utterances[i][j], ':', switch_case(l))
                    content.append(preds[i][j])
                    new_labels.append(l)
                    new_preds.append(preds[i][j])
    else:
        return []
    coo = []
    spa = []
    for i in spea:
        for j in i:
            spa.append(switch_case_v2(j))
    for i in utterances:
        for j in i:
            utt.append(j)
    y = []
    for i in range(50):
        y.append(content[i])
        coo.append({'Speaker': spa[i], 'Utterance': utt[i], 'Content': switch_case(content[i])})
    return coo, y

    # avg_accuracy = round(accuracy_score(new_labels, new_preds) * 100, 2)
    # if args.dataset_name in ['IEMOCAP', 'MELD', 'EmoryNLP']:
    #     avg_fscore = round(f1_score(new_labels, new_preds, average='weighted') * 100, 2)
    #     print('test_accuracy', avg_accuracy)
    #     print('test_f1', avg_fscore)
    #     return
    # else:
    #     avg_micro_fscore = round(f1_score(new_labels, new_preds, average='micro', labels=list(range(1, 7))) * 100, 2)
    #     avg_macro_fscore = round(f1_score(new_labels, new_preds, average='macro') * 100, 2)
    #     print('test_accuracy', avg_accuracy)
    #     print('test_micro_f1', avg_micro_fscore)
    #     print('test_macro_f1', avg_macro_fscore)
    #     return


global results

if __name__ == '__main__':

    # path = './saved_models/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_model_dir', type=str, default='')
    parser.add_argument('--bert_tokenizer_dir', type=str, default='')

    parser.add_argument('--state_dict_file', type=str, default='./saved_models/IEMOCAP/model_29_68.87.pkl')

    parser.add_argument('--bert_dim', type=int, default=1024)
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--mlp_layers', type=int, default=2, help='Number of output mlp layers.')
    parser.add_argument('--gnn_layers', type=int, default=4, help='Number of gnn layers.')
    parser.add_argument('--emb_dim', type=int, default=1024, help='Feature size.')

    parser.add_argument('--attn_type', type=str, default='rgcn', choices=['dotprod', 'linear', 'bilinear', 'rgcn'],
                        help='Feature size.')
    parser.add_argument('--no_rel_attn', action='store_true', default=False, help='no relation for edges')

    parser.add_argument('--max_sent_len', type=int, default=200,
                        help='max content length for each text, if set to 0, then the max length has no constrain')

    parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--dataset_name', default='IEMOCAP', type=str,
                        help='dataset name, IEMOCAP or MELD or DailyDialog')

    parser.add_argument('--windowp', type=int, default=1,
                        help='context window size for constructing edges in graph model for past utterances')

    parser.add_argument('--windowf', type=int, default=0,
                        help='context window size for constructing edges in graph model for future utterances')

    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')

    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR', help='learning rate')

    parser.add_argument('--dropout', type=float, default=0.2, metavar='dropout', help='dropout rate')

    parser.add_argument('--batch_size', type=int, default=32, metavar='BS', help='batch size')

    parser.add_argument('--epochs', type=int, default=30, metavar='E', help='number of epochs')

    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    parser.add_argument('--nodal_att_type', type=str, default=None, choices=['global', 'past'],
                        help='type of nodal attention')

    args = parser.parse_args()
    print(args)

    seed_everything()

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    print(args.cuda)

    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')
    #
    # if args.tensorboard:
    #     from tensorboardX import SummaryWriter
    #
    #     writer = SummaryWriter()

    cuda = args.cuda
    n_epochs = args.epochs
    batch_size = args.batch_size

    test_loader, speaker_vocab, label_vocab, person_vec = get_IEMOCAP_loaders_v2(
        dataset_name=args.dataset_name, batch_size=batch_size, num_workers=0, args=args)
    n_classes = len(label_vocab['itos'])

    print('building model..')
    model = DAGERC_fushion(args, n_classes)

    if torch.cuda.device_count() > 1:
        print('Multi-GPU...........')
        model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    if cuda:
        model.cuda()

    state_dict = torch.load(args.state_dict_file)
    model.load_state_dict(state_dict)
    result, content = evaluate(model, test_loader, cuda, args, speaker_vocab, label_vocab)

    root = tk.Tk()
    root.title("对话情感分析")
    root.geometry("1920x1080")


    def display_result(result):
        result_text.delete(1.0, tk.END)  # Clear the text box

        # Display the data in the table format
        if result:
            df = pd.DataFrame(result)
            result_text.insert(tk.END, df.to_string(index=False))
            result_text.insert(tk.INSERT, '\n')

        else:
            result_text.insert(tk.END, "No data to display")


    def download_table():
        if result:
            file_path = filedialog.asksaveasfilename(defaultextension=".csv")
            if file_path:
                df = pd.DataFrame(result)
                df.to_csv(file_path, index=False)
                print("Table downloaded successfully")
        # 分析按钮


    def browse_file():
        global file_pat
        file_pat = filedialog.askopenfilename()
        dfs = pd.read_csv(file_pat)
        # textvar.set(dfs.to_string(index=False))
        result_text.insert(tk.END, dfs.to_string(index=False))


    def create_matplotlib(content):
        """创建绘图对象"""
        # 设置中文显示字体
        mpl.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
        mpl.rcParams['axes.unicode_minus'] = False  # 负号显示
        # 创建绘图对象f figsize的单位是英寸 像素 = 英寸*分辨率
        figure = plt.figure(num=2, figsize=(200, 100), dpi=80, facecolor="gold", edgecolor='green', frameon=True)
        # 创建一副子图
        fig1 = plt.subplot(1, 1, 1)  # 三个参数，依次是：行，列，当前索引
        # 创建数据源：x轴是等间距的一组数
        x = np.arange(0, 50, 1)
        y1 = content

        line1 = fig1.plot(x, y1, color='red', linewidth=2, label='y=情感', linestyle='--')  # 画第一条线

        fig1.set_title("分析图", loc='center', pad=20, fontsize='xx-large', color='red')  # 设置标题
        # line1.set_label("正弦曲线")  # 确定图例
        # 定义legend 重新定义了一次label
        fig1.legend(['情感极性'], loc='lower right', facecolor='orange', frameon=True, shadow=True,
                    framealpha=0.7)
        # ,fontsize='xx-large'
        fig1.set_xlabel('(x)')  # 确定坐标轴标题
        fig1.set_ylabel("(y)情感极性,中立, 幸福, 悲伤, 愤怒, 惊讶, 恐惧, 厌恶")
        fig1.set_yticks([0, 1, 2, 3, 4, 5, 6])  # 设置坐标轴刻度
        fig1.grid(which='major', axis='x', color='gray', linestyle='-', linewidth=0.5, alpha=0.2)
        label = tk.Label(text='分析结果')
        label.pack()
        # 创建画布
        canvas = FigureCanvasTkAgg(figure)
        canvas.draw()
        canvas.get_tk_widget().pack()


    # Set interface colors
    root.configure(bg='#FFF9C4')  # Window background color
    frame = tk.Frame(root, padx=40, pady=40, bg='#FFF9C4')  # Frame background color
    textbox = tk.Text(frame, height=10, width=50, bg='#D3D3D3')  # Textbox background color
    frame.pack()


    # comment_text = tk.Text(frame, height=20, width=200, font=("Arial", 12))
    # comment_text.grid(row=1, column=0, columnspan=3)

    def thread_it(func, *args):
        '''将函数打包进线程'''
        # 创建
        t = threading.Thread(target=func, args=args)
        # 守护 !!!
        t.setDaemon(True)
        # 启动
        t.start()
        # 阻塞--卡死界面！
        # t.join()


    predict_button = tk.Button(frame, text="分析", command=lambda: thread_it(display_result, result),
                               font=("Arial", 14))
    predict_button.grid(row=2, column=0, columnspan=3, pady=12, sticky='ns')

    mat_button = tk.Button(frame, text="分析图", command=lambda: thread_it(create_matplotlib, content),
                           font=("Arial", 14))
    mat_button.grid(row=4, column=1, padx=10, pady=10, sticky='sw')

    choose_file_button = tk.Button(frame, text="选择文件", command=lambda: thread_it(browse_file), font=("Arial", 14))
    choose_file_button.grid(row=0, column=2, padx=10, pady=10, sticky='se')

    result_text = tk.Text(frame, height=20, width=200, font=("Arial", 12))
    result_text.grid(row=3, column=0, columnspan=3)

    save_button = tk.Button(frame, text="保存结果", command=lambda: thread_it(download_table), font=("Arial", 14))
    save_button.grid(row=4, column=0, padx=10, pady=10, sticky='sw')

    quit_button = tk.Button(frame, text="退出系统", command=root.quit, font=("Arial", 14))
    quit_button.grid(row=4, column=2, padx=10, pady=10, sticky='se')

    # 运行界面主循环
    root.mainloop()
