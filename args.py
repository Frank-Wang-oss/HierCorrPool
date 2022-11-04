import argparse


def args():
    args = argparse.ArgumentParser()

    args.add_argument('--window_size',default=10,type=int)
    args.add_argument('--max_rul',default=125,type=int)
    args.add_argument('--time_denpen_len',default=8,type=int)
    args.add_argument('--time_denpen_inter',default=1,type=int)
    args.add_argument('--batch_size',default=50,type=int)
    args.add_argument('--data_sub',default=1,type=int)
    args.add_argument('--class_dimension',default=1,type=int)
    args.add_argument('--k',default=1,type=int)
    args.add_argument('--setting',default=False)
    args.add_argument('--epoch',default=1, type = int)
    args.add_argument('--show_interval',default=1, type = int)
    args.add_argument('--save_name',default=None, type = str)
    args.add_argument('--graph_processing',default='GCN', type = str)
    args.add_argument('--time_processing',default='1DCNN', type = str)
    args.add_argument('--GL',default=False, type = bool)
    args.add_argument('--fusion',default='cat', type = str) ## choices are 'cat', 'cat_prop'
    args.add_argument('--num_sensor_last',default=6, type = int) ## choices are 'cat', 'cat_prop'
    args.add_argument('--layers',default=2, type = int) ## choices are 'cat', 'cat_prop'
    args.add_argument('--drop_last',default=True, type = bool) ## choices are 'cat', 'cat_prop'
    args.add_argument('--num_class', default=6, type=int)
    args.add_argument('--num_nodes', default=6, type=int)
    args.add_argument('--fe_type', default='LSTM', type=str)
    args.add_argument('--task_type', default='forecasting', type=str)
    args.add_argument('--dataset', default='RUL', type=str)
    args.add_argument('--i_ISRUC', default=0, type=int)
    args.add_argument('--hidden_dimension', default=0, type=int)
    args.add_argument('--embedding_dimension', default=0, type=int)
    args.add_argument('--hidden_node1', default=0, type=int)
    args.add_argument('--hidden_node2', default=0, type=int)
    args.add_argument('--convo_time_length', default=0, type=int)

    return args.parse_args()
