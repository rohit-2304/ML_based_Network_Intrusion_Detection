from xgboost import XGBClassifier
import pandas as pd
import sys

print("Loading the Model......\n")

model = XGBClassifier()

try:
    model.load_model("xgb_model.json")
except:
    print("Failed to Load the model.")
    sys.exit()


print("Reading the network traffic flows.....\n")

try:
    flows = pd.read_csv("flows/dos_attack.pcap_Flow.csv")
except:
    print("Could not load the traffic flows.")

print("Preprocessing data......\n")
flows.columns = flows.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('/','_')
cols = [
    'dst_port',
    'flow_duration',
    'tot_fwd_pkts',
    'tot_bwd_pkts',
    'totlen_fwd_pkts',
    'totlen_bwd_pkts',
    'fwd_pkt_len_max',
    'fwd_pkt_len_min',
    'fwd_pkt_len_mean',
    'fwd_pkt_len_std',
    'bwd_pkt_len_max',
    'bwd_pkt_len_min', 
    'bwd_pkt_len_mean', 
    'bwd_pkt_len_std',
    'flow_byts_s',
    'flow_pkts_s',
    'flow_iat_mean', 
    'flow_iat_std',
    'flow_iat_max', 
    'flow_iat_min',
    'fwd_iat_tot',
    'fwd_iat_mean',
    'fwd_iat_std', 
    'fwd_iat_max', 
    'fwd_iat_min',
    'bwd_iat_tot',
    'bwd_iat_mean',
    'bwd_iat_std',
    'bwd_iat_max',
    'bwd_iat_min',    
    'fwd_psh_flags',
    'fwd_urg_flags',
    'bwd_urg_flags',
    'fwd_header_len',
    'bwd_header_len',
    'fwd_pkts_s',
    'bwd_pkts_s',
    'pkt_len_min',
    'pkt_len_max',
    'pkt_len_mean', 
    'pkt_len_std', 
    'pkt_len_var',
    'fin_flag_cnt',
    'syn_flag_cnt', 
    'rst_flag_cnt', 
    'psh_flag_cnt', 
    'ack_flag_cnt',
    'urg_flag_cnt',
    'cwe_flag_count', 
    'ece_flag_cnt',
    'down_up_ratio',
    'pkt_size_avg',
    'fwd_seg_size_avg',
    'bwd_seg_size_avg',
    'fwd_header_len',
    'fwd_byts_b_avg',
    'fwd_pkts_b_avg',
    'fwd_blk_rate_avg',
    'bwd_byts_b_avg',
    'bwd_pkts_b_avg',
    'bwd_blk_rate_avg',
    'subflow_fwd_pkts',
    'subflow_fwd_byts',
    'subflow_bwd_pkts', 
    'subflow_bwd_byts',
    'init_fwd_win_byts',
    'init_bwd_win_byts',
    'fwd_act_data_pkts',
    'fwd_seg_size_min',
    'active_mean',
    'active_std',
    'active_max', 
    'active_min',
    'idle_mean',
    'idle_std',
    'idle_max',
    'idle_min', 
]

attacks = {0:'BENIGN', 1:'Bot', 2:'Brute Force', 3:'DDos', 4:'DoS', 5:'Port Scan',6:'Web Attack'}

#preprocessing
X = flows[cols].copy()
X['flow_duration'] = X['flow_duration'].apply(lambda x: x*1e6)  

print("Predicting Intrusion.......\n")
preds = model.predict(X.values)

malicious = []

for i in range(len(preds)):
    if preds[i] != 0 :
        malicious.append(i)

if len(malicious) > 0:
    print("########################  ALERT ########################")
    print("Malicous Activity Detected\n")

    for flow in malicious:
        print(f"Flow {flow} : {attacks[preds[flow]]} Detected")
        print(f"Time : {flows.iloc[flow]['timestamp']}")
        print(f"Attacker : IP = {flows.iloc[flow]['src_ip']} , Port = {flows.iloc[flow]['src_port']}")
        print(f"Protocol : {flows.iloc[flow]['protocol']}")
        print("--------------------------------------------------------\n")
    print("Please take the necessary steps to prevent damage.")

