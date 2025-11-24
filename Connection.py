#contains both the Connection object and the methods for packaging the data
#gzip files are most efficient, but the lzma format is still implemented

import sys
import csv
from datetime import datetime
import ipaddress
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
import gzip
import lzma
import time
from threading import Thread


#defines a Connection data object with a unique attribute for each element
class Connection(object):
    def __init__(self, row):
        self.name= int(row[0])
        self.flow_id= row[1]
        self.source_ip= int(ipaddress.IPv4Address(row[2]))
        self.source_port= int(row[3])
        self.destination_ip= int(ipaddress.IPv4Address(row[4]))
        self.destination_port= int(row[5])
        self.protocol= int(row[6])
        self.timestamp= datetime.fromisoformat(row[7]).timestamp()
        self.flow_duration= int(row[8])
        self.ttl_fwd_pkts= int(row[9])
        self.ttl_bwd_pkts= int(row[10])
        self.ttl_len_of_fwd_pkts= int(float(row[11]))
        self.ttl_len_of_bwd_pkts= int(float(row[12]))
        self.fwd_pkt_len_max= int(float(row[13]))
        self.fwd_pkt_len_min= int(float(row[14]))
        self.fwd_pkt_len_mean= float(row[15])
        self.fwd_pkt_len_std= float(row[16])
        self.bwd_pkt_len_max= int(float(row[17]))
        self.bwd_pkt_len_min= int(float(row[18]))
        self.bwd_pkt_len_mean= float(row[19])
        self.bwd_pkt_len_std= float(row[20])
        try:
            self.flow_bytes_per_s= float(row[21])
        except ValueError:
            if (row[21]=='Infinity'):
                self.flow_bytes_per_s=float('inf');
            else:
                self.flow_bytes_per_s=0.0
        try:
            self.flow_pkts_per_s= float(row[22])
        except ValueError:
            if (row[22]=='Infinity'):
                self.flow_pkts_per_s=float('inf');
            else:
                self.flow_pkts_per_s=0.0
        self.flow_iat_mean= float(row[23])
        self.flow_iat_std= float(row[24])
        self.flow_iat_max= int(float(row[25]))
        self.flow_iat_min= int(float(row[26]))
        self.fwd_iat_ttl= int(float(row[27]))
        self.fwd_iat_mean= float(row[28])
        self.fwd_iat_std= float(row[29])
        self.fwd_iat_max= int(float(row[30]))
        self.fwd_iat_min= int(float(row[31]))
        self.bwd_iat_ttl= int(float(row[32]))
        self.bwd_iat_mean= float(row[33])
        self.bwd_iat_std= float(row[34])
        self.bwd_iat_max= int(float(row[35]))
        self.bwd_iat_min= int(float(row[36]))
        self.fwd_psh_flags= int(row[37])
        self.bwd_psh_flags= int(row[38])
        self.fwd_urg_flags= int(row[39])
        self.bwd_urg_flags= int(row[40])
        self.fwd_header_len= int(row[41])
        self.bwd_header_len= int(row[42])
        self.fwd_pkts_per_s= float(row[43])
        self.bwd_pkts_per_s= float(row[44])
        self.min_pkt_len= int(float(row[45]))
        self.max_pkt_len= int(float(row[46]))
        self.pkt_len_mean= float(row[47])
        self.pkt_len_std= float(row[48])
        self.pkt_len_variance= float(row[49])
        self.fin_flag_count= int(row[50])
        self.syn_flag_count= int(row[51])
        self.rst_flag_count= int(row[52])
        self.psh_flag_count= int(row[53])
        self.ack_flag_count= int(row[54])
        self.urg_flag_count= int(row[55])
        self.cwe_flag_count= int(row[56])
        self.ece_flag_count= int(row[57])
        self.down_to_up_ratio= float(row[58])
        self.average_pkt_size= float(row[59])
        self.avg_fwd_segment_size= float(row[60])
        self.avg_bwd_segment_size= float(row[61])
        self.fwd_header_len1= int(row[62])
        self.fwd_avg_bytes_per_bulk= float(row[63])
        self.fwd_avg_pkts_per_bulk= float(row[64])
        self.fwd_avg_bulk_rate= float(row[65])
        self.bwd_avg_bytes_per_bulk= float(row[66])
        self.bwd_avg_pkts_per_bulk= float(row[67])
        self.bwd_avg_bulk_rate= float(row[68])
        self.subflow_fwd_pkts= int(row[69])
        self.subflow_fwd_bytes= int(row[70])
        self.subflow_bwd_pkts= int(row[71])
        self.subflow_bwd_bytes= int(row[72])
        self.init_win_bytes_fwd= int(row[73])
        self.init_win_bytes_bwd= int(row[74])
        self.act_data_pkt_fwd= int(row[75])
        self.min_seg_size_fwd= int(row[76])
        self.active_mean= float(row[77])
        self.active_std= float(row[78])
        self.active_max= int(float(row[79]))
        self.active_min= int(float(row[80]))
        self.idle_mean= float(row[81])
        self.idle_std= float(row[82])
        self.idle_max= int(float(row[83]))
        self.idle_min= int(float(row[84]))
        self.simillarhttp= row[85]
        self.inbound= int(row[86])
        self.label= label_assign(row[87])
            
    #prints all the data in the object, mostly used for testing
    def __str__(self):    
        return '''Name: {}\nFlow ID: {}\nSource IP: {}\nSource Port: {}\nDestination IP: {}
Destination Port: {}\nProtocol: {}\nTimestamp: {}\nFlow Duration: {}
Total Fwd Packets: {}\nTotal Backward Packets: {}\nTotal Length of Fwd Packets: {}
Total Length of Bwd Packets: {}\nFwd Packet Length Max: {}\nFwd Packet Length Min: {}
Fwd Packet Length Mean: {}\nFwd Packet Length Std: {}\nBwd Packet Length Max: {}
Bwd Packet Length Min: {}\nBwd Packet Length Mean: {}\nBwd Packet Length Std: {}
Flow Bytes/s: {}\nFlow Packets/s: {}\nFlow IAT Mean: {}\nFlow IAT Std: {}
Flow IAT Max: {}\nFlow IAT Min: {}\nFwd IAT Total: {}\nFwd IAT Mean: {}
Fwd IAT Std: {}\nFwd IAT Max: {}\nFwd IAT Min: {}\nBwd IAT Total: {}
Bwd IAT Mean: {}\nBwd IAT Std: {}\nBwd IAT Max: {}\nBwd IAT Min: {}
Fwd PSH Flags: {}\nBwd PSH Flags: {}\nFwd URG Flags: {}\nBwd URG Flags: {}
Fwd Header Length: {}\nBwd Header Length: {}\nFwd Packets/s: {}\nBwd Packets/s: {}
Min Packet Length: {}\nMax Packet Length: {}\nPacket Length Mean: {}
Packet Length Std: {}\nPacket Length Variance: {}\nFIN Flag Count: {}
SYN Flag Count: {}\nRST Flag Count: {}\nPSH Flag Count: {}\nACK Flag Count: {}
URG Flag Count: {}\nCWE Flag Count: {}\nECE Flag Count: {}\nDown/Up Ratio: {}
Average Packet Size: {}\nAvg Fwd Segment Size: {}\nAvg Bwd Segment Size: {}
Fwd Header Length.1: {}\nFwd Avg Bytes/Bulk: {}\nFwd Avg Packets/Bulk: {}
Fwd Avg Bulk Rate: {}\nBwd Avg Bytes/Bulk: {}\nBwd Avg Packets/Bulk: {}
Bwd Avg Bulk Rate: {}\nSubflow Fwd Packets: {}\nSubflow Fwd Bytes: {}
Subflow Bwd Packets: {}\nSubflow Bwd Bytes: {}\nInit_Win_bytes_forward: {}
Init_Win_bytes_backward: {}\nact_data_pkt_fwd: {}\nmin_seg_size_forward: {}
Active Mean: {}\nActive Std: {}\nActive Max: {}\nActive Min: {}\nIdle Mean: {}
Idle Std: {}\nIdle Max: {}\nIdle Min: {}\nSimillarHTTP: {}\nInbound: {}\nLabel: {}'''.format(
        self.name, self.flow_id, self.source_ip, self.source_port, 
        self.destination_ip, self.destination_port, self.protocol, self.timestamp, 
        self.flow_duration, self.ttl_fwd_pkts, self.ttl_bwd_pkts, 
        self.ttl_len_of_fwd_pkts, self.ttl_len_of_bwd_pkts, self.fwd_pkt_len_max, 
        self.fwd_pkt_len_min, self.fwd_pkt_len_mean, self.fwd_pkt_len_std, 
        self.bwd_pkt_len_max, self.bwd_pkt_len_min, self.bwd_pkt_len_mean, 
        self.bwd_pkt_len_std, self.flow_bytes_per_s, self.flow_pkts_per_s, 
        self.flow_iat_mean, self.flow_iat_std, self.flow_iat_max, 
        self.flow_iat_min, self.fwd_iat_ttl, self.fwd_iat_mean, 
        self.fwd_iat_std, self.fwd_iat_max, self.fwd_iat_min, self.bwd_iat_ttl, 
        self.bwd_iat_mean, self.bwd_iat_std, self.bwd_iat_max, 
        self.bwd_iat_min, self.fwd_psh_flags, self.bwd_psh_flags, 
        self.fwd_urg_flags, self.bwd_urg_flags, self.fwd_header_len, 
        self.bwd_header_len, self.fwd_pkts_per_s, self.bwd_pkts_per_s, 
        self.min_pkt_len, self.max_pkt_len, self.pkt_len_mean, 
        self.pkt_len_std, self.pkt_len_variance, self.fin_flag_count, 
        self.syn_flag_count, self.rst_flag_count, self.psh_flag_count, 
        self.ack_flag_count, self.urg_flag_count, self.cwe_flag_count, 
        self.ece_flag_count, self.down_to_up_ratio, self.average_pkt_size, 
        self.avg_fwd_segment_size, self.avg_bwd_segment_size, 
        self.fwd_header_len1, self.fwd_avg_bytes_per_bulk, 
        self.fwd_avg_pkts_per_bulk, self.fwd_avg_bulk_rate, 
        self.bwd_avg_bytes_per_bulk, self.bwd_avg_pkts_per_bulk, 
        self.bwd_avg_bulk_rate, self.subflow_fwd_pkts, self.subflow_fwd_bytes, 
        self.subflow_bwd_pkts, self.subflow_bwd_bytes, self.init_win_bytes_fwd, 
        self.init_win_bytes_bwd, self.act_data_pkt_fwd, self.min_seg_size_fwd, 
        self.active_mean, self.active_std, self.active_max, self.active_min, 
        self.idle_mean, self.idle_std, self.idle_max, self.idle_min, 
        self.simillarhttp, self.inbound, label_string(self.label))

#same data typing as Conneciton Object, but stored as a list
def formatter(row):
    row[0]= int(row[0])
    row[1]= row[1]
    #makes an ip address object and returns the 32 bit integer value
    row[2]= int(ipaddress.IPv4Address(row[2]))
    row[3]= int(row[3])
    #makes an ip address object and returns the 32 bit integer value
    row[4]= int(ipaddress.IPv4Address(row[4]))
    row[5]= int(row[5])
    row[6]= int(row[6])
    #creates a date time object from the string and turns it into a posix/epoch timestamp
    row[7]= datetime.fromisoformat(row[7]).timestamp()
    row[8]= int(row[8])
    row[9]= int(row[9])
    row[10]= int(row[10])
    row[11]= int(float(row[11]))
    row[12]= int(float(row[12]))
    row[13]= int(float(row[13]))
    row[14]= int(float(row[14]))
    row[15]= float(row[15])
    row[16]= float(row[16])
    row[17]= int(float(row[17]))
    row[18]= int(float(row[18]))
    row[19]= float(row[19])
    row[20]= float(row[20])
    #these two values were tricky because they could be null or 'Infinity'
    try:
        row[21]= float(row[21])
    except ValueError:
        if (row[21]=='Infinity'):
            row[21]=float('inf');
        else:
            row[21]=0.0
    try:
        row[22]= float(row[22])
    except ValueError:
        if (row[22]=='Infinity'):
            row[22]=float('inf');
        else:
            row[22]=0.0
    row[23]= float(row[23])
    row[24]= float(row[24])
    row[25]= int(float(row[25]))
    row[26]= int(float(row[26]))
    row[27]= int(float(row[27]))
    row[28]= float(row[28])
    row[29]= float(row[29])
    row[30]= int(float(row[30]))
    row[31]= int(float(row[31]))
    row[32]= int(float(row[32]))
    row[33]= float(row[33])
    row[34]= float(row[34])
    row[35]= int(float(row[35]))
    row[36]= int(float(row[36]))
    row[37]= int(row[37])
    row[38]= int(row[38])
    row[39]= int(row[39])
    row[40]= int(row[40])
    row[41]= int(row[41])
    row[42]= int(row[42])
    row[43]= float(row[43])
    row[44]= float(row[44])
    row[45]= int(float(row[45]))
    row[46]= int(float(row[46]))
    row[47]= float(row[47])
    row[48]= float(row[48])
    row[49]= float(row[49])
    row[50]= int(row[50])
    row[51]= int(row[51])
    row[52]= int(row[52])
    row[53]= int(row[53])
    row[54]= int(row[54])
    row[55]= int(row[55])
    row[56]= int(row[56])
    row[57]= int(row[57])
    row[58]= float(row[58])
    row[59]= float(row[59])
    row[60]= float(row[60])
    row[61]= float(row[61])
    row[62]= int(row[62])
    row[63]= float(row[63])
    row[64]= float(row[64])
    row[65]= float(row[65])
    row[66]= float(row[66])
    row[67]= float(row[67])
    row[68]= float(row[68])
    row[69]= int(row[69])
    row[70]= int(row[70])
    row[71]= int(row[71])
    row[72]= int(row[72])
    row[73]= int(row[73])
    row[74]= int(row[74])
    row[75]= int(row[75])
    row[76]= int(row[76])
    row[77]= float(row[77])
    row[78]= float(row[78])
    row[79]= int(float(row[79]))
    row[80]= int(float(row[80]))
    row[81]= float(row[81])
    row[82]= float(row[82])
    row[83]= int(float(row[83]))
    row[84]= int(float(row[84]))
    row[85]= row[85]
    row[86]= int(row[86])
    #uses a public method to convert a string into a integer label 0-11
    row[87]= label_assign(row[87])
    return row
#setting the labels as ints so the model doesnt have to parse strings
def label_assign(label):
    label=label.lower()
    match (label):
        case("benign"):
            return 0
        case("drdos_dns"):
            return 1
        case("drdos_ldap"):
            return 2
        case("drdos_mssql"):
            return 3
        case("drdos_netbios"):
            return 4
        case("drdos_ntp"):
            return 5
        case("drdos_snmp"):
            return 6
        case("drdos_ssdp"):
            return 7
        case("drdos_udp"):
            return 8
        case("syn"):
            return 9
        case("tftp"):
            return 10
        case("udp-lag"):
            return 11
def label_string(num):
    match (num):
        case(0):
            return "benign"
        case(1):
            return "drdos_dns"
        case(2):
            return "drdos_ldap"
        case(3):
            return "drdos_mssql"
        case(4):
            return "drdos_netbios"
        case(5):
            return "drdos_ntp"
        case(6):
            return "drdos_snmp"
        case(7):
            return "drdos_ssdp"
        case(8):
            return "drdos_udp"
        case(9):
            return "syn"
        case(10):
            return "tftp"
        case(11):
            return "udp-lag"

#returns an index for a value in a formatted list of when given a name of a field
def item_by_name(key):
    key=key.lower()
    match key:
        case "name":
           return 0
        case "flow id" | "flow_id":
            return 1
        case "source ip" | "source_ip":
            return 2
        case "source port" | "source_port":
            return 3
        case "destination ip" | "destination_ip":
            return 4
        case "destination port" | "destination_port":
            return 5
        case "protocol":
            return 6
        case "timestamp":
            return 7
        case "flow duration" | "flow_duration":
            return 8
        case "total fwd packets" | "ttl_fwd_pkts":
            return 9
        case "total backward packets" | "ttl_bwd_pkts":
            return 10
        case "total length of fwd packets" | "ttl_len_of_fwd_pkts":
            return 11
        case "total length of bwd packets" | "ttl_len_of_bwd_pkts":
            return 12
        case "fwd packet length max" | "fwd_pkt_len_max":
            return 13
        case "fwd packet length min" | "fwd_pkt_len_min":
            return 14
        case "fwd packet length mean" | "fwd_pkt_len_mean":
            return 15
        case "fwd packet length std" | "fwd_pkt_len_std":
            return 16
        case "bwd packet length max" | "bwd_pkt_len_max":
            return 17
        case "bwd packet length min" | "bwd_pkt_len_min":
            return 18
        case "bwd packet length mean" | "bwd_pkt_len_mean":
            return 19
        case "bwd packet length std" | "bwd_pkt_len_std":
            return 20
        case "flow bytes/s" | "flow_bytes_per_s":
            return 21
        case "flow packets/s" | "flow_pkts_per_s":
            return 22
        case "flow iat mean" | "flow_iat_mean":
            return 23
        case "flow iat std" | "flow_iat_std":
            return 24
        case "flow iat max" | "flow_iat_max":
            return 25
        case "flow iat min" | "flow_iat_min":
            return 26
        case "fwd iat total" | "fwd_iat_ttl":
            return 27
        case "fwd iat mean" | "fwd_iat_mean":
            return 28
        case "fwd iat std" | "fwd_iat_std":
            return 29
        case "fwd iat max" | "fwd_iat_max":
            return 30
        case "fwd iat min" | "fwd_iat_min":
            return 31
        case "bwd iat total" | "bwd_iat_ttl":
            return 32
        case "bwd iat mean" | "bwd_iat_mean":
            return 33
        case "bwd iat std" | "bwd_iat_std":
            return 34
        case "bwd iat max" | "bwd_iat_max":
            return 35
        case "bwd iat min" | "bwd_iat_min":
            return 36
        case "fwd psh flags" | "fwd_psh_flags":
            return 37
        case "bwd psh flags" | "bwd_psh_flags":
            return 38
        case "fwd urg flags" | "fwd_urg_flags":
            return 39
        case "bwd urg flags" | "bwd_urg_flags":
            return 40
        case "fwd header length" | "fwd_header_len":
            return 41
        case "bwd header length" | "bwd_header_len":
            return 42
        case "fwd packets/s" | "fwd_pkts_per_s":
            return 43
        case "bwd packets/s" | "bwd_pkts_per_s":
            return 44
        case "min packet length" | "min_pkt_len":
            return 45
        case "max packet length" | "max_pkt_len":
            return 46
        case "packet length mean" | "pkt_len_mean":
            return 47
        case "packet length std" | "pkt_len_std":
            return 48
        case "packet length variance" | "pkt_len_variance":
            return 49
        case "fin flag count" | "fin_flag_count":
            return 50
        case "syn flag count" | "syn_flag_count":
            return 51
        case "rst flag count" | "rst_flag_count":
            return 52
        case "psh flag count" | "psh_flag_count":
            return 53
        case "ack flag count" | "ack_flag_count":
            return 54
        case "urg flag count" | "urg_flag_count":
            return 55
        case "cwe flag count" | "cwe_flag_count":
            return 56
        case "ece flag count" | "ece_flag_count":
            return 57
        case "down/up ratio" | "down_to_up_ratio":
            return 58
        case "average packet size" | "average_pkt_size":
            return 59
        case "avg fwd segment size" | "avg_fwd_segment_size":
            return 60
        case "avg bwd segment size" | "avg_bwd_segment_size":
            return 61
        case "fwd header length.1" | "fwd_header_len.1":
            return 62
        case "fwd avg bytes/bulk" | "fwd_avg_bytes_per_bulk":
            return 63
        case "fwd avg packets/bulk" | "fwd_avg_pkts_per_bulk":
            return 64
        case "fwd avg bulk rate" | "fwd_avg_bulk_rate":
            return 65
        case "bwd avg bytes/bulk" | "bwd_avg_bytes_per_bulk":
            return 66
        case "bwd avg packets/bulk" | "bwd_avg_pkts_per_bulk":
            return 67
        case "bwd avg bulk rate" | "bwd_avg_bulk_rate":
            return 68
        case "subflow fwd packets" | "subflow_fwd_pkts":
            return 69
        case "subflow fwd bytes" | "subflow_fwd_bytes":
            return 70
        case "subflow bwd packets" | "subflow_bwd_pkts":
            return 71
        case "subflow bwd bytes" | "subflow_bwd_bytes":
            return 72
        case "init_win_bytes_forward" | "init_win_bytes_fwd":
            return 73
        case "init_win_bytes_backward" | "init_win_bytes_bwd":
            return 74
        case "act_data_pkt_fwd":
            return 75
        case "min_seg_size_forward" | "min_seg_size_fwd":
            return 76
        case "active mean" | "active_mean":
            return 77
        case "active std" | "active_std":
            return 78
        case "active max" | "active_max":
            return 79
        case "active min" | "active_min":
            return 80
        case "idle mean" | "idle_mean":
            return 81
        case "idle std" | "idle_std":
            return 82
        case "idle max" | "idle_max":
            return 83
        case "idle min" | "idle_min":
            return 84
        case "simillarhttp":
            return 85
        case "inbound":
            return 86
        case "label":
            return 87
        case _:
            print('Incorrect key usage, returned -1')
            return -1
            
#method for converting a csv file to a given file type
#to_read is the csv file, output is where the file is saved
#key determines if it generates lists or Connection objects
def pickle_maker(to_read, output, key):
    start_time = time.time()
    pre_pickle=[]
    
    #this segment automatically turns a provided filename into a csv or gzip
    to_read = to_read.split('.')[0] + '.csv'
    output = output.split('.')[0] + '.gz'
    
    print('File successfully opened! ' + to_read +'\n', end='')
    
    if (key=='object'):
        with open(to_read) as csvfile:
            cucumber = csv.reader(csvfile)
            cucumber.__next__()
            for row in cucumber:
                if(row[0]!=''):
                    pre_pickle.append(Connection(row))
        print("Listified " + to_read + " as " + str(len(pre_pickle)) + " objects in" + "--- %s seconds ---" % (time.time() - start_time))
    else:
        with open(to_read) as csvfile:
            cucumber = csv.reader(csvfile)
            cucumber.__next__()
            for row in cucumber:
                if(row[0]!=''):
                    pre_pickle.append(formatter(row))
        print("Listified " + to_read + " as " + str(len(pre_pickle)) + " lists in" + "--- %s seconds ---" % (time.time() - start_time))
    save_as(pre_pickle, output)
        
#saves a list and displays the time it takes to compress/save the file
def save_as(to_save, output):
    if(output.split('.')[1]=='gz'):
        start_time = time.time()
        with gzip.open(output, "wb") as f:
            print('File successfully opened! ' + output +'\n', end='')
            pickle.dump(to_save, f)
        print(output + " saved in" + "--- %s seconds ---" % (time.time() - start_time))
    elif (output.split('.')[1]=='lzma'):
        start_time = time.time()
        with lzma.open(output, "wb") as f:
            print('File successfully opened! ' + output +'\n', end='')
            pickle.dump(to_save, f)
        print(output + " saved in" + "--- %s seconds ---" % (time.time() - start_time))
    else:
        start_time = time.time()        
        with open(output, 'wb') as f:
            print('File successfully opened! ' + output +'\n', end='')
            pickle.dump(to_save, f)
        print(output + " saved in" + "--- %s seconds ---" % (time.time() - start_time))

#takes a file generated by this module and returns it as a list, auto detects lzma and gzip types
def load_pickle(filename):
    if(filename.split('.')[1]=='gz'):
        with gzip.open(filename, "rb") as f:
            return pickle.load(f)
    elif (output.split('.')[1]=='xz'):
        with lzma.open(filename, "rb") as f:
            return pickle.load(f)
    else:     
        with open(filename, 'rb') as f:
            return pickle.load(f)
                  
#takes a list of formatted data, a field of type str or int, and a search term
#and makes a new list of rows which 
def get_match_rows(big_data, field, search_term):
    start_time = time.time()
    new_list=[]
    
    if(isinstance(field, str)):
        field=item_by_name(field)
        
    if (field<0 or field>=88):
        print('Invalid field: ' + str(field))
        return None
    for i in range(len(big_data)):
        row=big_data[i]
        if (row[field]==search_term):
            new_list.append(row)
    print("New list of %s elements made in--- %s seconds ---" % (len(new_list), time.time() - start_time))

    return new_list

#iterates over every element and returns a list of boolean values based on if the entire column contains certain relevant data
def check_col_relevance(big_data):
    start_time = time.time()
    results=[True]*88
    base_case=big_data[0]
    for i in range(1,len(big_data)):
        row=big_data[i]
        for j in range(len(row)):
            if (results[j]):
                results[j] = (row[j]==0 or row[j]==0)
                
    print("Columns checked in--- %s seconds ---" % (time.time() - start_time))

    return results


#saves a static list of csv files as a gzip compressed pickle containing 1 list of all the relevant rows
def save_all():
    start_time = time.time()
    files = ['DrDoS_UDP','DrDos_DNS', 'DrDoS_LDAP', 'DrDoS_MSSQL', 'DrDoS_NetBIOS',
             'DrDoS_NTP', 'DrDoS_SNMP', 'DrDoS_SSDP', 'Syn', 'UDPLag', 'TFTP']
    
    threads = [None] * len(files)
    
    for i in range(len(threads)):
        pickle_maker(files[i], files[i], 'list')
 
    my_time = (time.time() - start_time)    
    print("11 files pickled in" + "--- %s seconds --- (avg: %s)" % (my_time, my_time/11 ))

#just used to check load times
def load_check():
    files = ['DrDoS_UDP','DrDos_DNS', 'DrDoS_LDAP', 'DrDoS_MSSQL', 'DrDoS_NetBIOS',
             'DrDoS_NTP', 'DrDoS_SNMP', 'DrDoS_SSDP', 'Syn', 'UDPLag', 'TFTP']
    start_time=time.time()
    for i in range(len(files)):
        load_time = time.time()
        pick = load_pickle(files[i]+ '.gz')
        print(files[i] +" opened in" + "--- %s seconds ---" % (time.time() - load_time))
    my_time = (time.time() - start_time)
    print("11 files loaded in" + "--- %s seconds --- (avg: %s)" % (my_time, my_time/11 ))

def main():
    files = ['DrDoS_UDP','DrDos_DNS', 'DrDoS_LDAP', 'DrDoS_MSSQL', 'DrDoS_NetBIOS',
             'DrDoS_NTP', 'DrDoS_SNMP', 'DrDoS_SSDP', 'Syn', 'UDPLag', 'TFTP']
    with open('binary_check.csv', 'w', newline='') as csvfile:
        binarywriter = csv.writer(csvfile, delimiter=' ',
                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(files)):
            binarywriter.writerow(check_col_relevance(load_pickle(files[i]+ '.gz')))

if __name__ == '__main__':
    sys.exit(main())
