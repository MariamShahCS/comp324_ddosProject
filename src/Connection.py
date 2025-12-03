#contains both the Connection object and the methods for packaging the data
#gzip files are most efficient, but the lzma format is still implemented

import sys
import csv
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
import gzip
import lzma
import time


#DECEMBER CHANGES
#
#
#added labels for the 2017 data set
#added a formatter2017 method
#picklemaker now saves as the default name, no more _74len added at the end
#picklemaker now has a parameter, format_key, to determine which formatter method to use
#picklmaker automatically saves TFTP as 4 files
#
#methods for testing & list manipulation not included, available upon request


#same data typing as Conneciton Object, but stored as a list
def formatter(row):
    new_row = [int(row[5]), int(row[8]), int(row[9]), int(row[10]), int(float(row[11])), int(float(row[12])), int(float(row[13])),
               int(float(row[14])), float(row[15]), float(row[16]), int(float(row[17])), int(float(row[18])), float(row[19]), float(row[20])]
    
    #these two values were tricky because they could be null or 'Infinity'
    try:
        new_row.append(float(row[21]))
    except ValueError:
        if (row[21]=='Infinity'):
            new_row.append(float('inf'))
        else:
            new_row.append(0.0)
    try:
        new_row.append(float(row[22]))
    except ValueError:
        if (row[22]=='Infinity'):
            new_row.append(float('inf'))
        else:
            new_row.append(0.0)
    new_row.append(float(row[23]))
    new_row.append(float(row[24]))
    new_row.append(int(float(row[25])))
    new_row.append(int(float(row[26])))
    new_row.append(int(float(row[27])))
    new_row.append(float(row[28]))
    new_row.append(float(row[29]))
    new_row.append(int(float(row[30])))
    new_row.append(int(float(row[31])))
    new_row.append(int(float(row[32])))
    new_row.append(float(row[33]))
    new_row.append(float(row[34]))
    new_row.append(int(float(row[35])))
    new_row.append(int(float(row[36])))
    new_row.append(int(row[37]))
    new_row.append(int(row[41]))
    new_row.append(int(row[42]))
    new_row.append(float(row[43]))
    new_row.append(float(row[44]))
    new_row.append(int(float(row[45])))
    new_row.append(int(float(row[46])))
    new_row.append(float(row[47]))
    new_row.append(float(row[48]))
    new_row.append(float(row[49]))
    new_row.append(int(row[51]))
    new_row.append(int(row[52]))
    new_row.append(int(row[54]))
    new_row.append(int(row[55]))
    new_row.append(int(row[56]))
    new_row.append(float(row[58]))
    new_row.append(float(row[59]))
    new_row.append(float(row[60]))
    new_row.append(float(row[61]))
    new_row.append(int(row[62]))
    new_row.append(int(row[69]))
    new_row.append(int(row[70]))
    new_row.append(int(row[71]))
    new_row.append(int(row[72]))
    new_row.append(int(row[73]))
    new_row.append(int(row[74]))
    new_row.append(int(row[75]))
    new_row.append(int(row[76]))
    new_row.append(float(row[77]))
    new_row.append(float(row[78]))
    new_row.append(int(float(row[79])))
    new_row.append(int(float(row[80])))
    new_row.append(float(row[81]))
    new_row.append(float(row[82]))
    new_row.append(int(float(row[83])))
    new_row.append(int(float(row[84])))
    #uses a public method to convert a string into a integer label 0-11
    new_row.append(label_assign(row[87]))
    return new_row

#the same method but for 2017 datasets
#(I could have used a dict reader and this would be much easier, but i didn't remember until i already did this)
def formatter2017(row):
    new_row = [int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(float(row[4])), int(float(row[5])), int(float(row[6])), int(float(row[7])), float(row[8]), float(row[9]),
               int(float(row[10])), int(float(row[11])), float(row[12]), float(row[13])]

    #these two values were tricky because they could be null or 'Infinity'
    try:
        new_row.append(float(row[14]))
    except ValueError:
        if (row[14]=='Infinity'):
            new_row.append(float('inf'))
        else:
            new_row.append(0.0)
    try:
        new_row.append(float(row[15]))
    except ValueError:
        if (row[15]=='Infinity'):
            new_row.append(float('inf'))
        else:
            new_row.append(0.0)
    new_row.append(float(row[16]))
    new_row.append(float(row[17]))
    new_row.append(int(float(row[18])))
    new_row.append(int(float(row[19])))
    new_row.append(int(float(row[20])))
    new_row.append(float(row[21]))
    new_row.append(float(row[22]))
    new_row.append(int(float(row[23])))
    new_row.append(int(float(row[24])))
    new_row.append(int(float(row[25])))
    new_row.append(float(row[26]))
    new_row.append(float(row[27]))
    new_row.append(int(float(row[28])))
    new_row.append(int(float(row[29])))
    new_row.append(int(row[30]))
    new_row.append(int(row[34]))
    new_row.append(int(row[35]))
    new_row.append(float(row[36]))
    new_row.append(float(row[37]))
    new_row.append(int(float(row[38])))
    new_row.append(int(float(row[39])))
    new_row.append(float(row[40]))
    new_row.append(float(row[41]))
    new_row.append(float(row[42]))
    new_row.append(int(row[44]))
    new_row.append(int(row[45]))
    new_row.append(int(row[47]))
    new_row.append(int(row[48]))
    new_row.append(int(row[49]))
    new_row.append(float(row[51]))
    new_row.append(float(row[52]))
    new_row.append(float(row[53]))
    new_row.append(float(row[54]))
    new_row.append(int(row[55]))
    new_row.append(int(row[62]))
    new_row.append(int(row[63]))
    new_row.append(int(row[64]))
    new_row.append(int(row[65]))
    new_row.append(int(row[66]))
    new_row.append(int(row[67]))
    new_row.append(int(row[68]))
    new_row.append(int(row[69]))
    new_row.append(float(row[70]))
    new_row.append(float(row[71]))
    new_row.append(int(float(row[72])))
    new_row.append(int(float(row[73])))
    new_row.append(float(row[74]))
    new_row.append(float(row[75]))
    new_row.append(int(float(row[76])))
    new_row.append(int(float(row[77])))
    #uses a public method to convert a string into a integer label 0-11
    new_row.append(label_assign(row[78]))
    return new_row

#setting the labels as ints so the model doesnt have to parse strings
def label_assign(label):
    label=label.lower()
    if (label.startswith("web attack")):
        label = label.replace("web attack ï¿½ ", "")
    match (label):
        case("benign"):
            return 0
        case("drdos_dns"):
            return 1
        case("drdos_ldap"):
            return 2
        case("drdos_mssql" | "sql injection"):
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
        case("udp-lag" | "udplag"):
            return 11
        case ("brute force"):
            return 12
        case ("xss"):
            return 13
        case("dos slowloris"):
            return 14
        case("dos slowhttptest"):
            return 15
        case("dos hulk"):
            return 16
        case("dos goldeneye"):
            return 17
        case("heartbleed"):
            return 18
        case("ftp-patator"):
            return 19
        case("ssh-patator"):
            return 20
        case("infiltration"):
            return 21
        case("ddos" | "webddos"):
            return 22
        case("portscan"):
            return 23
        case("bot"):
            return 24
        case _:
            return 25
        
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
        case(12):
            return "brute force"
        case(13):
            return "xss"
        case(14):
            return "dos slowloris"
        case(15):
            return "dos slowhttptest"
        case(16):
            return "dos hulk"
        case(17):
            return "dos goldeneye"
        case(18):
            return "heartbleed"
        case(19):
            return "ftp-patator"
        case(20):
            return "ssh-patator"
        case(21):
            return "infiltration"
        case(22):
            return "ddos"
        case(23):
            return "portscan"
        case(24):
            return "bot"
        case(25):
            return "other"

#returns an index for a value in a formatted list of when given a name of a field
def item_by_name(key):
    key=key.lower()
    match key:
        case "destination port" | "destination_port":
            return 0
        case "flow duration" | "flow_duration":
            return 1
        case "total fwd packets" | "ttl_fwd_pkts":
            return 2
        case "total backward packets" | "ttl_bwd_pkts":
            return 3
        case "total length of fwd packets" | "ttl_len_of_fwd_pkts":
            return 4
        case "total length of bwd packets" | "ttl_len_of_bwd_pkts":
            return 5
        case "fwd packet length max" | "fwd_pkt_len_max":
            return 6
        case "fwd packet length min" | "fwd_pkt_len_min":
            return 7
        case "fwd packet length mean" | "fwd_pkt_len_mean":
            return 8
        case "fwd packet length std" | "fwd_pkt_len_std":
            return 9
        case "bwd packet length max" | "bwd_pkt_len_max":
            return 10
        case "bwd packet length min" | "bwd_pkt_len_min":
            return 11
        case "bwd packet length mean" | "bwd_pkt_len_mean":
            return 12
        case "bwd packet length std" | "bwd_pkt_len_std":
            return 13
        case "flow bytes/s" | "flow_bytes_per_s":
            return 14
        case "flow packets/s" | "flow_pkts_per_s":
            return 15
        case "flow iat mean" | "flow_iat_mean":
            return 16
        case "flow iat std" | "flow_iat_std":
            return 17
        case "flow iat max" | "flow_iat_max":
            return 18
        case "flow iat min" | "flow_iat_min":
            return 19
        case "fwd iat total" | "fwd_iat_ttl":
            return 20
        case "fwd iat mean" | "fwd_iat_mean":
            return 21
        case "fwd iat std" | "fwd_iat_std":
            return 22
        case "fwd iat max" | "fwd_iat_max":
            return 23
        case "fwd iat min" | "fwd_iat_min":
            return 24
        case "bwd iat total" | "bwd_iat_ttl":
            return 25
        case "bwd iat mean" | "bwd_iat_mean":
            return 26
        case "bwd iat std" | "bwd_iat_std":
            return 27
        case "bwd iat max" | "bwd_iat_max":
            return 28
        case "bwd iat min" | "bwd_iat_min":
            return 29
        case "fwd psh flags" | "fwd_psh_flags":
            return 30
        case "fwd header length" | "fwd_header_len":
            return 31
        case "bwd header length" | "bwd_header_len":
            return 32
        case "fwd packets/s" | "fwd_pkts_per_s":
            return 33
        case "bwd packets/s" | "bwd_pkts_per_s":
            return 34
        case "min packet length" | "min_pkt_len":
            return 35
        case "max packet length" | "max_pkt_len":
            return 36
        case "packet length mean" | "pkt_len_mean":
            return 37
        case "packet length std" | "pkt_len_std":
            return 38
        case "packet length variance" | "pkt_len_variance":
            return 39
        case "syn flag count" | "syn_flag_count":
            return 40
        case "rst flag count" | "rst_flag_count":
            return 41
        case "ack flag count" | "ack_flag_count":
            return 42
        case "urg flag count" | "urg_flag_count":
            return 43
        case "cwe flag count" | "cwe_flag_count":
            return 44
        case "down/up ratio" | "down_to_up_ratio":
            return 45
        case "average packet size" | "average_pkt_size":
            return 46
        case "avg fwd segment size" | "avg_fwd_segment_size":
            return 47
        case "avg bwd segment size" | "avg_bwd_segment_size":
            return 48
        case "fwd header length.1" | "fwd_header_len.1":
            return 49
        case "subflow fwd packets" | "subflow_fwd_pkts":
            return 50
        case "subflow fwd bytes" | "subflow_fwd_bytes":
            return 51
        case "subflow bwd packets" | "subflow_bwd_pkts":
            return 52
        case "subflow bwd bytes" | "subflow_bwd_bytes":
            return 53
        case "init_win_bytes_forward" | "init_win_bytes_fwd":
            return 54
        case "init_win_bytes_backward" | "init_win_bytes_bwd":
            return 55
        case "act_data_pkt_fwd":
            return 56
        case "min_seg_size_forward" | "min_seg_size_fwd":
            return 57
        case "active mean" | "active_mean":
            return 58
        case "active std" | "active_std":
            return 59
        case "active max" | "active_max":
            return 60
        case "active min" | "active_min":
            return 61
        case "idle mean" | "idle_mean":
            return 62
        case "idle std" | "idle_std":
            return 63
        case "idle max" | "idle_max":
            return 64
        case "idle min" | "idle_min":
            return 65
        case "label":
            return 66
        case _:
            print('Incorrect key usage, returned -1')
            return -1
            
#method for converting a csv file to a given file type
#to_read is the csv file, output is where the file is saved, format_key determines if you use the 2017 formatter method
def pickle_maker(to_read, output, format_key):
    start_time = time.time()
    pre_pickle=[]
    
    #this segment automatically turns a provided filename into a csv or gzip
    to_read = to_read.split('.')[0] + '.csv'
    output = output.split('.')[0] + '.gz'
    
    print('File successfully opened! ' + to_read +'\n', end='')

    with open(to_read) as csvfile:
        cucumber = csv.reader(csvfile)
        cucumber.__next__()
        for row in cucumber:
            if(row[0]!=''):
                if (format_key=='2017'):
                    pre_pickle.append(formatter2017(row))
                else:
                    pre_pickle.append(formatter(row))
                
    print("Listified " + to_read + " as " + str(len(pre_pickle)) + " lists in" + "--- %s seconds ---" % (time.time() - start_time))
    print(pre_pickle[1])
    
    if (output=='TFTP.gz'):
        save_as(pre_pickle[0:int(len(pre_pickle)/4)], output.split('.')[0] + '_1.gz')
        save_as(pre_pickle[int(len(pre_pickle)/4):int(len(pre_pickle)/2)], output.split('.')[0] + '_2.gz')
        save_as(pre_pickle[int(len(pre_pickle)/2):(int(len(pre_pickle)/4)*3)], output.split('.')[0] + '_3.gz')
        save_as(pre_pickle[int((len(pre_pickle)/4)*3):len(pre_pickle)], output.split('.')[0] + '_4.gz')
    else:
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
        
    if (field<0 or field>=67):
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
    results=[True]*67
    base_case=big_data[0]
    for i in range(1,len(big_data)):
        row=big_data[i]
        for j in range(len(row)):
            if (results[j]):
                results[j] = (row[j]==0 or row[j]==1)
                
    print("Columns checked in--- %s seconds ---" % (time.time() - start_time))

    return results


#saves a static list of csv files as a gzip compressed pickle containing 1 list of all the relevant rows
def save_all():
    start_time = time.time()
    files = ['DrDoS_UDP','DrDos_DNS', 'DrDoS_LDAP', 'DrDoS_MSSQL', 'DrDoS_NetBIOS',
             'DrDoS_NTP', 'DrDoS_SNMP', 'DrDoS_SSDP', 'Syn', 'UDPLag', 'TFTP']
    
    threads = [None] * len(files)
    
    for i in range(len(threads)):
        pickle_maker(files[i], files[i], 2019)
 
    my_time = (time.time() - start_time)    
    print("11 files pickled in" + "--- %s seconds --- (avg: %s)" % (my_time, my_time/11 ))

#makes normalized list from a subst of gzip files, only takes as many attack as benign in file. Files with more benign data will not work here
def combine_normal_list(files):
    result=[]
    
    for i in range(len(files)):
        pick = load_pickle(files[i]+ '.gz')
        benign = get_match_rows(pick, 66, 0)
        j=0
        sent=0
        
        #j iteration is redundant, but avoids potential errors
        while (sent<len(benign) and j < len(pick)):
            if (pick[j][66]>0):
                result.append(pick[j])
                result.append(benign[sent])
                sent+=1
            j+=1
            
    return result

#'DrDos_DNS', 'DrDoS_LDAP', 'DrDoS_MSSQL', 'DrDoS_NetBIOS', 'DrDoS_NTP', 'DrDoS_SNMP', 'DrDoS_SSDP', 'DrDoS_UDP', 'Syn', 'TFTP', 'UDPLag'
#'Wednesday-workingHours','Monday-WorkingHours', 'Tuesday-WorkingHours', 'Thursday-WorkingHours-Afternoon-Infilteration', 'Friday-WorkingHours-Afternoon-DDos', 'Friday-WorkingHours-Afternoon-PortScan', 'Friday-WorkingHours-Morning', 'Thursday-WorkingHours-Morning-WebAttacks'
#used to check load times and ratios of benign to attack rows
def load_check():
    files = ['DrDos_DNS', 'DrDoS_LDAP', 'DrDoS_MSSQL', 'DrDoS_NetBIOS',
             'DrDoS_NTP', 'DrDoS_SNMP', 'DrDoS_SSDP', 'DrDoS_UDP', 'Syn', 'TFTP_1', 'UDPLag', 'TFTP_2', 'TFTP_3', 'TFTP_4']
    start_time=time.time()
    for i in range(len(files)):
        load_time = time.time()
        pick = load_pickle(files[i]+ '.gz')
        print(files[i] +" opened in" + "--- %s seconds ---" % (time.time() - load_time))
        beny = 0
        baddy = 0
        for j in range(len(pick)):
            if (pick[j][66]==0):
                beny+=1
            else:
                baddy+=1
        if baddy==0:
            ratio=beny
        else:
            ratio=beny/baddy
        print (files[i] + " contains %s Benign rows and %s Attack rows totalling %s rows and ratio of 1:%s Benign" % (beny, baddy, len(pick), ratio))
    my_time = (time.time() - start_time)
    print("11 files loaded in" + "--- %s seconds --- (avg: %s)" % (my_time, my_time/11 ))

def main():
    load_check()
    files = ['DrDos_DNS', 'DrDoS_LDAP', 'DrDoS_MSSQL', 'DrDoS_NetBIOS',
             'DrDoS_NTP', 'DrDoS_SNMP', 'DrDoS_SSDP', 'DrDoS_UDP', 'Syn', 'TFTP', 'UDPLag']
    
if __name__ == '__main__':
    sys.exit(main())
