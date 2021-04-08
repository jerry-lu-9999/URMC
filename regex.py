import pandas as pd
import re

def regex(str):
    # False is no fever, true is fever
    if "GOLISANO CHILDREN'S HOSPITAL" in str or "After Visit Summary Signature Page" in str:
        return False
    temp_sign = 0

    # Celcius check
    celsius = re.compile(r'(\d+\.\d+)\s?(°C|C)', re.IGNORECASE)
    matched = re.search(celsius, str)
    # group(0) is the whole string, 1 is the first group, etc.
    if matched:
        if float(matched.group(1)) >= 38:
            temp_sign |= 1

    # Fahrenheit check
    fah = re.compile(r"(\d+\.\d+)\s?(°F|f|F)", re.IGNORECASE)
    matched = re.search(fah, str)
    if matched:
        if float(matched.group(1)) >= 100.4:
            temp_sign |= 1

    final = re.compile(r"(?:fever |temperature |Tmax )(?:\S+\s+){1,4}(\d{2,3}\.?\d?)", re.IGNORECASE)
    matched = re.search(final, str)
    if matched:
        if float(matched.group(1)) >= 100.4:
            temp_sign |= 1

    return True if temp_sign == 1 else False


corpus = pd.read_excel("C:/Users/jlu39/Desktop/2017 Master Notes Data. ED, Admission Notes. 8.20.xlsx",
                       sheet_name="ED, H&P Notes Only",
                       header=0,
                       usecols="D,T,X")

# corpus = pd.read_excel("C:/Users/jlu39/Desktop/2016 Master Notes Data. ED, Admission Notes. 9.9.20.xlsx",
#                        sheet_name="ED, H&P Notes Only",
#                        header=0,
#                        usecols="D,K,O")

sign = 0
both_neg = 0
both_pos = 0
com_fever_man_no = 0
com_no_man_fever = 0

master_dict = {}
nest_dict = {}
list_for_note = []
list_for_empty_diag = []
false_neg = []
false_pos = []
fever = None
# TODO: we will need a hashmap after all: 55349450
for index in range(len(corpus.index) - 1):
    if pd.notnull(corpus.iloc[index, corpus.columns.get_loc('pat_enc_csn_id')]):
        enc_id = corpus.iloc[index, corpus.columns.get_loc('pat_enc_csn_id')]
    else:
        continue
    note = corpus.iloc[index, corpus.columns.get_loc('note')]
    next_enc_id = corpus.iloc[index + 1, corpus.columns.get_loc('pat_enc_csn_id')]

    if pd.notnull(corpus.iloc[index, corpus.columns.get_loc('Fever?')]):
        fever = corpus.iloc[index, corpus.columns.get_loc('Fever?')]

    # ignore check for AVS cuz it is clean data already
    list_for_note.append(note)

    if (not enc_id == next_enc_id) or (index == len(corpus.index) - 2):
        if fever is None:
            list_for_empty_diag.append(enc_id)
        nest_dict['Fever?'] = fever

        if enc_id in master_dict.keys():
            nest_dict['note'] = master_dict[enc_id]['note'] + "".join(str(e) for e in list_for_note)
            master_dict[enc_id] = nest_dict
            if enc_id in list_for_empty_diag:
                list_for_empty_diag.remove(enc_id)
        else:
            nest_dict['note'] = "".join(str(e) for e in list_for_note)
            master_dict[enc_id] = nest_dict

        if regex(nest_dict['note']):
            sign |= 1

        if fever == 'Y' and sign == 1:
            both_pos += 1
        elif fever == 'Y' and sign == 0:
            com_no_man_fever += 1
            false_neg.append(enc_id)
        elif fever == 'N' and sign == 0:
            both_neg += 1
        elif fever == 'N' and sign == 1:
            com_fever_man_no += 1
            false_pos.append(enc_id)
        sign = 0
        list_for_note = []
        nest_dict = {}
        fever = None

print(list_for_empty_diag)
print(false_pos)
print(false_neg)

total_size = both_pos + both_neg + com_no_man_fever + com_fever_man_no
print(total_size)
print("TN = %d, TP = %d, FN = %d, FP = %d" % (both_neg, both_pos, com_no_man_fever, com_fever_man_no))
print("Regular expression score ->", 1 - (com_fever_man_no + com_no_man_fever) / total_size)
