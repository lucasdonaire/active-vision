import numpy as np 
import pandas as pd 
import scipy.io as sio
import traceback

def parse_cell_line(line):
    parts = line.strip().split(':')
    if not parts or parts[0] != 'start':
        return None
    row_data = {}
    i = 0
    while i < len(parts):
        key = parts[i]
        if key == 'end':
            break
        if key == 'site':
            row_data['grid_coord'] = parts[i+1]
            row_data['depth'] = parts[i+2]
            row_data['brain_area'] = parts[i+3]
            i += 4  
        else:
            if i + 1 < len(parts):
                val = parts[i+1]
                row_data[key] = np.nan if val == 'NaN' else val
            i += 2
    return row_data


def classify_rf(row):
    s_val = str(row.get('sample', 'NaN')).strip()
    a_val = str(row.get('array', 'NaN')).strip()
    
    resp_sample = s_val in ['0', '0.0']
    resp_array = a_val in ['0', '0.0']
    
    if resp_sample and not resp_array:
        return 'Focal Foveal'       
    elif resp_sample and resp_array:
        return 'Broad Foveal'       
    elif not resp_sample and resp_array:
        return 'Peripheral'         
    else:
        return 'Non-responsive'     


def get_dataframe_cell_list(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for row in file:
                process_row = parse_cell_line(row)
                if process_row is not None:
                    data.append(process_row)
                    
    except FileNotFoundError:
        print(f"Erro: O arquivo '{file_path}' não foi encontrado.")

    df = pd.DataFrame(data)
    cols_to_numeric = ['start', 'depth', 'sample', 'array', 'SI']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['RF_type'] = df.apply(classify_rf, axis=1)
    df['session'] = df['search'].apply(lambda x: x[:9])
    df['neuron_id'] = df['search'].apply(lambda x: x[9:])
    return df


def get_image_cathegory(image_id):
    return {1:'flower',2:'house',3:'face',4:'hand',9:'fix_point'}[image_id // 1000]



def get_df_trials(list_neuron_ids, path_destiny, list_prefered_cat, list_brain_area, firings_cue_and_array=False):
    trial_num_TrialInfoIdx = 0 # Numero da trial
    trial_clue_imgid_TrialInfoIdx = 4 # Id da foto da pista
    trial_FP_on_time_TrialInfoIdx =  18 # Fixation point na tela
    trial_fix_FP_TrialInfoIdx =  18 # Macaco fixa o FP
    trial_clue_on_time_TrialInfoIdx = 19 # pista aparece
    trial_clue_off_time_TrialInfoIdx = 20 # pista desaparece
    trial_FP_back_time_TrialInfoIdx =  21 # Fixation point reaparece
    trial_array_on_time_TrialInfoIdx = 22 # array aparece
    trial_sacc_time_TrialInfoIdx = 23 # tempo do inicio da primeira sacada
    rows_df = []
    for i, neuron_id in enumerate(list_neuron_ids):
        trial_path = f'{path_destiny}\{neuron_id}.mat'
        try:
            mat_data = sio.loadmat(trial_path)
            prefered_cat = list_prefered_cat[i]
            brain_area = list_brain_area[i]
            for trial_idx, trial_data in enumerate(mat_data['TrlInfoMatrix']):
                if mat_data['SearchEye'][trial_idx][0].shape[1] != 8:
                    continue
                else:
                    trial_num = int(trial_data[trial_num_TrialInfoIdx])
                    trial_clue_imgid = trial_data[trial_clue_imgid_TrialInfoIdx]
                    trial_FP_on_time = trial_data[trial_FP_on_time_TrialInfoIdx]
                    trial_fix_FP = trial_data[trial_fix_FP_TrialInfoIdx]
                    trial_clue_on_time = trial_data[trial_clue_on_time_TrialInfoIdx]
                    trial_clue_off_time = trial_data[trial_clue_off_time_TrialInfoIdx]
                    trial_FP_back_time = trial_data[trial_FP_back_time_TrialInfoIdx]
                    trial_array_on_time = trial_data[trial_array_on_time_TrialInfoIdx]
                    trial_sacc_time = trial_data[trial_sacc_time_TrialInfoIdx]
                    if pd.isnull(trial_sacc_time): continue
                    trial_photo_1sacc_imgid = mat_data['SearchEye'][trial_idx][0][1][6] # 0 pq ta tudo num array so, 1 primeira sacada (0 é fixacao), 6 id da foto da primeira sacada
                    trial_time_end_1fix = mat_data['SearchEye'][trial_idx][0][1][2] # quando ele para de fixar o primeiro alvo (seja pq a trial acabou, seja pq ele fez uma segunda sacada). nao olharemos mais do que isso.
                    firings = mat_data['neuron'][(mat_data['neuron'] >= trial_FP_on_time) & (mat_data['neuron'] <= trial_time_end_1fix)] # desde o começo do FP ate o final da primeira sacada
                    clue_cathegory = get_image_cathegory(trial_clue_imgid)
                    photo_1sacc_cathegory = get_image_cathegory(trial_photo_1sacc_imgid)
                    trial_presacc_time = trial_sacc_time - trial_array_on_time
                    if firings_cue_and_array:
                        firings_cue = [item - trial_clue_on_time for item in firings]
                        firings_array = [item - trial_array_on_time for item in firings]
                        firings = [item - trial_sacc_time for item in firings]
                        rows_df.append([neuron_id,prefered_cat,brain_area,trial_num,trial_clue_imgid,clue_cathegory,trial_photo_1sacc_imgid,photo_1sacc_cathegory,trial_presacc_time,trial_FP_on_time,trial_fix_FP,trial_clue_on_time,trial_clue_off_time,trial_FP_back_time,trial_array_on_time,trial_sacc_time,trial_time_end_1fix,firings,firings_cue,firings_array]) 
                    else:
                        firings = [item - trial_sacc_time for item in firings]
                        rows_df.append([neuron_id,prefered_cat,brain_area,trial_num,trial_clue_imgid,clue_cathegory,trial_photo_1sacc_imgid,photo_1sacc_cathegory,trial_presacc_time,trial_FP_on_time,trial_fix_FP,trial_clue_on_time,trial_clue_off_time,trial_FP_back_time,trial_array_on_time,trial_sacc_time,trial_time_end_1fix,firings]) 
        except Exception as e:
            print('ERRO!!!', neuron_id)
            print(traceback.format_exc())
    if firings_cue_and_array:
        df_trials = pd.DataFrame(rows_df, columns=['neuron','prefered','brain_area','trial_num','clue_id','clue_cath','imgsacc_id','imgsacc_cath','pre_sacc_time','FP_on','fix_FP','clue_on','clue_off','FP_back','array_on','sacc_start','end_first_fix','firings','firings_cue','firings_array'])
    else:
        df_trials = pd.DataFrame(rows_df, columns=['neuron','prefered','brain_area','trial_num','clue_id','clue_cath','imgsacc_id','imgsacc_cath','pre_sacc_time','FP_on','fix_FP','clue_on','clue_off','FP_back','array_on','sacc_start','end_first_fix','firings'])
    return df_trials