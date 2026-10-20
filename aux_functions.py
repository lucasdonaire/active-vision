import numpy as np 
import pandas as pd 
import scipy.io as sio
import seaborn as sns
import traceback
from scipy.stats import loguniform
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef
import matplotlib.pyplot as plt


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


## FIRST SACCADE ONLY
def get_df_trials(list_neuron_ids, path_destiny, list_prefered_cat, list_brain_area, list_RF_type, firings_cue_and_array=False):
    trial_num_TrialInfoIdx = 0 # Numero da trial
    trial_clue_imgid_TrialInfoIdx = 4 # Id da foto da pista
    trial_FP_on_time_TrialInfoIdx =  17 # Fixation point na tela
    trial_fix_FP_TrialInfoIdx =  18 # Macaco fixa o FP
    trial_clue_on_time_TrialInfoIdx = 19 # pista aparece
    trial_clue_off_time_TrialInfoIdx = 20 # pista desaparece
    trial_FP_back_time_TrialInfoIdx =  21 # Fixation point reaparece
    trial_array_on_time_TrialInfoIdx = 22 # array aparece
    trial_sacc_time_TrialInfoIdx = 23 # tempo do inicio da primeira sacada
    rows_df = []
    for i, neuron_id in enumerate(list_neuron_ids):
        # trial_path = f'{path_destiny}\{neuron_id}.mat' # windows
        trial_path = f'{path_destiny}/{neuron_id}.mat' # linux
        try:
            mat_data = sio.loadmat(trial_path)
            prefered_cat = list_prefered_cat[i]
            brain_area = list_brain_area[i]
            RF_type = list_RF_type[i]
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
                    trial_time_end_1fix = mat_data['SearchEye'][trial_idx][0][1][2] # 0 pq ta tudo num array so, 1 primeira sacada, 2 é tempo de finalizacao da fixacao # quando ele para de fixar o primeiro alvo (seja pq a trial acabou, seja pq ele fez uma segunda sacada). nao olharemos mais do que isso.
                    trial_time_start_1fix = mat_data['SearchEye'][trial_idx][0][1][1] # 0 pq ta tudo num array so, 1 primeira sacada, 1 é tempo de começo da fixacao # quando ele para de fixar o primeiro alvo (seja pq a trial acabou, seja pq ele fez uma segunda sacada). nao olharemos mais do que isso.
                    firings = mat_data['neuron'][(mat_data['neuron'] >= trial_FP_on_time) & (mat_data['neuron'] <= trial_time_end_1fix)] # desde o começo do FP ate o final da primeira sacada
                    clue_cathegory = get_image_cathegory(trial_clue_imgid)
                    photo_1sacc_cathegory = get_image_cathegory(trial_photo_1sacc_imgid)
                    trial_presacc_time = trial_sacc_time - trial_array_on_time
                    if firings_cue_and_array:
                        firings_cue = [item - trial_clue_on_time for item in firings] # 0 = cue onset
                        firings_array = [item - trial_array_on_time for item in firings] # 0 = array onset
                        firings = [item - trial_sacc_time for item in firings] # 0 = beggining of saccade
                        rows_df.append([neuron_id,prefered_cat,brain_area,RF_type,trial_num,trial_clue_imgid,clue_cathegory,trial_photo_1sacc_imgid,photo_1sacc_cathegory,trial_presacc_time,trial_FP_on_time,trial_fix_FP,trial_clue_on_time,trial_clue_off_time,trial_FP_back_time,trial_array_on_time,trial_sacc_time,trial_time_start_1fix,trial_time_end_1fix,firings,firings_cue,firings_array]) 
                    else:
                        firings = [item - trial_sacc_time for item in firings]
                        rows_df.append([neuron_id,prefered_cat,brain_area,RF_type,trial_num,trial_clue_imgid,clue_cathegory,trial_photo_1sacc_imgid,photo_1sacc_cathegory,trial_presacc_time,trial_FP_on_time,trial_fix_FP,trial_clue_on_time,trial_clue_off_time,trial_FP_back_time,trial_array_on_time,trial_sacc_time,trial_time_start_1fix,trial_time_end_1fix,firings]) 
        except Exception as e:
            print('ERRO!!!', neuron_id)
            print(traceback.format_exc())
    if firings_cue_and_array:
        df_trials = pd.DataFrame(rows_df, columns=['neuron','prefered','brain_area','RF_type','trial_num','clue_id','clue_cath','imgsacc_id','imgsacc_cath','pre_sacc_time','FP_on','fix_FP','clue_on','clue_off','FP_back','array_on','sacc_start','start_first_fix','end_first_fix','firings','firings_cue','firings_array'])
    else:
        df_trials = pd.DataFrame(rows_df, columns=['neuron','prefered','brain_area','RF_type','trial_num','clue_id','clue_cath','imgsacc_id','imgsacc_cath','pre_sacc_time','FP_on','fix_FP','clue_on','clue_off','FP_back','array_on','sacc_start','start_first_fix','end_first_fix','firings'])
    return df_trials




from scipy.signal import lfilter
def preprocess_spikes(spike_times, start_time=-500, end_time=0, sigma=15, bin_start=-200, bin_size=50):
    """
    Realiza o pipeline de pré-processamento para um único neurônio em um único trial:
    1. Suavização causal (unilateral) com kernel half-Gaussian (resolução de 1 ms).
    2. Agrupamento (binning) em intervalos regulares.
    
    Parâmetros:
    -----------
    spike_times : list ou np.ndarray
        Tempos de disparo (spikes) em segundos.
    start_time : int
        Início da janela temporal para a convolução (padrão: -300 ms).
    end_time : int
        Fim da janela temporal (padrão: 0 ms, início da sacada).
    sigma : float
        O desvio padrão do kernel Gaussiano em ms. Se for 0, realiza apenas o binning clássico.
    bin_start : int
        O início da janela onde os bins serão extraídos (padrão: -200 ms).
    bin_size : int
        Tamanho de cada bin de amostragem em ms (padrão: 50 ms).
        
    Retorna:
    --------
    bins : np.ndarray
        Vetor unidimensional com a taxa de disparo (Hz) média em cada bin.
    """
    spike_times = np.array([1000*item for item in spike_times])
    spike_times = np.array([item for item in spike_times if item >= start_time and item <= end_time])


    if sigma == 0:
        num_bins = int((end_time - bin_start) / bin_size)
        bins = []
        bin_edges = []
        spikes = np.array(spike_times)
        for i in range(num_bins):
            b_left = bin_start + i * bin_size
            b_right = b_left + bin_size
            n_spikes = np.sum((spikes >= b_left) & (spikes <= b_right))
            hz = (n_spikes / bin_size) * 1000.0
            bins.append(hz)
            bin_edges.append((b_left, b_right))
        return np.array(bins)
    
    time_grid_extended = np.arange(start_time, end_time + 1, 1)
    spike_train_extended = np.zeros_like(time_grid_extended, dtype=float)
    
    indices = np.round(spike_times - start_time).astype(int)
    for idx in indices:
        if 0 <= idx < len(spike_train_extended):
            spike_train_extended[idx] += 1.0
            
    # Criar o kernel causal half-Gaussian (suporte apenas para t >= 0)
    margin = int(5 * sigma)
    kernel_t = np.arange(0, margin)
    kernel = np.exp(- (kernel_t ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)  # Normalização para manter a amplitude em taxa de disparo
    kernel_hz = kernel * 1000.0  # Converter para Hz
    
    # Filtragem causal via lfilter (equivalente a uma convolução causal)
    firing_rate_extended = lfilter(kernel_hz, [1.0], spike_train_extended)
    
    # Remover a margem de segurança
    firing_rate = firing_rate_extended[margin:]
    time_grid = time_grid_extended[margin:]
    
    # Agrupar a taxa de disparo contínua nos bins desejados
    num_bins = int((end_time - bin_start) / bin_size)
    bins = []
    bin_edges = []
    
    for i in range(num_bins):
        b_left = bin_start + i * bin_size
        b_right = b_left + bin_size
        
        idx_left = np.where(time_grid == b_left)[0][0]
        if i == num_bins - 1:
            idx_right = np.where(time_grid == b_right)[0][0] + 1
        else:
            idx_right = np.where(time_grid == b_right)[0][0]
            
        mean_fr = np.mean(firing_rate[idx_left:idx_right])
        bins.append(mean_fr)
        bin_edges.append((b_left, b_right))
        
    return np.array(bins)




def get_df_trials_all_saccs(list_neuron_ids, path_destiny, list_prefered_cat, list_brain_area, list_RF_type, list_RF_IN, dicio_sacc_vectors):
    trial_num_TrialInfoIdx = 0 # Numero da trial
    trial_cue_imgid_TrialInfoIdx = 4 # Id da foto da pista
    trial_FP_on_time_TrialInfoIdx =  17 # Fixation point na tela
    trial_fix_FP_TrialInfoIdx =  18 # Macaco fixa o FP
    trial_cue_on_time_TrialInfoIdx = 19 # pista aparece
    trial_cue_off_time_TrialInfoIdx = 20 # pista desaparece
    trial_FP_back_time_TrialInfoIdx =  21 # Fixation point reaparece
    trial_array_on_time_TrialInfoIdx = 22 # array aparece
    trial_1sacc_time_TrialInfoIdx = 23 # tempo do inicio da primeira sacada
    rows_df = []
    dicio_firings = {}
    for i, neuron_id in enumerate(list_neuron_ids):
        trial_path = f'{path_destiny}/{neuron_id}.mat' 
        try:
            mat_data = sio.loadmat(trial_path)
            prefered_cat = list_prefered_cat[i]
            brain_area = list_brain_area[i]
            RF_type = list_RF_type[i]
            RF_in = list_RF_IN[i]
            if pd.isnull(RF_in):
                ROWS_RF_IN = [False]* 21
            else:
                RF_in_list = RF_in.split(',')
                ROWS_RF_IN = [str(index_rf) in RF_in_list for index_rf in range(21)]
            dicio_firings[neuron_id] = mat_data['neuron']
            for trial_idx, trial_data in enumerate(mat_data['TrlInfoMatrix']):
                if mat_data['SearchEye'][trial_idx][0].shape[1] != 8: # trial problematica
                    continue
                trial_1sacc_time = trial_data[trial_1sacc_time_TrialInfoIdx]
                if pd.isnull(trial_1sacc_time): continue
                trial_num = int(trial_data[trial_num_TrialInfoIdx])
                trial_cue_imgid = trial_data[trial_cue_imgid_TrialInfoIdx]
                trial_FP_on_time = trial_data[trial_FP_on_time_TrialInfoIdx]
                trial_fix_FP = trial_data[trial_fix_FP_TrialInfoIdx]
                trial_cue_on_time = trial_data[trial_cue_on_time_TrialInfoIdx]
                trial_cue_off_time = trial_data[trial_cue_off_time_TrialInfoIdx]
                trial_FP_back_time = trial_data[trial_FP_back_time_TrialInfoIdx]
                trial_array_on_time = trial_data[trial_array_on_time_TrialInfoIdx]
                cue_category = get_image_cathegory(trial_cue_imgid)
                trial_presacc_time = trial_1sacc_time - trial_array_on_time
                OFF_TIME_ANTERIOR = trial_1sacc_time
                LOC_ANTERIOR = 0
                CAT_ANTERIOR = 'FP'
                for sacc_idx, sacc_data in enumerate(mat_data['SearchEye'][trial_idx][0][1:]): # 0 pq tudo ta num grande array, 1+ pq 0 é a fixação no FP
                    sti_loc_idx = sacc_data[3] # id da localização no array (0-20)
                    image_id = sacc_data[6] # id da imagem
                    image_cat = get_image_cathegory(image_id)
                    target_on_time = sacc_data[1] # tempo de quando esse objeto foi fixado
                    target_off_time = sacc_data[2] # tempo de fim da sacada
                    SACC_VECTOR = dicio_sacc_vectors[(LOC_ANTERIOR, sti_loc_idx)]
                    VECTOR_IN_RF = False
                    if not pd.isnull(SACC_VECTOR):
                        VECTOR_IN_RF = ROWS_RF_IN[int(SACC_VECTOR)]
                    rows_df.append([
                        neuron_id,
                        prefered_cat,
                        brain_area,
                        RF_type,
                        RF_in,
                        trial_num,
                        trial_cue_imgid,
                        cue_category,
                        trial_presacc_time,
                        trial_FP_on_time,
                        trial_fix_FP,
                        trial_cue_on_time,
                        trial_cue_off_time,
                        trial_FP_back_time,
                        trial_array_on_time,
                        trial_1sacc_time,
                        sacc_idx,
                        f'{trial_num}_{sacc_idx}',
                        sti_loc_idx,
                        image_id,
                        image_cat,
                        OFF_TIME_ANTERIOR,
                        target_on_time,
                        target_off_time,
                        LOC_ANTERIOR,
                        CAT_ANTERIOR,
                        SACC_VECTOR,
                        VECTOR_IN_RF,
                        ])
                    OFF_TIME_ANTERIOR = target_off_time # depois de guardar a row, atualiza o tempo de Saída da sacada para o próximo
                    LOC_ANTERIOR = sti_loc_idx
                    CAT_ANTERIOR = image_cat

        except Exception as e:
            print('ERRO!!!', neuron_id) 
            print(traceback.format_exc())
            raise e

    df_trials = pd.DataFrame(rows_df, columns=[
        'neuron',
        'prefered',
        'brain_area',
        'RF_type',
        'RF_in',
        'trial_num',
        'cue_id',
        'cue_cath',
        'pre_sacc_time',
        'FP_on',
        'fix_FP',
        'cue_on',
        'cue_off',
        'FP_back',
        'array_on',
        '1sacc_start',
        'sacc_num',
        'trial+sacc',
        'sacc_stim_loc',
        'sacc_img_id',
        'sacc_img_cath',
        'sacc_start_time',
        'fix_time',
        'fix_end_time',
        'stim_loc_anterior',
        'img_cath_anterior',
        'saccade_vector',
        'vector_in_RF'
        ])
    return df_trials, dicio_firings



# FOVEAL ONLY, ALL SACCADES
def get_df_trials_all_saccs_foveal(list_neuron_ids, path_destiny, list_prefered_cat, list_brain_area):
    trial_num_TrialInfoIdx = 0 # Numero da trial
    trial_cue_imgid_TrialInfoIdx = 4 # Id da foto da pista
    trial_FP_on_time_TrialInfoIdx =  17 # Fixation point na tela
    trial_fix_FP_TrialInfoIdx =  18 # Macaco fixa o FP
    trial_cue_on_time_TrialInfoIdx = 19 # pista aparece
    trial_cue_off_time_TrialInfoIdx = 20 # pista desaparece
    trial_FP_back_time_TrialInfoIdx =  21 # Fixation point reaparece
    trial_array_on_time_TrialInfoIdx = 22 # array aparece
    trial_1sacc_time_TrialInfoIdx = 23 # tempo do inicio da primeira sacada
    rows_df = []
    dicio_firings = {}
    for i, neuron_id in enumerate(list_neuron_ids):
        trial_path = f'{path_destiny}/{neuron_id}.mat' 
        try:
            mat_data = sio.loadmat(trial_path)
            prefered_cat = list_prefered_cat[i]
            brain_area = list_brain_area[i]
            dicio_firings[neuron_id] = mat_data['neuron']
            for trial_idx, trial_data in enumerate(mat_data['TrlInfoMatrix']):
                if mat_data['SearchEye'][trial_idx][0].shape[1] != 8: # trial problematica
                    continue
                trial_1sacc_time = trial_data[trial_1sacc_time_TrialInfoIdx]
                if pd.isnull(trial_1sacc_time): continue
                trial_num = int(trial_data[trial_num_TrialInfoIdx])
                trial_cue_imgid = trial_data[trial_cue_imgid_TrialInfoIdx]
                trial_FP_on_time = trial_data[trial_FP_on_time_TrialInfoIdx]
                trial_fix_FP = trial_data[trial_fix_FP_TrialInfoIdx]
                trial_cue_on_time = trial_data[trial_cue_on_time_TrialInfoIdx]
                trial_cue_off_time = trial_data[trial_cue_off_time_TrialInfoIdx]
                trial_FP_back_time = trial_data[trial_FP_back_time_TrialInfoIdx]
                trial_array_on_time = trial_data[trial_array_on_time_TrialInfoIdx]
                cue_category = get_image_cathegory(trial_cue_imgid)
                trial_presacc_time = trial_1sacc_time - trial_array_on_time
                OFF_TIME_ANTERIOR = trial_1sacc_time
                LOC_ANTERIOR = 0
                CAT_ANTERIOR = 'fix_point'
                for sacc_idx, sacc_data in enumerate(mat_data['SearchEye'][trial_idx][0][1:]): # 0 pq tudo ta num grande array, 1+ pq 0 é a fixação no FP
                    sti_loc_idx = sacc_data[3] # id da localização no array (0-20)
                    image_id = sacc_data[6] # id da imagem
                    image_cat = get_image_cathegory(image_id)
                    target_on_time = sacc_data[1] # tempo de quando esse objeto foi fixado
                    target_off_time = sacc_data[2] # tempo de fim da sacada
                    rows_df.append([
                        neuron_id,
                        prefered_cat,
                        brain_area,
                        trial_num,
                        trial_cue_imgid,
                        cue_category,
                        trial_presacc_time,
                        trial_FP_on_time,
                        trial_fix_FP,
                        trial_cue_on_time,
                        trial_cue_off_time,
                        trial_FP_back_time,
                        trial_array_on_time,
                        trial_1sacc_time,
                        sacc_idx,
                        f'{trial_num}_{sacc_idx}',
                        sti_loc_idx,
                        image_id,
                        image_cat,
                        OFF_TIME_ANTERIOR,
                        target_on_time,
                        target_off_time,
                        LOC_ANTERIOR,
                        CAT_ANTERIOR,
                        ])
                    OFF_TIME_ANTERIOR = target_off_time # depois de guardar a row, atualiza o tempo de Saída da sacada para o próximo
                    LOC_ANTERIOR = sti_loc_idx
                    CAT_ANTERIOR = image_cat

        except Exception as e:
            print('ERRO!!!', neuron_id) 
            print(traceback.format_exc())
            raise e

    df_trials = pd.DataFrame(rows_df, columns=[
        'neuron',
        'prefered',
        'brain_area',
        'trial_num',
        'cue_id',
        'cue_cath',
        'pre_sacc_time',
        'FP_on',
        'fix_FP',
        'cue_on',
        'cue_off',
        'FP_back',
        'array_on',
        '1sacc_start',
        'sacc_num',
        'trial+sacc',
        'sacc_stim_loc',
        'sacc_img_id',
        'sacc_img_cath',
        'sacc_start_time',
        'fix_time',
        'fix_end_time',
        'stim_loc_anterior',
        'img_cath_anterior',
        ])
    return df_trials, dicio_firings



def get_df_run_ml(df_trials, start_time, end_time, bin_width, step, dicio_firings):
    t_ends = np.arange(start_time + bin_width, end_time + step, step)
    t_starts = t_ends - bin_width  # Para evitar calcular isto no loop

    neurons_order = list(df_trials['neuron'].unique())
    saccades_order = list(df_trials['trial+sacc'].unique())

    df_indexed = df_trials.set_index(['trial+sacc', 'neuron'])
    dict_fix_times = df_indexed['fix_time'].to_dict()

    df_saccades = df_trials.drop_duplicates('trial+sacc').set_index('trial+sacc')
    dict_cat_anterior = df_saccades['img_cath_anterior'].to_dict()
    dict_Y_cat = df_saccades['sacc_img_cath'].to_dict()
    dict_Y_equality = (df_saccades['sacc_img_cath'] == df_saccades['img_cath_anterior']).astype(int).to_dict()
    dict_cue = df_saccades['cue_cath'].to_dict()
    dict_Y_is_cue = (df_saccades['sacc_img_cath'] == df_saccades['cue_cath']).astype(int).to_dict()

    dicio_firings_np = {
        neuron: np.sort(np.array(spikes) * 1000.0) 
        for neuron, spikes in dicio_firings.items()
    }

    data_trials_run_model = []
    for saccade_id in saccades_order:
        Y_cat = dict_Y_cat.get(saccade_id, 0)
        Y_is_cue =  dict_Y_is_cue.get(saccade_id, 0)
        if Y_cat in ['face','house']:
            if Y_is_cue==1:
                Y_attention = 'Attention'
            else:
                Y_attention = 'No Attention'
        else:
            Y_attention = 'Other Cats'
        data_model_row = {
            'Y_cat': Y_cat,
            'Y_equality': dict_Y_equality.get(saccade_id, 0),
            'Y_is_cue': Y_is_cue,
            'cat_cue': dict_cue.get(saccade_id, 0),
            'cat_anterior': dict_cat_anterior.get(saccade_id, 0),
            'transition': f'{dict_cat_anterior.get(saccade_id, 0)}-{dict_Y_cat.get(saccade_id, 0)}',
            'Y_attention': Y_attention
        }
        
        for neuron in neurons_order:
            fix_time = dict_fix_times.get((saccade_id, neuron))
            fix_time_ms = fix_time * 1000.0
            session_spikes = dicio_firings_np[neuron][0]
            
            window_start_global = fix_time_ms + start_time
            window_end_global = fix_time_ms + end_time
            
            idx_start = np.searchsorted(session_spikes, window_start_global)
            idx_end = np.searchsorted(session_spikes, window_end_global)
            
            trial_spikes = session_spikes[idx_start:idx_end] - fix_time_ms
            
            for i, (t_s, t_e) in enumerate(zip(t_starts, t_ends)):
                count = np.sum((trial_spikes >= t_s) & (trial_spikes < t_e))
                fr = count / (bin_width / 1000.0)
                data_model_row[f'{neuron}_Bin{i}'] = fr
                
        data_trials_run_model.append(data_model_row)

    df_run_model = pd.DataFrame(data_trials_run_model)
    return df_run_model


def zscore_per_neuron(X_train, X_test):
    X_train_z = X_train.copy()
    X_test_z = X_test.copy()
    neuron_ids = set([col.split('_Bin')[0] for col in X_train.columns if '_Bin' in col])
    
    for nid in neuron_ids:
        cols_neuron = [col for col in X_train.columns if col.startswith(f"{nid}_Bin")]
        train_values = X_train[cols_neuron].values.flatten()
        
        mu = np.mean(train_values)
        sigma = np.std(train_values) if np.std(train_values) > 0 else 1e-8
            
        X_train_z[cols_neuron] = (X_train[cols_neuron] - mu) / sigma
        X_test_z[cols_neuron] = (X_test[cols_neuron] - mu) / sigma
        
    return X_train_z, X_test_z










def run_temporal_decoding_generic(df_run_model, decoding_type, start_time, bin_width, step, window_size_bins, num_folds=5, verbose=True):

    neuron_ids = list(set([col.split('_Bin')[0] for col in df_run_model.columns if '_Bin' in col]))
    total_bins = len([col for col in df_run_model.columns if col.startswith(f"{neuron_ids[0]}_Bin")])
    
    # Quantos "passos" cabem dentro de um bin? (ex: bin de 50ms, passo de 10ms -> offset = 5)
    offset = bin_width // step
    
    # Se window_size_bins=1, o offset não importa. Se for 4, ele sabe pular os bins sobrepostos.
    num_windows = total_bins - ((window_size_bins - 1) * offset)
    
    # 2. Mapeamento Temporal Dinâmico
    window_centers = []
    window_ends = []
    for w in range(num_windows):
        win_start_time = start_time + (w * step)
        win_end_time = win_start_time + (window_size_bins * bin_width)
        window_centers.append((win_start_time + win_end_time) / 2.0)
        window_ends.append(win_end_time)

    # 3. Definição Dinâmica do Tuning 
    # Em vez de pegar um bloco gigante, pegamos a janela exata que termina em 0ms (ou a mais próxima)
    target_tuning_time = 0 
    
    # Encontra o índice da janela cujo tempo final é o mais próximo de 0ms
    idx_tuning_window = np.argmin(np.abs(np.array(window_ends) - target_tuning_time))
    
    # Extrai estritamente a mesma quantidade de colunas que a janela deslizante vai usar
    offset = bin_width // step
    tuning_bins = [idx_tuning_window + (k * offset) for k in range(window_size_bins)]


    if decoding_type == 'CATEGORY':
        df_work = df_run_model[df_run_model['Y_cat'] != 'fix_point'].copy().reset_index(drop=True) # para decodificar categoria, ignorar sacadas para o FP
    else:
        df_work = df_run_model.copy().reset_index(drop=True)

    meta_cols = ['Y_cat', 'Y_equality', 'cat_anterior', 'transition', 'Y_is_cue', 'cat_cue', 'Y_attention']
    df_meta = df_work[meta_cols]

    transicao_unicos = df_meta['transition'].unique()
    atual_unicos = df_meta['Y_cat'].unique()
    anterior_unicos = df_meta['cat_anterior'].unique()
    cat_cue_unicos = df_meta['cat_cue'].unique()
    is_cue_unicos = df_meta['Y_is_cue'].unique()
    attention_unicos = df_meta['Y_attention'].unique()

    X = df_work.drop(columns=meta_cols)
    if decoding_type == 'CATEGORY':
        y = df_work['Y_cat']
    elif decoding_type == 'CHANGE_OF_CATEGORY':
        y = df_work['Y_equality']
    elif decoding_type == 'IS_CUE':
        y = df_work['Y_is_cue']
    elif decoding_type == 'CUE_CATEGORY':
        y = df_run_model['cat_cue']
    elif decoding_type == 'PREV_CATEGORY':
        y = df_run_model['cat_anterior']


    fold_transicao_accuracies = {item: np.full((num_folds, num_windows), np.nan) for item in transicao_unicos}
    fold_atual_accuracies = {item: np.full((num_folds, num_windows), np.nan) for item in atual_unicos}
    fold_anterior_accuracies = {item: np.full((num_folds, num_windows), np.nan) for item in anterior_unicos}
    fold_cat_cue_accuracies = {item: np.full((num_folds, num_windows), np.nan) for item in cat_cue_unicos}
    fold_is_cue_accuracies = {item: np.full((num_folds, num_windows), np.nan) for item in is_cue_unicos}
    fold_attention_accuracies = {item: np.full((num_folds, num_windows), np.nan) for item in attention_unicos}


    classes = np.unique(y)
    num_classes = len(classes)
    chance_level = (1.0 / num_classes) * 100

    outer_cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_accuracies = np.zeros((num_folds, num_windows))
    fold_cms = np.zeros((num_folds, num_windows, num_classes, num_classes))
    fold_f1 = np.zeros((num_folds, num_windows))
    fold_mcc = np.zeros((num_folds, num_windows))
    fold_gm = np.zeros((num_folds, num_windows))


    if verbose:
        print(f"Iniciando Decoding ({num_folds} folds, {num_windows} janelas, {total_bins} bins totais)...")
        print(f"Bins selecionados para Tuning: {tuning_bins}")

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        if verbose:
            print(f"--- Processando Fold {fold_idx + 1}/{num_folds} --- {pd.Timestamp.now().strftime('%H:%M:%S')}")
            
        X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Padronização
        X_train_z, X_test_z = zscore_per_neuron(X_train_raw, X_test_raw)
        
        # Tuning Pós-Sacádico Dinâmico
        # Usa o nome exato da coluna para evitar o bug do "endswith" 
        post_cols = [f"{nid}_Bin{b}" for nid in neuron_ids for b in tuning_bins]
        X_train_post = X_train_z[post_cols]
        
        search = RandomizedSearchCV(
            estimator=LinearSVC(class_weight='balanced', dual=False),
            param_distributions={'C': loguniform(1e-4, 1e2)},
            n_iter=15,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring='balanced_accuracy',
            random_state=42,
            n_jobs=-1
        )
        search.fit(X_train_post, y_train)
        best_C = search.best_params_['C']
        
        # Treino e Avaliação nas Moving Windows
        best_svm = LinearSVC(class_weight='balanced', C=best_C, dual=False, max_iter=2000)
        
        for w_idx in range(num_windows):
            # A mágica do offset: garante que ele pega os bins isolados corretos
            active_bins = [w_idx + (k * offset) for k in range(window_size_bins)]
            win_cols = [f"{nid}_Bin{b}" for nid in neuron_ids for b in active_bins]
            
            X_tr_win = X_train_z[win_cols]
            X_te_win = X_test_z[win_cols]
            
            best_svm.fit(X_tr_win, y_train)
            preds = best_svm.predict(X_te_win)
            
            # Acurácia
            acc = balanced_accuracy_score(y_test, preds)
            fold_accuracies[fold_idx, w_idx] = acc
            
            # Matriz de Confusão Normalizada
            cm = confusion_matrix(y_test, preds, labels=classes, normalize='true')
            fold_cms[fold_idx, w_idx] = cm

            # Suas métricas atuais
            acc = balanced_accuracy_score(y_test, preds)
            fold_accuracies[fold_idx, w_idx] = acc
            
            # NOVAS MÉTRICAS
            fold_f1[fold_idx, w_idx] = f1_score(y_test, preds, average='macro')
            mcc_raw = matthews_corrcoef(y_test, preds)
            fold_mcc[fold_idx, w_idx] = ((mcc_raw + 1) / 2)

            recalls = np.diag(cm)
            fold_gm[fold_idx, w_idx] = np.prod(recalls) ** (1.0 / num_classes)

            # AVALIANDO POR CATEGORIA
            meta_test = df_meta.iloc[test_idx]
            acertos = (preds == y_test.values).astype(int) 
            for transicao in transicao_unicos:
                mask_transicao = (meta_test['transition'] == transicao).values
                if np.sum(mask_transicao) > 0:
                    acuracia_transicao = np.mean(acertos[mask_transicao])
                    fold_transicao_accuracies[transicao][fold_idx, w_idx] = acuracia_transicao

            for atual in atual_unicos:
                mask_atual = (meta_test['Y_cat'] == atual).values
                if np.sum(mask_atual) > 0:
                    acuracia_atual = np.mean(acertos[mask_atual])
                    fold_atual_accuracies[atual][fold_idx, w_idx] = acuracia_atual

            for anterior in anterior_unicos:
                mask_anterior = (meta_test['cat_anterior'] == anterior).values
                if np.sum(mask_anterior) > 0:
                    acuracia_anterior = np.mean(acertos[mask_anterior])
                    fold_anterior_accuracies[anterior][fold_idx, w_idx] = acuracia_anterior

            for cat_cue in cat_cue_unicos:
                mask_cat_cue = (meta_test['cat_cue'] == cat_cue).values
                if np.sum(mask_cat_cue) > 0:
                    acuracia_cat_cue = np.mean(acertos[mask_cat_cue])
                    fold_cat_cue_accuracies[cat_cue][fold_idx, w_idx] = acuracia_cat_cue

            for is_cue in is_cue_unicos:
                mask_is_cue = (meta_test['Y_is_cue'] == is_cue).values
                if np.sum(mask_is_cue) > 0:
                    acuracia_is_cue = np.mean(acertos[mask_is_cue])
                    fold_is_cue_accuracies[is_cue][fold_idx, w_idx] = acuracia_is_cue

            for attention in attention_unicos:
                mask_attention = (meta_test['Y_attention'] == attention).values
                if np.sum(mask_attention) > 0:
                    acuracia_attention = np.mean(acertos[mask_attention])
                    fold_attention_accuracies[attention][fold_idx, w_idx] = acuracia_attention

    mean_acc = np.mean(fold_accuracies, axis=0) * 100
    sem_acc = (np.std(fold_accuracies, axis=0) / np.sqrt(num_folds)) * 100
    mean_cms = np.mean(fold_cms, axis=0) * 100
    mean_f1 = np.mean(fold_f1, axis=0) * 100
    mean_mcc = np.mean(fold_mcc, axis=0) * 100
    mean_gm = np.mean(fold_gm, axis=0) * 100
    mean_transition_accuracies = {}
    for transicao in transicao_unicos:
        mean_transition_accuracies[transicao] = np.nanmean(fold_transicao_accuracies[transicao], axis=0) * 100
    mean_atual_accuracies = {}
    for atual in atual_unicos:
        mean_atual_accuracies[atual] = np.nanmean(fold_atual_accuracies[atual], axis=0) * 100
    mean_anterior_accuracies = {}
    for anterior in anterior_unicos:
        mean_anterior_accuracies[anterior] = np.nanmean(fold_anterior_accuracies[anterior], axis=0) * 100
    mean_cat_cue_accuracies = {}
    for cat_cue in cat_cue_unicos:
        mean_cat_cue_accuracies[cat_cue] = np.nanmean(fold_cat_cue_accuracies[cat_cue], axis=0) * 100
    mean_is_cue_accuracies = {}
    for is_cue in is_cue_unicos:
        mean_is_cue_accuracies[is_cue] = np.nanmean(fold_is_cue_accuracies[is_cue], axis=0) * 100
    mean_attention_accuracies = {}
    for attention in attention_unicos:
        mean_attention_accuracies[attention] = np.nanmean(fold_attention_accuracies[attention], axis=0) * 100
    return {
        'mean_acc': mean_acc,
        'sem_acc': sem_acc,
        'mean_f1': mean_f1,
        'mean_mcc': mean_mcc,
        'mean_gm':mean_gm,
        'mean_acc_cat_atual': mean_atual_accuracies,
        'mean_acc_cat_anterior': mean_anterior_accuracies,
        'mean_acc_transition': mean_transition_accuracies,
        'mean_acc_is_cue': mean_is_cue_accuracies,
        'mean_acc_cat_cue': mean_cat_cue_accuracies,
        'mean_acc_attention': mean_attention_accuracies,
        'window_centers': np.array(window_centers),
        'window_ends': np.array(window_ends),
        'chance_level': chance_level,
        'fold_accuracies': fold_accuracies * 100,
        'mean_cms': mean_cms,
        'classes': classes
    }




def plot_temporal_confusion_matrix(result_decoding, target_time_ms,STR_DETAILS='',str_bins=''):
    """
    Plota a Matriz de Confusão para o instante de tempo (causal) mais próximo do solicitado.
    
    Parâmetros:
    -----------
    result_decoding : dict (retorno da função run_temporal_decoding)
    target_time_ms : int ou float (tempo em ms que você quer analisar, ex: 100)
    """
    
    # 1. Encontra o índice da janela temporal mais próxima do target_time_ms
    window_ends = result_decoding['window_ends']
    idx = np.argmin(np.abs(window_ends - target_time_ms))
    actual_time = window_ends[idx]
    
    # 2. Resgata a matriz e as classes exatas desse instante
    cm = result_decoding['mean_cms'][idx]
    classes = result_decoding['classes']
    
    # 3. Configura e plota o Heatmap
    plt.figure(figsize=(7, 6))
    
    # annot=True mostra os números, fmt='.1f' deixa 1 casa decimal, cmap é a paleta de cores
    sns.heatmap(cm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Acurácia (%)'},
                vmin=0, vmax=100)
    
    plt.title(f'{STR_DETAILS} | Matriz de Confusão aos {actual_time} ms | {str_bins} | Acurácia = {int(result_decoding["mean_acc"][list(result_decoding["window_ends"]).index(float(target_time_ms))])}%', fontsize=14, pad=15)
    # plt.title(f'{STR_DETAILS} | Matriz de Confusão (Fim da Janela: {actual_time} ms) (1 Bin de 50ms) | Acurácia = {int(result_decoding["mean_acc"][list(result_decoding["window_ends"]).index(float(target_time_ms))])}%', fontsize=14, pad=15)
    plt.ylabel('Classe Verdadeira', fontsize=12)
    plt.xlabel('Classe Prevista pelo Modelo', fontsize=12)
    
    # Rotaciona os rótulos para melhor leitura
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.show()



def plot_decoding(results, STR_DETAILS=''):
    plt.figure(figsize=(10, 5))
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Chegada da Sacada (t=0)')
    plt.axhline(y=results['chance_level'], color='black', linestyle=':', label=f"Chance Level ({results['chance_level']:.1f}%)")
    plt.plot(results['window_ends'], results['mean_acc'], color='dodgerblue', linewidth=2.5, marker='o', label='SVM Linear')
    plt.title(f'{STR_DETAILS} | Decodificabilidade (1 Bin de 50ms)', fontsize=14)
    plt.ylabel('Acurácia (%)', fontsize=12)
    plt.ylabel('Média Geométrica da Sensibilidade (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_decoding_by_class(results, class_name, classes, titulo_plot, STR_DETAILS='',txt_add_title='', chance_level=25):
    tempos = results['window_ends']
    plt.figure(figsize=(10,6))
    for c in classes:
        acc_temporal = results[class_name][c]
        plt.plot(tempos, acc_temporal, marker='o', label=c)
    plt.axvline(x=0, color='red', linestyle='--', label='Chegada da Sacada')
    plt.axhline(y=chance_level, color='black', linestyle='--', label='Chance level')
    plt.ylabel('Acurácia (%)')
    plt.xlabel('Tempo em relação à fixação (ms)')
    plt.legend()
    plt.title(f'{STR_DETAILS} {txt_add_title} | {titulo_plot}')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_temporal_confusion_matrix(result_decoding, target_time_ms,STR_DETAILS='',str_bins=''):
    """
    Plota a Matriz de Confusão para o instante de tempo (causal) mais próximo do solicitado.
    
    Parâmetros:
    -----------
    result_decoding : dict (retorno da função run_temporal_decoding)
    target_time_ms : int ou float (tempo em ms que você quer analisar, ex: 100)
    """
    
    # 1. Encontra o índice da janela temporal mais próxima do target_time_ms
    window_ends = result_decoding['window_ends']
    idx = np.argmin(np.abs(window_ends - target_time_ms))
    actual_time = window_ends[idx]
    
    # 2. Resgata a matriz e as classes exatas desse instante
    cm = result_decoding['mean_cms'][idx]
    classes = result_decoding['classes']
    
    # 3. Configura e plota o Heatmap
    plt.figure(figsize=(7, 6))
    
    # annot=True mostra os números, fmt='.1f' deixa 1 casa decimal, cmap é a paleta de cores
    sns.heatmap(cm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Acurácia (%)'},
                vmin=0, vmax=100)
    
    plt.title(f'{STR_DETAILS} | Matriz de Confusão aos {actual_time} ms | {str_bins} | Acurácia = {int(result_decoding["mean_acc"][list(result_decoding["window_ends"]).index(float(target_time_ms))])}%', fontsize=14, pad=15)
    # plt.title(f'{STR_DETAILS} | Matriz de Confusão (Fim da Janela: {actual_time} ms) (1 Bin de 50ms) | Acurácia = {int(result_decoding["mean_acc"][list(result_decoding["window_ends"]).index(float(target_time_ms))])}%', fontsize=14, pad=15)
    plt.ylabel('Classe Verdadeira', fontsize=12)
    plt.xlabel('Classe Prevista pelo Modelo', fontsize=12)
    
    # Rotaciona os rótulos para melhor leitura
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.show()

def plot_multiple_CM(res, list_target_times, STR_DETAILS='',str_bins=''):
    for target_time in list_target_times:
        plot_temporal_confusion_matrix(res, target_time_ms=target_time,STR_DETAILS=STR_DETAILS,str_bins=str_bins)

