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
        trial_path = f'{path_destiny}\{neuron_id}.mat'
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